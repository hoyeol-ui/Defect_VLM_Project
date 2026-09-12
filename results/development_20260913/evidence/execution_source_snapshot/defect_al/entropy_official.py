"""Official release overlay: immutable parent engine, scoped authorization and round gates."""
import math
from pathlib import Path

from .entropy import file_sha256, stable_sha256
from .entropy_extension import (Extension, ExtensionGuardError, checked_file, read_json,
                               write_json_new, now, cell_id, query_id)


def require_scope(payload, official_sha, engine_sha, plan, queries, node_id):
    if (payload.get('status') != 'AUTHORIZED' or payload.get('mode') != 'official'
            or payload.get('official_freeze_sha256') != official_sha
            or payload.get('freeze_sha256') != engine_sha
            or payload.get('locked_test_access') is not False
            or payload.get('integration_reuse') is not False):
        raise ExtensionGuardError('official authorization seal/policy mismatch')
    tids = payload.get('authorized_trajectories', [])
    if not tids or len(tids) != len(set(tids)) or set(tids) - {n['trajectory'] for n in plan}:
        raise ExtensionGuardError('invalid authorized trajectories')
    cells = {n['node_id'] for n in plan if n['trajectory'] in tids}
    acquisitions = {n['node_id'] for n in queries if n['trajectory'] in tids}
    if (set(payload.get('allowed_cells', [])) != cells or len(payload['allowed_cells']) != len(cells)
            or set(payload.get('allowed_queries', [])) != acquisitions
            or len(payload['allowed_queries']) != len(acquisitions)):
        raise ExtensionGuardError('authorization must contain complete fixed trajectories')
    if node_id not in cells | acquisitions:
        raise ExtensionGuardError('node outside authorized tranche')


def verify_checkpoint_selection(best, last):
    history = last['train_results']
    if len(history['epoch']) != 100 or int(history['epoch'][-1]) != 100:
        raise ExtensionGuardError('checkpoint history is not 100 epochs')
    values = [float(v) for v in history['metrics/mAP50-95(B)']]
    if (not all(math.isfinite(v) for v in values)
            or not math.isclose(float(best['train_metrics']['fitness']), max(values), rel_tol=0, abs_tol=1e-5)
            or not math.isclose(float(last['train_metrics']['fitness']), values[-1], rel_tol=0, abs_tol=1e-5)
            or best['train_results'] != last['train_results']):
        raise ExtensionGuardError('validation-best checkpoint inconsistency')


class OfficialExtension(Extension):
    def __init__(self, root, mode='official'):
        if mode != 'official':
            raise ExtensionGuardError('official namespace required')
        super().__init__(root, mode)
        self.release_root = self.root / 'official_release_v1'
        self.release = read_json(self.release_root / 'freeze.json')
        self.official_sha = file_sha256(self.release_root / 'freeze.json')
        if self.release['engine_freeze_sha256'] != self.freeze_sha:
            raise ExtensionGuardError('official release engine mismatch')
        for entry in self.release['files']:
            checked_file(entry['path'], entry['sha256'])
        self.active_authorization = None

    def verify_authorization(self, authorization, node_id):
        payload = read_json(authorization)
        require_scope(payload, self.official_sha, self.freeze_sha, self.plan, self.queries, node_id)
        checked_file(payload['integration_pass_path'], payload['integration_pass_sha256'])
        proof = read_json(payload['integration_pass_path'])
        if proof['status'] != 'PASS' or proof['freeze_sha256'] != self.freeze_sha:
            raise ExtensionGuardError('integration proof mismatch')
        self.active_authorization = Path(authorization).resolve()
        return payload

    def completed(self, node_id):
        result = super().completed(node_id)
        prefix = 'outcomes_private' if self.node(node_id)['kind'] == 'training' else 'acquisitions'
        expected = self.root / prefix / 'official' / node_id
        if expected not in Path(self.state(node_id)['artifact']).resolve().parents:
            raise ExtensionGuardError('official artifact cannot refer to integration namespace')
        if result.get('official_freeze_sha256') != self.official_sha:
            raise ExtensionGuardError('official artifact release mismatch')
        checked_file(result['official_authorization_path'], result['official_authorization_sha256'])
        return result

    def _claim(self, node_id, expected):
        if self.active_authorization is None:
            raise ExtensionGuardError('authorization required before any official claim')
        self.verify_authorization(self.active_authorization, node_id)
        node = self.node(node_id)
        if node['round'] > 1:
            self.validate_gate(node['trajectory'], node['round'] - 1)
        return super()._claim(node_id, expected)

    def _complete(self, node_id, artifact):
        if self.active_authorization is None:
            raise ExtensionGuardError('authorized lifecycle required')
        result = read_json(artifact)
        result.update(official_freeze_sha256=self.official_sha,
                      official_authorization_path=str(self.active_authorization),
                      official_authorization_sha256=file_sha256(self.active_authorization))
        if self.node(node_id)['kind'] == 'acquisition':
            result['inference_config'] = self.protocol['inference']
            result['inference_config_sha256'] = stable_sha256(self.protocol['inference'])
        # Newly created runtime artifacts only; no frozen or historical file is updated.
        import json
        Path(artifact).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        return super()._complete(node_id, artifact)

    def gate_path(self, trajectory, round_idx):
        return self.root / 'runtime/official/round_gates' / f'{trajectory}_R{round_idx}.json'

    def validate_gate(self, trajectory, round_idx):
        path = self.gate_path(trajectory, round_idx)
        if not path.is_file():
            raise ExtensionGuardError('previous round gate absent or invalid')
        gate = read_json(path)
        if (gate.get('status') != 'PASS' or gate.get('official_freeze_sha256') != self.official_sha
                or gate['trajectory'] != trajectory or gate['round'] != round_idx):
            raise ExtensionGuardError('previous round gate absent or invalid')
        for item in gate['files']:
            checked_file(item['path'], item['sha256'])
        for slot in ('A', 'B'):
            self.completed(cell_id(trajectory, round_idx, slot))
        return gate

    def historical_integrity(self):
        repo = self.root.parent
        protected = read_json(self.root / 'historical_guard.json')
        for path, digest in protected.items():
            checked_file(repo / path, digest)
        # Historical analysis/test artifacts are protected using metadata only;
        # their contents are never opened, evaluated, or interpreted by this run.
        metadata = read_json(self.release_root / 'protected_metadata.json')
        for item in metadata:
            stat = Path(item['path']).stat()
            if stat.st_size != item['size'] or stat.st_mtime_ns != item['mtime_ns']:
                raise ExtensionGuardError('protected artifact metadata changed')
        return len(protected), len(metadata)

    def audit_round(self, trajectory, round_idx):
        import torch
        import yaml
        qid = query_id(trajectory, round_idx)
        acquisition = self.completed(qid)
        dataset = self.dataset(acquisition['dataset'])
        train, validation = set(dataset['train_ids']), set(dataset['validation_ids'])
        if train & validation or set(dataset['images']) != train | validation:
            raise ExtensionGuardError('sealed input split mismatch')
        scores = read_json(next(f['path'] for f in acquisition['files'] if f['path'].endswith('scores.json')))
        selected = scores['selected_ids']
        previous = set(self.prior_labeled(trajectory, round_idx))
        current = Path(acquisition['manifest_path']).read_text().splitlines()
        if (len(selected) != acquisition['query_size'] or len(set(selected)) != len(selected)
                or previous & set(selected) or not set(selected) <= train
                or set(current) != previous | set(selected) or len(set(current)) != len(current)
                or set(current) & validation):
            raise ExtensionGuardError('query duplicate/reselection/leakage mismatch')
        expected_parent = self.policy_checkpoint(trajectory, round_idx)
        if acquisition['policy_checkpoint'] != expected_parent:
            raise ExtensionGuardError('query parent mismatch')
        results, files = [], []
        for slot in ('A', 'B'):
            cid = cell_id(trajectory, round_idx, slot)
            result = self.completed(cid)
            spec_path = self.root / 'training_inputs/official' / qid / f'{slot}_spec.json'
            spec = read_json(spec_path)
            node = self.node(cid)
            if (result['training_seed'] != node['training_seed'] or result['epochs_completed'] != 100
                    or result['cold_start'] is not True or result['resume'] is not False
                    or result['initialization_sha256'] != self.protocol['pretrained']['sha256']
                    or result['manifest_sha256'] != acquisition['manifest_sha256']
                    or result['training_config_sha256'] != stable_sha256(spec['training_config'])
                    or result['checkpoint_selection_split'] != 'validation' or result['locked_test_accessed'] is not False):
                raise ExtensionGuardError('cell seed/cold-start/config/manifest mismatch')
            checked_file(result['initialization_path'], result['initialization_sha256'])
            checked_file(spec['data_yaml_path'], spec['data_yaml_sha256'])
            data = yaml.safe_load(Path(spec['data_yaml_path']).read_text())
            if set(data) != {'path', 'train', 'val', 'names'}:
                raise ExtensionGuardError('test YAML forbidden')
            train_paths = [Path(p) for p in Path(data['train']).read_text().splitlines()]
            val_paths = [Path(p) for p in Path(data['val']).read_text().splitlines()]
            if (len(train_paths) != node['labeled_count'] or {p.stem for p in train_paths} != set(current)
                    or len(val_paths) != len(validation) or {p.stem for p in val_paths} != validation):
                raise ExtensionGuardError('training/validation count mismatch')
            for path in train_paths + val_paths:
                if self.root / 'data/official' not in path.resolve().parents:
                    raise ExtensionGuardError('input outside official data namespace')
                checked_file(path, dataset['images'][path.stem]['sha256'])
                checked_file(path.parent.parent / 'labels' / (path.stem + '.txt'), dataset['labels'][path.stem]['sha256'])
            best = torch.load(result['best_path'], map_location='cpu', weights_only=False)
            last = torch.load(result['last_path'], map_location='cpu', weights_only=False)
            verify_checkpoint_selection(best, last)
            for checkpoint in (best, last):
                if (checkpoint['train_args']['seed'] != node['training_seed']
                        or checkpoint['train_args']['resume'] is not False
                        or checkpoint['train_args']['model'] != result['initialization_path']
                        or checkpoint['train_args']['data'] != spec['data_yaml_path']):
                    raise ExtensionGuardError('checkpoint metadata mismatch')
            del best, last
            results.append({k: result[k] for k in ['node_id', 'training_seed', 'best_sha256', 'last_sha256',
                'manifest_sha256', 'initialization_sha256', 'training_config_sha256', 'train_runtime_seconds',
                'validation_runtime_seconds', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes']})
            files += result['files']
            for path in [Path(self.state(cid)['artifact']), spec_path, Path(spec['data_yaml_path']), Path(data['train']), Path(data['val'])]:
                files.append({'path': str(path), 'sha256': file_sha256(path)})
        next_parent = self.policy_checkpoint(trajectory, round_idx + 1) if round_idx < 5 else None
        if next_parent and (next_parent['slot'] != 'A' or next_parent['sha256'] != results[0]['best_sha256']):
            raise ExtensionGuardError('next policy parent is not learner A')
        if any(q['policy_slot'] != 'A' or any('_B_' in p for p in q['depends_on']) for q in self.queries):
            raise ExtensionGuardError('learner B in policy graph')
        protected, metadata = self.historical_integrity()
        files += acquisition['files']
        files.append({'path': self.state(qid)['artifact'], 'sha256': self.state(qid)['artifact_sha256']})
        gate = {'status': 'PASS', 'timestamp': now(), 'official_freeze_sha256': self.official_sha,
                'trajectory': trajectory, 'round': round_idx, 'train_count': len(current),
                'validation_count': len(validation), 'query_size': len(selected), 'cells': results,
                'previous_policy_parent': expected_parent, 'next_policy_parent': next_parent,
                'validation_best_consistency': 'PASS', 'locked_test_accesses': 0, 'leakage': 0,
                'duplicates': 0, 'reselections': 0, 'learner_B_policy_edges': 0,
                'historical_hashes_unchanged': protected, 'protected_metadata_unchanged': metadata, 'files': files}
        target = self.gate_path(trajectory, round_idx)
        if target.exists():
            return self.validate_gate(trajectory, round_idx)
        write_json_new(target, gate)
        return gate
