"""Sealed Entropy extension state machine. Historical experiments are read-only.

Only A supplies policy checkpoints. Acquisition and training are separate jobs;
the integration namespace cannot execute any cell beyond GC10 a00 R1 A/B.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import uuid

from .entropy import (EntropyIntegrityError, entropy_acquisition, file_sha256,
                      stable_sha256, validate_pool)


class ExtensionGuardError(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json_new(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')


def checked_file(path, expected):
    path = Path(path)
    if not path.is_file() or file_sha256(path) != expected:
        raise ExtensionGuardError(f'file seal mismatch: {path}')
    return path


def inside(base, path):
    base, path = Path(base).resolve(), Path(path).resolve()
    if path == base or base not in path.parents:
        raise ExtensionGuardError('output path must remain strictly inside extension')
    return path


def trajectory_id(dataset, index):
    if dataset not in ('GC10-DET', 'NEU-DET') or type(index) is not int or index not in range(10):
        raise ExtensionGuardError('invalid trajectory')
    return f'{dataset}_a{index:02d}'


def query_id(trajectory, round_idx):
    if type(round_idx) is not int or round_idx not in range(1, 6):
        raise ExtensionGuardError('only R1-R5 acquisitions exist')
    return f'{trajectory}_Q_R{round_idx}'


def cell_id(trajectory, round_idx, slot):
    query_id(trajectory, round_idx)
    if slot not in ('A', 'B'):
        raise ExtensionGuardError('invalid learner slot')
    return f'{trajectory}_Entropy_{slot}_R{round_idx}'


def build_plan(seed_rows, budgets):
    cells, queries = [], []
    for dataset in ('GC10-DET', 'NEU-DET'):
        rows = sorted((r for r in seed_rows if r['dataset'] == dataset), key=lambda r: int(r['acquisition_index']))
        if [int(r['acquisition_index']) for r in rows] != list(range(10)):
            raise ExtensionGuardError('exactly ten acquisition indices required per dataset')
        for row in rows:
            tid = trajectory_id(dataset, int(row['acquisition_index']))
            for round_idx in range(1, 6):
                qid = query_id(tid, round_idx)
                parent = f'{tid}_SHARED_A_R0' if round_idx == 1 else cell_id(tid, round_idx - 1, 'A')
                queries.append({'node_id': qid, 'kind': 'acquisition', 'trajectory': tid,
                                'dataset': dataset, 'acquisition_index': int(row['acquisition_index']),
                                'round': round_idx, 'slot': None, 'depends_on': [parent],
                                'policy_slot': 'A', 'query_size': budgets[dataset]['query_count_per_round']})
                for slot in ('A', 'B'):
                    cells.append({'node_id': cell_id(tid, round_idx, slot), 'kind': 'training',
                                  'trajectory': tid, 'dataset': dataset,
                                  'acquisition_index': int(row['acquisition_index']),
                                  'acquisition_seed': int(row['acquisition_seed']),
                                  'round': round_idx, 'slot': slot,
                                  'training_seed': int(row[f'training_seed_{slot.lower()}']),
                                  'labeled_count': budgets[dataset]['cumulative_counts'][round_idx],
                                  'depends_on': [qid], 'policy_participant': slot == 'A'})
    if len(cells) != 200 or len(queries) != 100:
        raise ExtensionGuardError('invalid extension registry cardinality')
    return cells, queries


def require_training_authorization(payload, freeze_sha, mode, node_id):
    if payload.get('status') != 'AUTHORIZED' or payload.get('freeze_sha256') != freeze_sha:
        raise ExtensionGuardError('training authorization absent or freeze mismatch')
    if payload.get('locked_test_access') is not False or payload.get('mode') != mode:
        raise ExtensionGuardError('authorization scope mismatch')
    if node_id not in payload.get('allowed_cells', []):
        raise ExtensionGuardError('cell outside authorized scope')
    if mode == 'integration' and set(payload['allowed_cells']) != {
        'GC10-DET_a00_Entropy_A_R1', 'GC10-DET_a00_Entropy_B_R1'
    }:
        raise ExtensionGuardError('integration permits exactly GC10 a00 R1 A/B')


def deny_locked_test(*args, **kwargs):
    raise ExtensionGuardError('locked-test access is unconditionally forbidden in this extension')


class Extension:
    def __init__(self, root, mode='official'):
        self.root = Path(root).resolve()
        if mode not in ('official', 'integration'):
            raise ExtensionGuardError('unknown execution namespace')
        self.mode = mode
        self.freeze = read_json(self.root / 'freeze.json')
        self.freeze_sha = file_sha256(self.root / 'freeze.json')
        for item in self.freeze['files']:
            checked_file(item['path'], item['sha256'])
        self.protocol = read_json(self.root / 'protocol.json')
        if self.protocol['policy_learner'] != 'A' or self.protocol['locked_test_access'] != 'FORBIDDEN':
            raise ExtensionGuardError('policy/final boundary mismatch')
        self.plan = read_json(self.root / 'planned_cells.json')
        self.queries = read_json(self.root / 'planned_acquisitions.json')
        self.all_nodes = {n['node_id']: n for n in self.plan + self.queries}
        expected_cells, expected_queries = build_plan(self.protocol['seeds'], self.protocol['budgets'])
        if self.plan != expected_cells or self.queries != expected_queries:
            raise ExtensionGuardError('plan differs from frozen seed/budget policy')
        self.r0 = read_json(self.root / 'historical_r0.json')
        self.db = inside(self.root, self.root / 'runtime' / mode / 'registry.sqlite')

    def node(self, node_id):
        if node_id not in self.all_nodes:
            raise ExtensionGuardError('unknown extension node')
        node = self.all_nodes[node_id]
        if self.mode == 'integration' and (node['trajectory'], node['round']) != ('GC10-DET_a00', 1):
            raise ExtensionGuardError('outside integration GC10 a00 R1 boundary')
        return node

    @contextmanager
    def connection(self):
        c = sqlite3.connect(self.db, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            yield c
        finally:
            c.close()

    def initialize_registry(self):
        self.db.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as c:
            c.execute('CREATE TABLE IF NOT EXISTS jobs (node_id TEXT PRIMARY KEY, state TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 0, artifact TEXT, artifact_sha256 TEXT)')
            c.execute('CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, node_id TEXT, state TEXT, timestamp TEXT)')
            c.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)')
            old = c.execute("SELECT value FROM metadata WHERE key='freeze_sha256'").fetchone()
            if old and old['value'] != self.freeze_sha:
                raise ExtensionGuardError('registry belongs to another freeze')
            c.execute("INSERT OR IGNORE INTO metadata VALUES ('freeze_sha256', ?)", (self.freeze_sha,))
            nodes = [n for n in self.all_nodes.values() if self.mode == 'official' or (n['trajectory'], n['round']) == ('GC10-DET_a00', 1)]
            for n in nodes:
                state = 'PENDING' if n['kind'] == 'acquisition' else 'BLOCKED_MANIFEST'
                c.execute('INSERT OR IGNORE INTO jobs(node_id,state) VALUES (?,?)', (n['node_id'], state))
            actual = {r[0] for r in c.execute('SELECT node_id FROM jobs')}
            if actual != {n['node_id'] for n in nodes}:
                raise ExtensionGuardError('runtime registry has unexpected nodes')
            c.commit()

    def state(self, node_id):
        self.node(node_id)
        with self.connection() as c:
            row = c.execute('SELECT * FROM jobs WHERE node_id=?', (node_id,)).fetchone()
        if not row:
            raise ExtensionGuardError('uninitialized node')
        return dict(row)

    def _claim(self, node_id, expected):
        self.node(node_id)
        with self.connection() as c:
            c.execute('BEGIN IMMEDIATE')
            row = c.execute('SELECT state,attempt FROM jobs WHERE node_id=?', (node_id,)).fetchone()
            if not row or row['state'] != expected:
                raise ExtensionGuardError('node already claimed, failed, or dependency unsatisfied')
            attempt = row['attempt'] + 1
            if attempt > 3:
                raise ExtensionGuardError('maximum infrastructure attempts exceeded')
            c.execute("UPDATE jobs SET state='RUNNING',attempt=? WHERE node_id=?", (attempt, node_id))
            c.execute('INSERT INTO events(node_id,state,timestamp) VALUES (?,?,?)', (node_id, 'RUNNING', now()))
            c.commit()
        return attempt

    def _complete(self, node_id, artifact):
        node = self.node(node_id)
        artifact = inside(self.root, artifact)
        digest = file_sha256(artifact)
        with self.connection() as c:
            c.execute('BEGIN IMMEDIATE')
            changed = c.execute("UPDATE jobs SET state='COMPLETE',artifact=?,artifact_sha256=? WHERE node_id=? AND state='RUNNING'",
                                (str(artifact), digest, node_id)).rowcount
            if changed != 1:
                raise ExtensionGuardError('completion without a RUNNING claim')
            if node['kind'] == 'acquisition':
                for slot in ('A', 'B'):
                    cid = cell_id(node['trajectory'], node['round'], slot)
                    c.execute("UPDATE jobs SET state='READY' WHERE node_id=? AND state='BLOCKED_MANIFEST'", (cid,))
            c.execute('INSERT INTO events(node_id,state,timestamp) VALUES (?,?,?)', (node_id, 'COMPLETE', now()))
            c.commit()

    def fail(self, node_id):
        with self.connection() as c:
            c.execute("UPDATE jobs SET state='FAILED' WHERE node_id=? AND state='RUNNING'", (node_id,))
            c.execute('INSERT INTO events(node_id,state,timestamp) VALUES (?,?,?)', (node_id, 'FAILED', now()))
            c.commit()

    def retry_failed(self, node_id, authorization=None):
        node = self.node(node_id)
        if node['kind'] == 'training' or self.mode == 'official':
            target = node_id if node['kind'] == 'training' else cell_id(node['trajectory'], node['round'], 'A')
            self.verify_authorization(authorization, target)
        ready = 'READY' if node['kind'] == 'training' else 'PENDING'
        with self.connection() as c:
            c.execute('BEGIN IMMEDIATE')
            row = c.execute('SELECT state,attempt FROM jobs WHERE node_id=?', (node_id,)).fetchone()
            if not row or row['state'] != 'FAILED' or row['attempt'] >= 3:
                raise ExtensionGuardError('only FAILED jobs below the fixed attempt limit can be retried')
            c.execute('UPDATE jobs SET state=? WHERE node_id=?', (ready, node_id))
            c.execute('INSERT INTO events(node_id,state,timestamp) VALUES (?,?,?)', (node_id,'RETRY_REQUESTED',now()))
            c.commit()

    def completed(self, node_id):
        row = self.state(node_id)
        if row['state'] != 'COMPLETE':
            raise ExtensionGuardError(f'parent not COMPLETE: {node_id}')
        path = inside(self.root, row['artifact'])
        checked_file(path, row['artifact_sha256'])
        payload = read_json(path)
        if payload['node_id'] != node_id or payload['freeze_sha256'] != self.freeze_sha:
            raise ExtensionGuardError('artifact identity/freeze mismatch')
        for entry in payload['files']:
            checked_file(inside(self.root, entry['path']), entry['sha256'])
        return payload

    def policy_checkpoint(self, trajectory, target_round):
        query_id(trajectory, target_round)
        if target_round == 1:
            ref = self.r0[f'{trajectory}_SHARED_A_R0']
            checked_file(ref['best_path'], ref['best_sha256'])
            checked_file(ref['manifest_path'], ref['manifest_sha256'])
            if ref['slot'] != 'A' or ref['round'] != 0 or ref['trajectory'] != trajectory:
                raise ExtensionGuardError('historical policy reference mismatch')
            return {'path': ref['best_path'], 'sha256': ref['best_sha256'],
                    'node_id': f'{trajectory}_SHARED_A_R0', 'slot': 'A', 'round': 0}
        # No B lookup: only previous A can be a policy parent.
        result = self.completed(cell_id(trajectory, target_round - 1, 'A'))
        if result['slot'] != 'A' or result['round'] != target_round - 1 or result['trajectory'] != trajectory:
            raise ExtensionGuardError('policy checkpoint role/round mismatch')
        checked_file(result['best_path'], result['best_sha256'])
        return {'path': result['best_path'], 'sha256': result['best_sha256'],
                'node_id': result['node_id'], 'slot': 'A', 'round': target_round - 1}

    def prior_labeled(self, trajectory, target_round):
        if target_round == 1:
            ref = self.r0[f'{trajectory}_SHARED_A_R0']
            return checked_file(ref['manifest_path'], ref['manifest_sha256']).read_text().splitlines()
        result = self.completed(query_id(trajectory, target_round - 1))
        return Path(result['manifest_path']).read_text().splitlines()

    def dataset(self, name):
        return read_json(self.root / 'datasets' / f'{name}.json')

    def verify_authorization(self, authorization, node_id):
        payload = read_json(authorization)
        require_training_authorization(payload, self.freeze_sha, self.mode, node_id)
        if self.mode == 'official':
            if set(payload['allowed_cells']) != {n['node_id'] for n in self.plan}:
                raise ExtensionGuardError('official authorization must match exact 200-cell registry')
            proof = checked_file(payload['integration_pass_path'], payload['integration_pass_sha256'])
            passed = read_json(proof)
            if passed.get('status') != 'PASS' or passed.get('freeze_sha256') != self.freeze_sha:
                raise ExtensionGuardError('real integration PASS required before official run')
            integration = Extension(self.root, 'integration')
            for slot in ('A', 'B'):
                result = integration.completed(f'GC10-DET_a00_Entropy_{slot}_R1')
                if result.get('epochs_completed') != 100 or result.get('cold_start') is not True:
                    raise ExtensionGuardError('real completed integration cells required')
        return payload

    def acquire(self, dataset, index, round_idx):
        tid = trajectory_id(dataset, index)
        qid = query_id(tid, round_idx)
        self.node(qid)
        if self.mode == 'official':
            raise ExtensionGuardError('official acquisition requires run_authorized_acquire')
        return self._acquire(qid)

    def run_authorized_acquire(self, dataset, index, round_idx, authorization):
        tid = trajectory_id(dataset, index)
        self.verify_authorization(authorization, cell_id(tid, round_idx, 'A'))
        return self._acquire(query_id(tid, round_idx))

    def _acquire(self, qid):
        node = self.node(qid)
        if self.state(qid)['state'] == 'COMPLETE':
            return self.completed(qid)
        policy = self.policy_checkpoint(node['trajectory'], node['round'])
        labeled = self.prior_labeled(node['trajectory'], node['round'])
        data = self.dataset(node['dataset'])
        train = data['train_ids']
        records = [{'image_id': i, 'image_path': data['images'][i]['path']}
                   for i in train if i not in set(labeled)]
        validate_pool(records, labeled, train)
        for row in records:
            checked_file(row['image_path'], data['images'][row['image_id']]['sha256'])
        attempt = self._claim(qid, 'PENDING')
        output = inside(self.root, self.root / 'acquisitions' / self.mode / qid / f'attempt_{attempt:02d}')
        output.mkdir(parents=True, exist_ok=False)
        try:
            scores = entropy_acquisition(policy['path'], records, node['query_size'], self.protocol['inference'],
                                         labeled_ids=labeled, train_ids=train, checkpoint_sha256=policy['sha256'])
            if set(scores['selected_ids']) & set(data['validation_ids']):
                raise ExtensionGuardError('validation ID in query')
            expected = self.protocol['budgets'][node['dataset']]['cumulative_counts'][node['round']]
            if len(scores['next_labeled_ids']) != expected:
                raise ExtensionGuardError('cumulative budget mismatch')
            # Recheck source bytes after scoring, before sealing the selected IDs.
            for row in records:
                checked_file(row['image_path'], data['images'][row['image_id']]['sha256'])
            score_path = output / 'scores.json'
            write_json_new(score_path, scores)
            manifest_path = output / 'labeled_ids.txt'
            manifest_path.write_text('\n'.join(scores['next_labeled_ids']) + '\n', encoding='ascii', newline='\n')
            query_path = output / 'query_ids.txt'
            query_path.write_text('\n'.join(sorted(scores['selected_ids'])) + '\n', encoding='ascii', newline='\n')
            payload = {**node, 'freeze_sha256': self.freeze_sha, 'timestamp': now(), 'policy_checkpoint': policy,
                       'manifest_path': str(manifest_path), 'manifest_sha256': file_sha256(manifest_path),
                       'score_hashes': scores['hashes'], 'labeled_count': expected, 'gt_used_for_selection': False,
                       'files': [{'path': str(p), 'sha256': file_sha256(p)} for p in [score_path, manifest_path, query_path]]}
            write_json_new(output / 'acquisition.json', payload)
            self._complete(qid, output / 'acquisition.json')
            return payload
        except BaseException:
            self.fail(qid)
            raise

    def _copy_sealed(self, source, expected, destination):
        source = checked_file(source, expected)
        destination = inside(self.root, destination)
        if destination.exists():
            return checked_file(destination, expected)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f'.{uuid.uuid4().hex}.tmp')
        shutil.copyfile(source, temporary)  # Separate bytes: never a historical hardlink.
        checked_file(temporary, expected)
        temporary.replace(destination)
        return destination

    def prepare_training(self, node_id):
        node = self.node(node_id)
        if node['kind'] != 'training':
            raise ExtensionGuardError('training cell required')
        acquisition = self.completed(node['depends_on'][0])
        data = self.dataset(node['dataset'])
        ids = Path(acquisition['manifest_path']).read_text().splitlines()
        if len(ids) != node['labeled_count'] or set(ids) - set(data['train_ids']):
            raise ExtensionGuardError('training membership mismatch')
        if set(ids) & set(data['validation_ids']):
            raise ExtensionGuardError('train/validation overlap')
        # Copy labels only after query/manifest sealing; these inputs never enter scoring.
        cache = inside(self.root, self.root / 'data' / self.mode / node['dataset'])
        for image_id in sorted(set(ids) | set(data['validation_ids'])):
            image = data['images'][image_id]
            label = data['labels'][image_id]
            self._copy_sealed(image['path'], image['sha256'], cache / 'images' / f'{image_id}.jpg')
            self._copy_sealed(label['path'], label['sha256'], cache / 'labels' / f'{image_id}.txt')
        shared = inside(self.root, self.root / 'training_inputs' / self.mode / node['depends_on'][0])
        shared.mkdir(parents=True, exist_ok=True)
        for name, values in [('train', ids), ('validation', data['validation_ids'])]:
            text = '\n'.join((cache / 'images' / f'{i}.jpg').as_posix() for i in sorted(values)) + '\n'
            path = shared / f'{name}.txt'
            if path.exists() and path.read_text() != text:
                raise ExtensionGuardError('shared A/B data list drift')
            if not path.exists():
                path.write_text(text, encoding='utf-8', newline='\n')
        import yaml
        dataset_config = {'path': cache.as_posix(), 'train': (shared / 'train.txt').as_posix(),
                          'val': (shared / 'validation.txt').as_posix(), 'names': data['names']}
        yaml_path = shared / 'dataset.yaml'
        text = yaml.safe_dump(dataset_config, sort_keys=False)
        if yaml_path.exists() and yaml_path.read_text() != text:
            raise ExtensionGuardError('shared A/B YAML drift')
        if not yaml_path.exists():
            yaml_path.write_text(text, encoding='utf-8', newline='\n')
        config = read_json(self.root / 'training_config.json')
        config.update(data=str(yaml_path), seed=node['training_seed'])
        payload = {**node, 'freeze_sha256': self.freeze_sha,
                   'manifest_path': acquisition['manifest_path'], 'manifest_sha256': acquisition['manifest_sha256'],
                   'data_yaml_path': str(yaml_path), 'data_yaml_sha256': file_sha256(yaml_path),
                   'train_list_sha256': file_sha256(shared / 'train.txt'),
                   'validation_list_sha256': file_sha256(shared / 'validation.txt'),
                   'training_config': config, 'training_config_sha256': stable_sha256(config),
                   'initialization_path': self.protocol['pretrained']['path'],
                   'initialization_sha256': self.protocol['pretrained']['sha256']}
        target = shared / f"{node['slot']}_spec.json"
        if target.exists():
            if read_json(target) != payload:
                raise ExtensionGuardError('sealed cell spec drift')
        else:
            write_json_new(target, payload)
        return payload
