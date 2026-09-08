"""Prepare entropy inference manifests only. Never train or evaluate a detector."""
from pathlib import Path
import csv
import json
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from defect_al.entropy import PINNED_INFERENCE, file_sha256


def main():
    report = ROOT / 'reports/entropy_preflight_20260908'
    audit = json.loads((report / 'repository_audit.json').read_text())
    parent = yaml.safe_load((ROOT / 'confirmatory_v1/confirmatory_protocol_v1.yaml').read_text(encoding='utf-8'))
    freeze = ROOT / 'confirmatory_v1/data_freeze'
    docs = ROOT / 'docs/design'
    docs.mkdir(parents=True, exist_ok=True)
    config = {**PINNED_INFERENCE,
              'software_version': audit['environment']['packages'],
              'source_sha256': audit['environment']['source_sha256']}
    manifest = {
        'protocol_id': 'basic-detection-entropy-yolov8-v1',
        'status': 'SMOKE_ONLY_OFFICIAL_BLOCKED',
        'execution_authority': {'inference_smoke': True, 'training': False, 'locked_test_evaluation': False},
        'official_blockers': [
            'Learner A/B checkpoint to acquisition trajectory coupling is not specified in the design.',
            'Legacy training and final-evaluation guards are sealed to 440 Random/DINO cells; no entropy integration exists.'
        ],
        'definition': {'p': 'post_NMS_Results.boxes.conf', 'entropy': '-p*ln(p)-(1-p)*ln(1-p)',
                       'aggregation': 'sum', 'no_detection': 0.0, 'class_weighting': False,
                       'endpoint_policy': 'exact limits at 0 and 1; no interior clipping',
                       'tie_break': ['entropy_sum descending', 'canonical image_id ascending'],
                       'conf_provenance': 'prespecified YOLOv8 adaptation, not a literature-optimal threshold'},
        'scope': {'selectors': ['Random', 'DINOv2 frozen representation diversity', 'Basic Detection Entropy'],
                  'reference': '100% labeled-data reference', 'new_training_cells_planned': 200,
                  'reference_training_runs_planned': 4},
        'split': parent['split'], 'budgets': parent['budgets'],
        'seed_freeze': parent['seed_freeze'], 'training': parent['yolo'],
        'parent_protocol_sha256': file_sha256(ROOT / 'confirmatory_v1/confirmatory_protocol_v1.yaml'),
        'parent_v1_1_sha256': file_sha256(ROOT / 'confirmatory_v1_1/confirmatory_protocol_v1_1.yaml'),
        'environment': audit['environment'], 'inference': config,
        'smoke_selection_rule': 'first acquisition index (0), learner slot A, existing validation-best; no outcome selection',
        'official_learner_coupling': 'UNRESOLVED; smoke slot A is not an official coupling decision',
        'datasets': {},
    }
    for dataset, short in [('GC10-DET', 'gc10'), ('NEU-DET', 'neu')]:
        checkpoint = audit['checkpoints'][dataset]
        train = sorted(r['sample_id'] for r in csv.DictReader((freeze / f'{short}_acquisition_universe_blind.csv').open()))
        r0 = ROOT / f'confirmatory_v1_1/manifest_seals/{dataset}/a00/shared_R0.txt'
        labeled = r0.read_text().splitlines()
        cache = ROOT / f'confirmatory_v1_1/full_run/yolo_data_cache/{dataset}'
        rows = [{'image_id': i, 'image_path': str(cache / 'images' / f'{i}.jpg')}
                for i in train if i not in set(labeled)]
        dest = report / dataset
        dest.mkdir(parents=True, exist_ok=True)
        transport = dest / 'unlabeled_images.json'
        transport.write_text(json.dumps(rows, indent=2) + '\n')
        dataset_spec = {
            'train_manifest': str(freeze / f'{short}_acquisition_universe_blind.csv'),
            'train_manifest_sha256': file_sha256(freeze / f'{short}_acquisition_universe_blind.csv'),
            'r0_manifest': str(r0), 'r0_manifest_sha256': file_sha256(r0),
            'unlabeled_manifest': str(transport), 'unlabeled_manifest_sha256': file_sha256(transport),
            'split_assignment': str(freeze / 'split_assignment.csv'),
            'split_assignment_sha256': file_sha256(freeze / 'split_assignment.csv'),
            'checkpoint': checkpoint['path'], 'checkpoint_sha256': checkpoint['sha256'],
            'checkpoint_registry': str(ROOT / 'confirmatory_v1_1/full_run/planned_cells.jsonl'),
            'checkpoint_registry_sha256': file_sha256(ROOT / 'confirmatory_v1_1/full_run/planned_cells.jsonl'),
            'checkpoint_cell_id': checkpoint['spec']['cell_id'],
            'acquisition_index': 0, 'acquisition_seed': checkpoint['spec']['acquisition_seed'],
            'learner_slot': 'A', 'learner_seed': checkpoint['spec']['training_seed'],
            'checkpoint_round': 0, 'query_round': 1,
            'train_count': len(train), 'labeled_count': len(labeled), 'unlabeled_count': len(rows),
            'query_size': parent['budgets'][dataset]['query_count_per_round'],
            'split_counts': parent['datasets'][dataset]['split_counts'],
        }
        manifest['datasets'][dataset] = dataset_spec
    path = docs / 'entropy_baseline_protocol_manifest_v1.yaml'
    path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding='utf-8', newline='\n')
    print(path)


if __name__ == '__main__':
    main()
