"""R0 -> R1 inference only, one frozen acquisition/learner per dataset.

Two invocations with different --repeat values perform independent full inference.
No detector train/val call, GT annotation reader, AP computation, or threshold sweep.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import statistics
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from defect_al.entropy import entropy_acquisition, file_sha256, EntropyIntegrityError


def canonical_pixel_id(path):
    from PIL import Image
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        h = hashlib.sha256(b'CONFIRMATORY_RGB_U8_V1\0')
        h.update(rgb.width.to_bytes(4, 'big'))
        h.update(rgb.height.to_bytes(4, 'big'))
        h.update(rgb.tobytes())
        return h.hexdigest()


def distribution(values):
    return dict(min=min(values), median=statistics.median(values),
                mean=statistics.mean(values), max=max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['GC10-DET', 'NEU-DET'])
    parser.add_argument('--repeat', required=True, choices=['1', '2'])
    args = parser.parse_args()
    path = ROOT / 'docs/design/entropy_baseline_protocol_manifest_v1.yaml'
    manifest = yaml.safe_load(path.read_text(encoding='utf-8'))
    if manifest['execution_authority'] != {'inference_smoke': True, 'training': False, 'locked_test_evaluation': False}:
        raise EntropyIntegrityError('this runner is smoke-only')
    spec = manifest['datasets'][args.dataset]
    if spec['checkpoint_round'] != 0 or spec['query_round'] != 1 or spec['acquisition_index'] != 0 or spec['learner_slot'] != 'A':
        raise EntropyIntegrityError('outside the single R0 smoke scope')
    for key in ['train_manifest', 'r0_manifest', 'unlabeled_manifest', 'split_assignment', 'checkpoint_registry']:
        if file_sha256(spec[key]) != spec[key + '_sha256']:
            raise EntropyIntegrityError(f'seal mismatch: {key}')
    registry = [json.loads(line) for line in Path(spec['checkpoint_registry']).read_text().splitlines()]
    cell = next(r for r in registry if r['cell_id'] == spec['checkpoint_cell_id'])
    if (cell['reusable_best_sha256'] != spec['checkpoint_sha256'] or
            Path(cell['reusable_artifact_path']) / 'weights/best.pt' != Path(spec['checkpoint']) or
            cell['manifest_sha256'] != spec['r0_manifest_sha256'] or
            cell['training_seed'] != spec['learner_seed']):
        raise EntropyIntegrityError('R0 checkpoint/manifest/learner provenance mismatch')
    rows = json.loads(Path(spec['unlabeled_manifest']).read_text())
    train = [r['sample_id'] for r in csv.DictReader(Path(spec['train_manifest']).open())]
    labeled = Path(spec['r0_manifest']).read_text().splitlines()
    if (len(rows), len(train), len(labeled)) != (spec['unlabeled_count'], spec['train_count'], spec['labeled_count']):
        raise EntropyIntegrityError('pool count mismatch')
    # Only split IDs are read by this integrity wrapper; no GT, final images or metrics.
    membership = [r for r in csv.DictReader(Path(spec['split_assignment']).open()) if r['dataset'] == args.dataset]
    groups = {split: {r['sample_id'] for r in membership if r['split'] == split}
              for split in ['acquisition', 'development', 'final']}
    if (set(train) != groups['acquisition'] or groups['acquisition'] & groups['development'] or
            groups['acquisition'] & groups['final'] or groups['development'] & groups['final']):
        raise EntropyIntegrityError('split overlap/mismatch')
    destination = ROOT / f'reports/entropy_preflight_20260908/{args.dataset}/repeat_{args.repeat}'
    destination.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    clock = time.perf_counter()
    raw_hashes = {}
    for row in rows:
        if canonical_pixel_id(row['image_path']) != row['image_id']:
            raise EntropyIntegrityError('transport image pixels do not match canonical ID')
        raw_hashes[row['image_id']] = file_sha256(row['image_path'])
    print(f'{args.dataset} repeat {args.repeat}: verified {len(rows)} unlabeled image identities; inference starting', flush=True)
    output = entropy_acquisition(spec['checkpoint'], rows, spec['query_size'], manifest['inference'],
                                 labeled_ids=labeled, train_ids=train,
                                 checkpoint_sha256=spec['checkpoint_sha256'])
    selected = set(output['selected_ids'])
    integrity = {
        'query_size_correct': len(selected) == spec['query_size'],
        'labeled_selected_count': len(selected & set(labeled)),
        'validation_selected_count': len(selected & groups['development']),
        'test_selected_count': len(selected & groups['final']),
        'split_hash_unchanged': file_sha256(spec['split_assignment']) == spec['split_assignment_sha256'],
        'image_hashes_unchanged': all(file_sha256(r['image_path']) == raw_hashes[r['image_id']] for r in rows),
        'pixel_identity_count_verified': len(rows),
        'next_pools_disjoint': not (set(output['next_labeled_ids']) & set(output['next_unlabeled_ids'])),
        'next_pools_union_is_train': set(output['next_labeled_ids']) | set(output['next_unlabeled_ids']) == set(train),
        'duplicate_ids': len(rows) - len({r['image_id'] for r in rows}),
    }
    if (not all(integrity[k] for k in ['query_size_correct', 'split_hash_unchanged', 'image_hashes_unchanged',
                                      'next_pools_disjoint', 'next_pools_union_is_train']) or
            any(integrity[k] for k in ['labeled_selected_count', 'validation_selected_count', 'test_selected_count', 'duplicate_ids'])):
        raise EntropyIntegrityError('post-smoke integrity failure')
    # Canonical timestamp-free content is independently hashable across processes.
    (destination / 'scores.json').write_text(json.dumps(output, sort_keys=True, indent=2, allow_nan=False) + '\n')
    n = [r['n_detections'] for r in output['score_table']]
    s = [r['entropy_sum'] for r in output['score_table']]
    from collections import Counter
    from scipy.stats import spearmanr
    counts = Counter(s)
    rho = float(spearmanr(s, n).statistic) if len(set(s)) > 1 and len(set(n)) > 1 else None
    stats = {
        'unlabeled_pool_size': len(rows), 'zero_detection_images': n.count(0),
        'zero_detection_rate': n.count(0) / len(n), 'n_detections': distribution(n),
        'entropy_sum': distribution(s), 'unique_entropy_score_count': len(counts),
        'exact_tie_excess_count': len(s) - len(counts),
        'images_in_tie_groups': sum(v for v in counts.values() if v > 1),
        'exact_tie_pair_count': sum(v * (v - 1) // 2 for v in counts.values()),
        'selected_query_size': len(selected), 'nan_inf_count': sum(not math.isfinite(v) for v in s),
        'spearman_entropy_vs_n_detections': rho,
        'top_ids': output['ranking'][:3], 'middle_ids': output['ranking'][len(rows)//2:len(rows)//2+3],
        'bottom_ids': output['ranking'][-3:],
    }
    annotated = []
    for row in output['score_table']:
        annotated.append({**row, 'dataset': args.dataset, 'acquisition_seed': spec['acquisition_seed'],
                          'round': 0, 'query_round': 1, 'learner_seed': spec['learner_seed'],
                          'checkpoint_path': spec['checkpoint'], 'checkpoint_sha256': spec['checkpoint_sha256'],
                          **{k: manifest['inference'][k] for k in ['conf', 'iou', 'max_det', 'imgsz']},
                          'timestamp': started,
                          'software_version': json.dumps(manifest['inference']['software_version'], sort_keys=True)})
    with (destination / 'scores_with_provenance.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(annotated[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(annotated)
    for name, ids in [('query_R1', output['selected_ids']), ('labeled_R1', output['next_labeled_ids']),
                      ('unlabeled_R1', output['next_unlabeled_ids'])]:
        (destination / f'{name}.txt').write_text('\n'.join(sorted(ids)) + '\n', encoding='ascii', newline='\n')
    metadata = {'dataset': args.dataset, 'repeat': int(args.repeat), 'status': 'PASS',
                'started_at': started, 'ended_at': datetime.now(timezone.utc).isoformat(),
                'duration_seconds': time.perf_counter() - clock,
                'protocol_manifest_sha256': file_sha256(path),
                'scorer_source_sha256': file_sha256(ROOT / 'src/defect_al/entropy.py'),
                'runner_source_sha256': file_sha256(Path(__file__)),
                'stats': stats, 'integrity': integrity, 'hashes': output['hashes'],
                'scores_file_sha256': file_sha256(destination / 'scores.json'),
                'training_executed': False, 'gt_annotations_read': False, 'test_images_read': False,
                'performance_metrics_computed': False}
    (destination / 'summary.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
