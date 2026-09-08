"""Create the separate immutable Entropy plan/freeze without starting training."""
import csv
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from defect_al.entropy import file_sha256
from defect_al.entropy_extension import (Extension, ExtensionGuardError, build_plan,
                                        checked_file, read_json, write_json_new, now)


def historical_records():
    sys.path.insert(0, str(REPO / 'confirmatory_v1_1/final_analysis/src'))
    from final_common import build_verified_cell_registry
    public, records = build_verified_cell_registry()  # read-only SQLite + checkpoint/manifest hashes
    if public != read_json(REPO / 'confirmatory_v1_1/final_lock/CELL_REGISTRY.json'):
        raise ExtensionGuardError('historical 440 registry differs from archived seal')
    last_checked = 0
    for row in records:
        meta = row['training_metadata']
        last = Path(row['best_pt_path']).with_name('last.pt')
        expected = row['reusable_last_sha256'] if row.get('reusable_artifact_path') else meta['last_pt_sha256']
        checked_file(last, expected)
        last_checked += 1
    return public, records, last_checked


def main():
    import yaml
    ext = REPO / 'entropy_extension_v1'
    if ext.exists():
        raise ExtensionGuardError('extension already exists; never overwrite a frozen plan')
    before = read_json(REPO / 'reports/entropy_b1_b2_20260908/historical_before.json')
    for relative, digest in before.items():
        checked_file(REPO / relative, digest)
    public, historical, last_count = historical_records()
    previous = read_json(REPO / 'reports/entropy_preflight_20260908/repository_audit.json')
    preflight = yaml.safe_load((REPO / 'docs/design/entropy_baseline_protocol_manifest_v1.yaml').read_text())
    parent = yaml.safe_load((REPO / 'confirmatory_v1/confirmatory_protocol_v1.yaml').read_text(encoding='utf-8'))
    cells, queries = build_plan(previous['seed_mapping'], parent['budgets'])
    refs = {}
    for row in historical:
        if row['round'] != 0:
            continue
        tid = f"{row['dataset']}_a{row['acquisition_index']:02d}"
        refs[row['cell_id']] = {'trajectory': tid, 'slot': row['training_seed_slot'], 'round': 0,
                               'dataset': row['dataset'], 'acquisition_index': row['acquisition_index'],
                               'acquisition_seed': row['acquisition_seed'], 'training_seed': row['training_seed'],
                               'manifest_path': row['manifest_path'], 'manifest_sha256': row['manifest_sha256'],
                               'best_path': row['best_pt_path'], 'best_sha256': row['best_pt_sha256'],
                               'training_args_path':str(Path(row['best_pt_path']).parents[1] / 'args.yaml'),
                               'training_args_sha256':file_sha256(Path(row['best_pt_path']).parents[1] / 'args.yaml'),
                               'read_only': True, 'new_training_cell': False}
    if len(refs) != 40:
        raise ExtensionGuardError('exactly 40 historical R0 checkpoint references required')
    for cell in cells:
        r0 = refs[f"{cell['trajectory']}_SHARED_{cell['slot']}_R0"]
        if r0['training_seed'] != cell['training_seed']:
            raise ExtensionGuardError('new learner seed differs from its shared R0')
    for tid in {c['trajectory'] for c in cells}:
        if refs[f'{tid}_SHARED_A_R0']['manifest_sha256'] != refs[f'{tid}_SHARED_B_R0']['manifest_sha256']:
            raise ExtensionGuardError('historical A/B R0 manifest mismatch')
    env = previous['environment']
    package = Path(env['ultralytics_path'])
    source_paths = list(env['source_sha256']) + ['cfg/__init__.py','engine/trainer.py','data/build.py',
                                               'data/loaders.py','utils/metrics.py','nn/autobackend.py']
    source_hashes = {p:file_sha256(package / p) for p in source_paths}
    config = dict(previous['checkpoints']['GC10-DET']['training_args'])
    for key in ['data','seed','project','name','save_dir']:
        config.pop(key, None)
    # All remaining actual arguments are common to both datasets/R0 runs.
    other = dict(previous['checkpoints']['NEU-DET']['training_args'])
    for key in ['data','seed','project','name','save_dir']:
        other.pop(key, None)
    if config != other:
        raise ExtensionGuardError('historical dataset training config mismatch')
    protocol = {
        'id':'entropy-extension-v1', 'stage':'INTEGRATION_CANDIDATE_NOT_OFFICIALLY_AUTHORIZED',
        'created_at':now(), 'decision_source':'user B1 instruction 20260908 before Entropy training/AP outcomes',
        'policy_learner':'A', 'evaluation_learners':['A','B'], 'trajectory_sharing':'same selected manifest for A/B',
        'learner_B_in_acquisition':False, 'trajectory_count':20, 'new_training_cells':200,
        'historical_R0_references':40, 'rounds':[1,2,3,4,5], 'cold_start_every_cell':True,
        'locked_test_access':'FORBIDDEN', 'training_requires_authorization':True,
        'official_requires_real_integration_pass':True,
        'integration_scope':{'dataset':'GC10-DET','acquisition_index':0,'round':1,'learners':['A','B']},
        'integration_results_auto_promoted':False,
        'retry_policy':{'max_infrastructure_attempts':3,'automatic_retry':False,'resume':False},
        'statistical_unit':'acquisition index, separately within each dataset (n=10)',
        'learner_reduction':'arithmetic mean of A/B six-point AULCs; policy-role asymmetry disclosed',
        'B1_document':str(REPO / 'docs/design/entropy_b1_resolution_20260908.md'),
        'seeds':previous['seed_mapping'], 'budgets':parent['budgets'],
        'inference':preflight['inference'],
        'pretrained':{'path':config['model'],'sha256':parent['yolo']['pretrained_checkpoint']['sha256']},
        'environment':{'python_version':'3.11.9','python_executable':env['executable'],
                       'packages':env['packages'],'cuda':env['cuda'],'gpu':env['gpu'],
                       'ultralytics_sources':source_hashes},
        'historical_protocol_sha256':file_sha256(REPO / 'confirmatory_v1_1/confirmatory_protocol_v1_1.yaml'),
        'parent_preflight_sha256':file_sha256(REPO / 'docs/design/entropy_baseline_protocol_manifest_v1.yaml'),
        'new_final_evaluation_endpoint':None,
    }
    ext.mkdir()
    write_json_new(ext / 'protocol.json', protocol)
    write_json_new(ext / 'training_config.json', config)
    write_json_new(ext / 'planned_cells.json', cells)
    write_json_new(ext / 'planned_acquisitions.json', queries)
    write_json_new(ext / 'historical_r0.json', refs)
    write_json_new(ext / 'historical_guard.json', before)
    write_json_new(ext / 'final_lock.json', {'status':'LOCKED','locked_test_access':'FORBIDDEN',
                   'evaluation_endpoint':None,'historical_authorization_accepted':False,
                   'reason':'This extension implements training/validation only; no locked-test execution.'})
    # Build an input transport/hash index without GT class/count statistics.
    splits = list(csv.DictReader((REPO / 'confirmatory_v1/data_freeze/split_assignment.csv').open()))
    sys.path.insert(0, str(REPO / 'scripts/entropy_preflight'))
    from run_smoke import canonical_pixel_id
    for dataset in ('GC10-DET','NEU-DET'):
        train = sorted(r['sample_id'] for r in splits if r['dataset']==dataset and r['split']=='acquisition')
        validation = sorted(r['sample_id'] for r in splits if r['dataset']==dataset and r['split']=='development')
        if set(train) & set(validation) or len(train) != parent['datasets'][dataset]['split_counts']['acquisition']:
            raise ExtensionGuardError('train/validation split mismatch')
        cache = REPO / 'confirmatory_v1_1/full_run/yolo_data_cache' / dataset
        images, labels = {}, {}
        for image_id in train + validation:
            image, label = cache / 'images' / f'{image_id}.jpg', cache / 'labels' / f'{image_id}.txt'
            if canonical_pixel_id(image) != image_id:
                raise ExtensionGuardError('image pixels differ from canonical ID')
            images[image_id] = {'path':str(image),'sha256':file_sha256(image)}
            labels[image_id] = {'path':str(label),'sha256':file_sha256(label)}
        names = yaml.safe_load((cache / 'states/a00_SHARED_R0.yaml').read_text())['names']
        write_json_new(ext / 'datasets' / f'{dataset}.json',
                       {'dataset':dataset,'train_ids':train,'validation_ids':validation,
                        'images':images,'labels':labels,'names':names,'locked_test_images_indexed':0})
        print(f'{dataset}: sealed {len(train)} train and {len(validation)} validation images; test images 0',flush=True)
    frozen = [p for p in ext.rglob('*') if p.is_file()]
    frozen += [REPO / p for p in ['src/defect_al/entropy.py','src/defect_al/entropy_extension.py',
                                  'src/defect_al/entropy_training.py','scripts/entropy_extension/run.py',
                                  'scripts/entropy_extension/prepare.py','docs/design/entropy_b1_resolution_20260908.md']]
    write_json_new(ext / 'freeze.json', {'stage':'INTEGRATION_CANDIDATE','created_at':now(),
                   'files':[{'path':str(p),'sha256':file_sha256(p)} for p in frozen],
                   'official_authorization_issued':False,'integration_training_authorization_issued':False})
    for mode in ['official','integration']:
        Extension(ext,mode).initialize_registry()
    times = []
    for row in historical:
        if row['dataset']=='GC10-DET' and row['round']==1:
            meta=row['training_metadata']
            times.append({'cell_id':row['cell_id'],'train_seconds':meta['train_runtime_seconds'],
                          'validation_seconds':meta['development_eval_runtime_seconds']})
    total=[r['train_seconds']+r['validation_seconds'] for r in times]
    estimate={'source':'40 historical GC10 R1 cells, 241 images, 100 epochs, same RTX3070 protocol',
              'performance_values_read':False,'measurements':times,
              'two_cells_median_minutes':2*statistics.median(total)/60,
              'two_cells_historical_min_minutes':2*min(total)/60,
              'two_cells_historical_max_minutes':2*max(total)/60,
              'planning_range_minutes':[16,22]}
    write_json_new(REPO / 'reports/entropy_b1_b2_20260908/gpu_time_estimate.json',estimate)
    write_json_new(REPO / 'reports/entropy_b1_b2_20260908/historical_checkpoint_integrity.json',
                   {'status':'PASS','best_checkpoints_verified':len(public),'last_checkpoints_verified':last_count,
                    'matches_archived_registry':True,'R0_references_verified':len(refs),'locked_test_accessed':False})
    print(json.dumps({'planned_training_cells':len(cells),'acquisition_nodes':len(queries),
                      'trajectories':20,'freeze_sha256':file_sha256(ext/'freeze.json'),'training_executed':False}))


if __name__ == '__main__':
    main()
