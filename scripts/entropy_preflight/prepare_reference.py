"""Write manifests/configs and dormant command text for four full-data runs."""
import csv
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from defect_al.entropy import file_sha256


def main():
    manifest = yaml.safe_load((ROOT / 'docs/design/entropy_baseline_protocol_manifest_v1.yaml').read_text())
    audit = json.loads((ROOT / 'reports/entropy_preflight_20260908/repository_audit.json').read_text())
    base = ROOT / 'docs/design/full_data_reference_20260908'
    base.mkdir(parents=True, exist_ok=True)
    runs = []
    for dataset in ['GC10-DET', 'NEU-DET']:
        spec = manifest['datasets'][dataset]
        cache = ROOT / f'confirmatory_v1_1/full_run/yolo_data_cache/{dataset}'
        directory = base / dataset
        directory.mkdir(parents=True, exist_ok=True)
        ids = sorted(r['sample_id'] for r in csv.DictReader(Path(spec['train_manifest']).open()))
        (directory / 'train_ids.txt').write_text('\n'.join(ids) + '\n', encoding='ascii', newline='\n')
        (directory / 'train_images.txt').write_text('\n'.join((cache / 'images' / f'{i}.jpg').as_posix() for i in ids) + '\n', encoding='ascii', newline='\n')
        data = yaml.safe_load((cache / 'states/a00_SHARED_R0.yaml').read_text())
        data['train'] = (directory / 'train_images.txt').as_posix()
        data['val'] = (cache / 'lists/development.txt').as_posix()
        (directory / 'dataset.yaml').write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8', newline='\n')
        seed_row = next(r for r in audit['seed_mapping'] if r['dataset'] == dataset and r['acquisition_index'] == '0')
        short = 'gc10' if dataset == 'GC10-DET' else 'neu'
        validation_manifest = ROOT / f'confirmatory_v1/data_freeze/{short}_development_locked_manifest.csv'
        test_manifest = ROOT / f'confirmatory_v1/data_freeze/{short}_final_locked_manifest.csv'
        for slot in ['A', 'B']:
            # Explicit prespecified subset of existing learner seeds, with no outcome screening.
            seed = int(seed_row[f'training_seed_{slot.lower()}'])
            args = dict(audit['checkpoints'][dataset]['training_args'])
            args.pop('save_dir', None)
            args['data'] = (directory / 'dataset.yaml').as_posix()
            args['seed'] = seed
            args['name'] = f'{dataset}_full_reference_{slot}'
            args['project'] = (ROOT / 'runs/full_data_reference_20260908').as_posix()
            args['exist_ok'] = False
            config_path = directory / f'train_{slot}.yaml'
            config_path.write_text(yaml.safe_dump(args, sort_keys=False), encoding='utf-8', newline='\n')
            expected = Path(args['project']) / args['name']
            runs.append({'dataset': dataset, 'learner_seed': seed, 'repeat_slot': slot,
                         'seed_origin': 'existing acquisition_index=0 learner slot; no acquisition axis in full reference',
                         'train_config': config_path.as_posix(), 'train_config_sha256': file_sha256(config_path),
                         'data_yaml': (directory / 'dataset.yaml').as_posix(),
                         'data_yaml_sha256': file_sha256(directory / 'dataset.yaml'),
                         'train_ids_sha256': file_sha256(directory / 'train_ids.txt'),
                         'validation_manifest': validation_manifest.as_posix(),
                         'validation_manifest_sha256': file_sha256(validation_manifest),
                         'validation_image_list': data['val'],
                         'validation_image_list_sha256': file_sha256(data['val']),
                         'locked_test_manifest': test_manifest.as_posix(),
                         'locked_test_manifest_sha256': file_sha256(test_manifest),
                         'train_count': len(ids), 'expected_output': expected.as_posix(),
                         'expected_best': (expected / 'weights/best.pt').as_posix(),
                         'expected_last': (expected / 'weights/last.pt').as_posix(),
                         'training_executed': False})
    (base / 'runplan.json').write_text(json.dumps(runs, indent=2) + '\n')
    python = manifest['environment']['executable'].replace('\\', '/')
    lines = ['# 100% labeled-data reference run plan', '',
             'PREPARED ONLY. No training or evaluation executed. This is a reference, not a selector or a statistical bound.', '',
             'Two datasets × two learner seeds = four runs. Use the first existing A/B learner pair per dataset, selected by index only. '
             'Acquisition seed is not a repeat axis here. This explicit seed subset is a preparation assumption to review before execution.', '',
             '| Dataset | Repeat | Learner seed | Train images |', '|---|---|---:|---:|']
    lines += [f"| {r['dataset']} | {r['repeat_slot']} | {r['learner_seed']} | {r['train_count']} |" for r in runs]
    lines += ['', 'Each dataset.yaml points to the complete sealed acquisition pool and the same fixed development validation list. '
              'It has no test key. No annotation file was opened to prepare these manifests. '
              'Train configs retain every recorded R0 training argument, changing only seed, data, name and project; save_dir is runtime-generated.', '',
              '## Dormant training commands', '',
              'Run only in a separately authorized session after checkpoint/config/data hash checks and a new four-run registry freeze. '
              'Do not pass these configs to the historical 440-cell runner.', '', '```powershell']
    for r in runs:
        code = f"import yaml; from ultralytics import YOLO; c=yaml.safe_load(open('{r['train_config']}')); YOLO(c['model']).train(**c)"
        lines.append(f'& "{python}" -c "{code}"')
    lines += ['```', '', '## Dormant validation commands', '', '```powershell']
    for r in runs:
        code = f"from ultralytics import YOLO; YOLO('{r['expected_best']}').val(data='{r['data_yaml']}', split='val', imgsz=640, batch=8, device=0, workers=4, conf=0.001, iou=0.7, max_det=300, plots=False, save_json=False, verbose=False, project='{r['expected_output']}', name='sealed_development_eval')"
        lines.append(f'& "{python}" -c "{code}"')
    lines += ['```', '', '## Locked-test evaluation preparation', '',
              'The historical evaluation launcher requires exactly 440 historical cells and its own authorization seal. '
              'It cannot be reused for these four new runs. A separate four-cell checkpoint registry and explicit evaluation authorization are required. '
              'The following API calls describe the intended post-authorization evaluation; they do not provide the missing guard/authorization wrapper. '
              'Existing locked-final YAML paths are listed by filename only; no final data or outcome was loaded in this session.', '', '```powershell']
    for r in runs:
        final_yaml = (ROOT / f"confirmatory_v1_1/final_analysis/final_evaluation/data_cache/{r['dataset']}/locked_final.yaml").as_posix()
        code = f"from ultralytics import YOLO; YOLO('{r['expected_best']}').val(data='{final_yaml}', split='val', imgsz=640, batch=8, device=0, workers=4, conf=0.001, iou=0.7, max_det=300, plots=False, save_json=False, verbose=False, project='{r['expected_output']}', name='authorized_locked_test_eval')"
        lines.append(f'& "{python}" -c "{code}"')
    lines += ['```', '', 'The locked-final adapter uses its `val` key for the locked-test image list; verify that mapping and all split hashes inside the new authorization wrapper before use.', '',
              '## Exact paths and hashes', '', 'See `full_data_reference_20260908/runplan.json` for all config, manifest, checkpoint-output paths and SHA-256 values. '
              'Pretrained model, 100 epochs, SGD, batch 8, 640px, augmentation, cold start and validation-selected best.pt remain identical to the archived R0 args. '
              'Expected last.pt is retained for audit only. Four independent cold starts; no warm-start or acquisition.', '']
    (ROOT / 'docs/design/full_data_reference_runplan_20260908.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Prepared {len(runs)} run configurations; executed 0 runs.')


if __name__ == '__main__':
    main()
