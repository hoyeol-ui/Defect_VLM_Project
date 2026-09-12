"""One authorized cold-start cell in an isolated process; no locked-test entrypoint."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.metadata
import os
from pathlib import Path
import sys
import time

from .entropy import file_sha256, stable_sha256
from .entropy_extension import ExtensionGuardError, checked_file, inside, now, write_json_new
from .entropy_official import OfficialExtension as Extension


def verify_environment(protocol):
    expected = protocol['environment']
    if '.'.join(map(str, sys.version_info[:3])) != expected['python_version']:
        raise ExtensionGuardError('Python version mismatch')
    for name, version in expected['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ExtensionGuardError(f'package version mismatch: {name}')
    import torch
    import ultralytics
    if not torch.cuda.is_available() or torch.version.cuda != expected['cuda']:
        raise ExtensionGuardError('CUDA unavailable or runtime mismatch')
    if torch.cuda.get_device_name(0) != expected['gpu']:
        raise ExtensionGuardError('GPU differs from frozen local runtime')
    package = Path(ultralytics.__file__).parent
    for relative, digest in expected['ultralytics_sources'].items():
        checked_file(package / relative, digest)
    return expected


def run_cell(extension_root, mode, node_id, authorization):
    ext = Extension(extension_root, mode)
    node = ext.node(node_id)
    if node['kind'] != 'training':
        raise ExtensionGuardError('training cell expected')
    # Authorization is checked before image/label materialization, imports, or GPU work.
    auth = ext.verify_authorization(authorization, node_id)
    spec = ext.prepare_training(node_id)
    if ext.state(node_id)['state'] == 'COMPLETE':
        return ext.completed(node_id)
    checked_file(spec['initialization_path'], spec['initialization_sha256'])
    config = spec['training_config']
    if (config['resume'] is not False or config['epochs'] != 100 or config['val'] is not True
            or config['seed'] != node['training_seed'] or config['model'] != spec['initialization_path']):
        raise ExtensionGuardError('cold-start training policy mismatch')
    import yaml
    data = yaml.safe_load(Path(spec['data_yaml_path']).read_text())
    if set(data) != {'path', 'train', 'val', 'names'}:
        raise ExtensionGuardError('training YAML must expose only train/validation')
    for key in ('train', 'val'):
        inside(ext.root, data[key])
    attempt = ext._claim(node_id, 'READY')
    output = inside(ext.root, ext.root / 'outcomes_private' / mode / node_id / f'attempt_{attempt:02d}')
    log_path = inside(ext.root, ext.root / 'logs_private' / mode / f'{node_id}_attempt_{attempt:02d}.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    public = sys.stdout
    try:
        if output.exists():
            raise ExtensionGuardError('refusing to overwrite training output')
        with log_path.open('x', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
            environment = verify_environment(ext.protocol)
            import torch
            from ultralytics import YOLO
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            args = dict(config)
            args.update(project=str(output.parent), name=output.name, exist_ok=False)
            # No previous-round weights enter this constructor.
            model = YOLO(spec['initialization_path'], verbose=False)
            def heartbeat(trainer):
                epoch = int(trainer.epoch) + 1
                if epoch == 1 or epoch % 10 == 0:
                    print(f'ENTROPY_TRAIN {node_id} epoch {epoch}/100', file=public, flush=True)
            model.add_callback('on_train_epoch_end', heartbeat)
            started = time.perf_counter()
            model.train(**args)
            train_seconds = time.perf_counter() - started
            if Path(model.trainer.save_dir).resolve() != output.resolve() or int(model.trainer.epoch) + 1 != 100:
                raise ExtensionGuardError('output directory or epoch count mismatch')
            best, last = output / 'weights/best.pt', output / 'weights/last.pt'
            if not best.is_file() or not last.is_file():
                raise ExtensionGuardError('missing best/last lifecycle artifacts')
            # Same post-training validation as historical protocol, kept private.
            started = time.perf_counter()
            YOLO(str(best), verbose=False).val(
                data=spec['data_yaml_path'], split='val', imgsz=640, batch=8, device=0,
                workers=4, conf=0.001, iou=0.7, max_det=300, plots=False,
                save_json=False, verbose=False, project=str(output), name='sealed_validation_eval')
            validation_seconds = time.perf_counter() - started
            checked_file(spec['initialization_path'], spec['initialization_sha256'])
            checked_file(spec['manifest_path'], spec['manifest_sha256'])
            checked_file(spec['data_yaml_path'], spec['data_yaml_sha256'])
            # Check prepared bytes again; no post-training GT composition analysis.
            checked_file(data['train'], spec['train_list_sha256'])
            checked_file(data['val'], spec['validation_list_sha256'])
            actual = yaml.safe_load((output / 'args.yaml').read_text())
            for key, expected in args.items():
                if key == 'device':
                    equal = str(actual.get(key)) == str(expected)
                else:
                    equal = actual.get(key) == expected
                if not equal:
                    raise ExtensionGuardError(f'actual training argument drift: {key}')
            result = {**node, 'status': 'COMPLETE', 'attempt': attempt, 'timestamp': now(),
                      'freeze_sha256': ext.freeze_sha, 'authorization_sha256': file_sha256(authorization),
                      'cold_start': True, 'resume': False, 'epochs_completed': 100,
                      'manifest_path': spec['manifest_path'], 'manifest_sha256': spec['manifest_sha256'],
                      'training_config_sha256': spec['training_config_sha256'],
                      'actual_train_kwargs_sha256': stable_sha256(args),
                      'data_yaml_sha256': spec['data_yaml_sha256'],
                      'initialization_path': spec['initialization_path'],
                      'initialization_sha256': spec['initialization_sha256'],
                      'best_path': str(best), 'best_sha256': file_sha256(best),
                      'last_path': str(last), 'last_sha256': file_sha256(last),
                      'checkpoint_selection_split': 'validation', 'locked_test_accessed': False,
                      'train_runtime_seconds': train_seconds, 'validation_runtime_seconds': validation_seconds,
                      'environment': environment, 'outcome_values_exposed': False,
                      'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(0),
                      'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(0),
                      'files': [{'path': str(p), 'sha256': file_sha256(p)} for p in [best, last, output / 'args.yaml']]}
        result_path = output / 'cell_result.json'
        write_json_new(result_path, result)
        ext._complete(node_id, result_path)
        print(f'ENTROPY_COMPLETE {node_id}; best/last/config/manifest seals saved', file=public, flush=True)
        return result
    except BaseException:
        ext.fail(node_id)
        raise
