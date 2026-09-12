"""Basic Detection Entropy acquisition; no training, labels, or evaluation API.

The caller supplies only the fixed train membership and image transport paths.
Official learner-slot coupling is intentionally outside this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


class EntropyIntegrityError(ValueError):
    """Fail closed on an incomplete or inconsistent acquisition input."""


def binary_entropy(p: float) -> float:
    p = float(p)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise EntropyIntegrityError("confidence must be finite and in [0, 1]")
    # Exact endpoint limits avoid log(0), without clipping interior scores.
    if p == 0 or p == 1:
        return 0.0
    return -p * math.log(p) - (1 - p) * math.log1p(-p)


def image_entropy(box_confidences) -> float:
    return math.fsum(binary_entropy(p) for p in box_confidences)


def rank_entropy_scores(score_table) -> list[dict]:
    rows = [dict(row) for row in score_table]
    ids = [row['image_id'] for row in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise EntropyIntegrityError("duplicate or invalid image IDs")
    for row in rows:
        if not math.isfinite(row['entropy_sum']) or row['entropy_sum'] < 0:
            raise EntropyIntegrityError("invalid entropy score")
    return sorted(rows, key=lambda row: (-row['entropy_sum'], row['image_id']))


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stable_sha256(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('ascii')).hexdigest()


# Every setting affecting image preprocessing, model inference or NMS is pinned.
# multi_label is an internal NMS argument, not a valid YOLO.predict keyword.
PINNED_INFERENCE = {
    'imgsz': 640, 'conf': 0.25, 'iou': 0.70, 'max_det': 300,
    'agnostic_nms': False, 'multi_label': False, 'augment': False,
    'batch': 1, 'device': '0', 'half': False, 'rect': False,
    'classes': None, 'dnn': False, 'compile': False, 'embed': None,
    'end2end': False, 'task': 'detect', 'mode': 'predict',
    'save': False, 'save_txt': False, 'save_conf': False, 'save_crop': False,
    'save_frames': False, 'save_json': False, 'show': False, 'verbose': False,
    'visualize': False, 'plots': False, 'vid_stride': 1, 'stream_buffer': False,
    'seed': 0, 'deterministic': True,
    'tf32': False, 'cudnn_benchmark': False, 'cublas_workspace_config': ':4096:8',
    'nms_max_time_img': 0.05, 'nms_max_nms': 30000, 'nms_max_wh': 7680,
}


def validate_inference_config(config):
    required = set(PINNED_INFERENCE) | {'software_version', 'source_sha256'}
    if set(config) != required:
        raise EntropyIntegrityError(f"inference config fields differ: {set(config) ^ required}")
    for key, expected in PINNED_INFERENCE.items():
        if type(config[key]) is not type(expected) or config[key] != expected:
            raise EntropyIntegrityError(f"unfrozen inference config: {key}")
    if not config['software_version'] or not config['source_sha256']:
        raise EntropyIntegrityError('missing software provenance')


def _ids(values, name):
    values = list(values)
    if len(values) != len(set(values)) or any(
        not isinstance(i, str) or re.fullmatch('[0-9a-f]{64}', i) is None for i in values
    ):
        raise EntropyIntegrityError(f'duplicate/noncanonical {name} IDs')
    return set(values)


def validate_pool(records, labeled_ids, train_ids):
    for row in records:
        if set(row) != {'image_id', 'image_path'}:
            raise EntropyIntegrityError('image records must contain only ID and transport path')
    unlabeled = _ids([r['image_id'] for r in records], 'unlabeled')
    labeled = _ids(labeled_ids, 'labeled')
    train = _ids(train_ids, 'train')
    if labeled & unlabeled or labeled | unlabeled != train:
        raise EntropyIntegrityError('labeled/unlabeled overlap or incomplete fixed train pool')
    return unlabeled, labeled, train


def _predict(checkpoint, records, config):
    """Use the audited high-level predictor; reject source/version drift."""
    import os
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = config['cublas_workspace_config']
    import importlib.metadata as metadata
    import logging
    import torch
    import torchvision  # Pin the torchvision NMS branch before inference.
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER

    for package, version in config['software_version'].items():
        if metadata.version(package) != version:
            raise EntropyIntegrityError(f'package version drift: {package}')
    root = Path(ultralytics.__file__).parent
    for relative, expected in config['source_sha256'].items():
        if file_sha256(root / relative) != expected:
            raise EntropyIntegrityError(f'Ultralytics source drift: {relative}')
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = config['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32 = config['tf32']
    torch.backends.cudnn.allow_tf32 = config['tf32']

    class RejectTruncatedNMS(logging.Handler):
        def emit(self, record):
            if 'NMS time limit' in record.getMessage():
                raise EntropyIntegrityError('NMS timed out; incomplete detections cannot be scored')

    guard = RejectTruncatedNMS()
    LOGGER.addHandler(guard)
    try:
        model = YOLO(str(checkpoint), task='detect', verbose=False)
        if model.task != 'detect' or getattr(model.model, 'end2end', False):
            raise EntropyIntegrityError('standard YOLOv8 detection checkpoint required')
        internal = {'multi_label', 'tf32', 'cudnn_benchmark', 'cublas_workspace_config',
                    'nms_max_time_img', 'nms_max_nms', 'nms_max_wh'}
        kwargs = {k: config[k] for k in PINNED_INFERENCE if k not in internal}
        # One path per call ensures batch=1 really is one image, including warmup.
        for row in records:
            result = model.predict(source=row['image_path'], stream=False, **kwargs)
            if len(result) != 1 or Path(result[0].path).resolve() != Path(row['image_path']).resolve():
                raise EntropyIntegrityError('prediction identity/count mismatch')
            boxes = result[0].boxes
            if boxes is None or boxes.data.ndim != 2 or boxes.data.shape[1] != 6:
                raise EntropyIntegrityError('expected standard Nx6 Results.boxes')
            yield row['image_id'], boxes.conf.detach().cpu().double().tolist()
    finally:
        LOGGER.removeHandler(guard)


def entropy_acquisition(checkpoint, unlabeled_manifest, query_size, inference_config,
                        *, labeled_ids, train_ids, checkpoint_sha256,
                        _predict_fn=None):
    """Score all U_r and return deterministic rows, ranking, query and next pool.

    Manifest records have only image_id and image_path. GT-bearing fields fail.
    The injected predictor is for synthetic unit tests only; production uses _predict.
    """
    validate_inference_config(inference_config)
    records = sorted((dict(r) for r in unlabeled_manifest), key=lambda r: r.get('image_id', ''))
    unlabeled, labeled, train = validate_pool(records, labeled_ids, train_ids)
    if type(query_size) is not int or not 0 < query_size <= len(unlabeled):
        raise EntropyIntegrityError('invalid query size')
    checkpoint = Path(checkpoint)
    if checkpoint.name != 'best.pt' or file_sha256(checkpoint) != checkpoint_sha256:
        raise EntropyIntegrityError('validation-best checkpoint name/hash mismatch')
    paths = [Path(r['image_path']).resolve() for r in records]
    if len(set(paths)) != len(paths) or any(not p.is_file() or p.stem != r['image_id']
                                          for p, r in zip(paths, records)):
        raise EntropyIntegrityError('missing/duplicate/noncanonical image transport path')
    rows, seen = [], set()
    for image_id, confidences in (_predict_fn or _predict)(checkpoint, records, inference_config):
        if image_id not in unlabeled or image_id in seen:
            raise EntropyIntegrityError('unexpected/duplicate prediction ID')
        seen.add(image_id)
        confidences = list(confidences)
        if len(confidences) > inference_config['max_det']:
            raise EntropyIntegrityError('max_det exceeded')
        if any(not math.isfinite(float(p)) or not inference_config['conf'] < float(p) <= 1
               for p in confidences):
            raise EntropyIntegrityError('confidence outside retained-detection bounds')
        rows.append({'image_id': image_id, 'n_detections': len(confidences),
                     'entropy_sum': image_entropy(confidences)})
    if seen != unlabeled:
        raise EntropyIntegrityError('incomplete inference coverage')
    if file_sha256(checkpoint) != checkpoint_sha256:
        raise EntropyIntegrityError('checkpoint changed during inference')
    ranked = rank_entropy_scores(rows)
    for rank, row in enumerate(ranked, 1):
        row.update(rank=rank, selected=rank <= query_size)
    selected = [r['image_id'] for r in ranked[:query_size]]
    next_labeled = labeled | set(selected)
    next_unlabeled = unlabeled - set(selected)
    if (len(selected) != query_size or len(set(selected)) != query_size or
            labeled & set(selected) or next_labeled & next_unlabeled or
            next_labeled | next_unlabeled != train):
        raise EntropyIntegrityError('post-acquisition pool integrity failure')
    score_table = sorted(ranked, key=lambda r: r['image_id'])
    ranking = [r['image_id'] for r in ranked]
    return {'score_table': score_table, 'ranking': ranking, 'selected_ids': selected,
            'next_labeled_ids': sorted(next_labeled), 'next_unlabeled_ids': sorted(next_unlabeled),
            'hashes': {'score_table': stable_sha256(score_table),
                       'ranking': stable_sha256(ranking), 'selected_ids': stable_sha256(selected)}}
