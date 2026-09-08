"""Instrument one canonical first image per dataset to record resolved predictor args.

Calls the production scorer's predictor unchanged; no train or evaluation method.
"""
from pathlib import Path
import json
import sys
from unittest.mock import patch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from defect_al.entropy import _predict, image_entropy, file_sha256


def main():
    manifest = yaml.safe_load((ROOT / 'docs/design/entropy_baseline_protocol_manifest_v1.yaml').read_text())
    from ultralytics import YOLO
    original = YOLO.predict
    for dataset, spec in manifest['datasets'].items():
        captured = {}
        def traced(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            captured.update({
                'passed_kwargs': kwargs,
                'resolved_predictor_args': vars(self.predictor.args),
                'model_fp16': self.predictor.model.fp16,
                'model_end2end': getattr(self.predictor.model, 'end2end', False),
                'predictor_class': type(self.predictor).__module__ + '.' + type(self.predictor).__name__,
            })
            assert self.predictor.model.fp16 is False
            assert self.predictor.args.quantize is None
            assert self.predictor.args.rect is False
            return result
        row = sorted(json.loads(Path(spec['unlabeled_manifest']).read_text()), key=lambda r: r['image_id'])[0]
        with patch.object(YOLO, 'predict', traced):
            results = list(_predict(spec['checkpoint'], [row], manifest['inference']))
        baseline = json.loads((ROOT / f'reports/entropy_preflight_20260908/{dataset}/repeat_1/scores.json').read_text())
        expected = next(r for r in baseline['score_table'] if r['image_id'] == row['image_id'])
        assert results[0][0] == row['image_id']
        assert image_entropy(results[0][1]) == expected['entropy_sum']
        assert len(results[0][1]) == expected['n_detections']
        captured.update({'image_id': row['image_id'], 'score_matches_full_smoke': True,
                         'inference_config': manifest['inference'],
                         'checkpoint_sha256': file_sha256(spec['checkpoint']),
                         'scorer_source_sha256': file_sha256(ROOT / 'src/defect_al/entropy.py')})
        target = ROOT / f'reports/entropy_preflight_20260908/{dataset}/effective_predict_args.json'
        target.write_text(json.dumps(captured, indent=2, default=str) + '\n')
        print(f'{dataset}: effective kwargs captured; FP32 verified; first-image score matches full smoke')


if __name__ == '__main__':
    main()
