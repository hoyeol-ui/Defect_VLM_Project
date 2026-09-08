"""Actual installed NMS/Results contract, using synthetic predictions, no images or GT."""
import inspect

import pytest


def test_installed_standard_detection_confidence_contract():
    torch = pytest.importorskip('torch')
    pytest.importorskip('torchvision')
    from ultralytics.engine.results import Boxes
    from ultralytics.utils.nms import non_max_suppression
    # BCN: 4 decoded box coordinates + 2 independent sigmoid class scores.
    # Seven anchors avoid the NMS end-to-end shortcut for shape[-1] == 6.
    prediction = torch.zeros((1, 6, 7), dtype=torch.float32)
    prediction[0, :4, :] = torch.tensor([[20], [20], [10], [10]])
    prediction[0, 4, :3] = torch.tensor([0.8, 0.1, 0.25])
    prediction[0, 5, :3] = torch.tensor([0.7, 0.9, 0.1])
    kept = non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.7,
                               agnostic=False, multi_label=False, max_det=300)[0]
    boxes = Boxes(kept, orig_shape=(40, 40))
    assert kept.shape == (2, 6)
    assert boxes.conf.tolist() == pytest.approx([0.9, 0.8])
    assert boxes.cls.tolist() == [1, 0]
    # Confidence is the selected max class sigmoid, not class product/objectness.
    assert boxes.conf[1].item() != pytest.approx(0.8 * 0.7)


def test_internal_nms_defaults_match_pinned_source_contract():
    from ultralytics.utils.nms import non_max_suppression
    defaults = {k: p.default for k, p in inspect.signature(non_max_suppression).parameters.items()}
    assert defaults['multi_label'] is False
    assert defaults['labels'] == ()
    assert defaults['max_time_img'] == 0.05
    assert defaults['max_nms'] == 30000
    assert defaults['max_wh'] == 7680


def test_half_false_is_explicit_fp32_in_installed_predictor_config():
    from ultralytics.cfg import get_cfg
    assert get_cfg(overrides={'half': False}).quantize is None
