import copy
import math

import pytest

from defect_al.entropy import (EntropyIntegrityError, PINNED_INFERENCE, binary_entropy,
                               entropy_acquisition, file_sha256, image_entropy,
                               rank_entropy_scores)


def test_binary_half():
    assert binary_entropy(0.5) == pytest.approx(math.log(2), abs=1e-15)


def test_symmetry():
    assert binary_entropy(0.2) == pytest.approx(binary_entropy(0.8), abs=1e-15)


def test_empty_is_exact_zero():
    assert image_entropy([]) == 0.0


def test_sum_not_mean_or_max():
    assert image_entropy([0.5, 0.5]) == pytest.approx(2 * math.log(2), abs=1e-15)


def test_exact_tie_is_id_ascending():
    assert [r['image_id'] for r in rank_entropy_scores([
        {'image_id': 'b', 'entropy_sum': 1.0}, {'image_id': 'a', 'entropy_sum': 1.0}
    ])] == ['a', 'b']


def test_near_tie_is_not_rounded():
    assert rank_entropy_scores([{'image_id': 'a', 'entropy_sum': 1.0},
                                {'image_id': 'b', 'entropy_sum': 1.0 + 1e-14}])[0]['image_id'] == 'b'


@pytest.mark.parametrize('p', [-0.1, 1.1, float('nan'), float('inf')])
def test_invalid_confidence(p):
    with pytest.raises(EntropyIntegrityError):
        binary_entropy(p)


def test_endpoint_limits():
    assert binary_entropy(0) == binary_entropy(1) == 0.0
    assert binary_entropy(1e-15) > 0


@pytest.fixture
def acquisition(tmp_path):
    ids = [f'{i:064x}' for i in range(4)]
    checkpoint = tmp_path / 'best.pt'
    checkpoint.write_bytes(b'synthetic checkpoint; never loaded as YOLO')
    rows = []
    for i in ids[1:]:
        image = tmp_path / f'{i}.jpg'
        image.write_bytes(b'synthetic image; never decoded')
        rows.append({'image_id': i, 'image_path': str(image)})
    def predict(checkpoint, records, config):
        for row, values in zip(records, [[0.5, 0.5], [], [0.9]]):
            yield row['image_id'], values
    return dict(checkpoint=checkpoint, unlabeled_manifest=rows, query_size=2,
                inference_config={**PINNED_INFERENCE, 'software_version': {'fixture': '1'},
                                  'source_sha256': {'fixture': '1'}},
                labeled_ids=ids[:1], train_ids=ids,
                checkpoint_sha256=file_sha256(checkpoint), _predict_fn=predict)


def test_labeled_in_scorer_input_hard_failure(acquisition):
    acquisition['labeled_ids'].append(acquisition['unlabeled_manifest'][0]['image_id'])
    with pytest.raises(EntropyIntegrityError, match='overlap'):
        entropy_acquisition(**acquisition)


def test_two_calls_identical_all_hashes(acquisition):
    a = entropy_acquisition(**acquisition)
    acquisition['unlabeled_manifest'].reverse()
    b = entropy_acquisition(**acquisition)
    assert a == b
    assert set(a['hashes']) == {'score_table', 'ranking', 'selected_ids'}
    assert len(a['next_labeled_ids']) == 3
    assert len(a['next_unlabeled_ids']) == 1


@pytest.mark.parametrize('key', list(PINNED_INFERENCE) + ['software_version', 'source_sha256'])
def test_each_missing_config_field_hard_failure(acquisition, key):
    del acquisition['inference_config'][key]
    with pytest.raises(EntropyIntegrityError, match='config fields'):
        entropy_acquisition(**acquisition)


def test_no_gt_fields_allowed(acquisition):
    acquisition['unlabeled_manifest'][0]['gt_label'] = 'forbidden'
    with pytest.raises(EntropyIntegrityError, match='only ID'):
        entropy_acquisition(**acquisition)


@pytest.mark.parametrize('case', ['duplicate', 'missing', 'foreign'])
def test_prediction_coverage_hard_failure(acquisition, case):
    ids = [r['image_id'] for r in acquisition['unlabeled_manifest']]
    output = {'duplicate': [ids[0], ids[0]], 'missing': ids[:1], 'foreign': ['f' * 64]}[case]
    acquisition['_predict_fn'] = lambda *args: ((i, [0.5]) for i in output)
    with pytest.raises(EntropyIntegrityError):
        entropy_acquisition(**acquisition)


@pytest.mark.parametrize('size', [0, -1, 4, True])
def test_invalid_query_size(acquisition, size):
    acquisition['query_size'] = size
    with pytest.raises(EntropyIntegrityError):
        entropy_acquisition(**acquisition)


def test_checkpoint_hash_mismatch(acquisition):
    acquisition['checkpoint_sha256'] = '0' * 64
    with pytest.raises(EntropyIntegrityError, match='checkpoint'):
        entropy_acquisition(**acquisition)


def test_fixed_pool_missing_id(acquisition):
    acquisition['train_ids'] = acquisition['train_ids'][:-1]
    with pytest.raises(EntropyIntegrityError, match='fixed train pool'):
        entropy_acquisition(**acquisition)


@pytest.mark.parametrize('p', [0.25, 0.1, float('nan'), 1.1])
def test_retained_detection_bounds(acquisition, p):
    acquisition['_predict_fn'] = lambda cp, rows, cfg: ((r['image_id'], [p]) for r in rows)
    with pytest.raises(EntropyIntegrityError):
        entropy_acquisition(**acquisition)


def test_all_zero_scores_select_canonical_ids(acquisition):
    acquisition['_predict_fn'] = lambda cp, rows, cfg: ((r['image_id'], []) for r in rows)
    result = entropy_acquisition(**acquisition)
    assert result['selected_ids'] == sorted(r['image_id'] for r in acquisition['unlabeled_manifest'])[:2]


def test_config_change_rejected(acquisition):
    acquisition['inference_config']['conf'] = 0.1
    with pytest.raises(EntropyIntegrityError, match='unfrozen'):
        entropy_acquisition(**acquisition)
