"""Small numerical/boundary tests; no project data or GPU required."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("geometry_audit", Path(__file__).with_name("analyze.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_cosine_invariant_to_positive_rescaling_and_no_input_mutation():
    features = np.array([[1., 0], [0, 3.], [-2., 0]], dtype=np.float32)
    old = features.copy()
    z, d = audit.cosine_matrix(features)
    np.testing.assert_allclose(d, [[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    np.testing.assert_array_equal(features, old)
    assert z.dtype == np.float64


def test_geometry_excludes_diagonal_and_uses_per_image_mean_five_neighbors():
    theta = np.array([0., .1, .3, .6, .9, 1.3])
    z, d = audit.cosine_matrix(np.column_stack([np.cos(theta), np.sin(theta)]))
    got = audit.geometry(z, d, np.arange(6))
    expected_pairs = [1 - np.cos(theta[i] - theta[j]) for i in range(6) for j in range(i + 1, 6)]
    expected_knn = [np.mean([1 - np.cos(theta[i] - theta[j]) for j in range(6) if i != j]) for i in range(6)]
    assert got["within_class_pair_median"] == pytest.approx(np.median(expected_pairs))
    assert got["within_class_knn5_mean_median"] == pytest.approx(np.median(expected_knn))
    assert got["centroid_distance_p90"] >= got["centroid_distance_median"]


def test_multilabel_overlap_is_not_exact_equality():
    sets, masks = audit.label_masks(["1;2", "2", "3"])
    assert sets[0] != sets[1]
    assert (masks[0] & masks[1]) != 0
    assert (masks[0] & masks[2]) == 0


def test_whole_set_null_matches_hand_enumerated_three_vertex_cycle():
    _, masks = audit.label_masks(["1;2", "2", "3"])
    # Every permutation on this cycle uses each undirected pair once.
    values = audit.chance_distribution(masks, np.array([1, 2, 0]), 100, 17)
    np.testing.assert_allclose(values, 1 / 3)
    np.testing.assert_array_equal(values, audit.chance_distribution(masks, np.array([1, 2, 0]), 100, 17))


def test_self_neighbor_and_zero_vector_are_rejected():
    with pytest.raises(RuntimeError, match="Self edge"):
        audit.chance_distribution(np.array([1, 2]), np.array([0, 1]), 2, 1)
    with pytest.raises(RuntimeError, match="Zero-norm"):
        audit.cosine_matrix(np.array([[0., 0.]]))


def test_rank_ties_and_constant_input():
    assert audit.rank_correlation([1, 1, 3], [3, 3, 1]) == pytest.approx(-1)
    assert audit.rank_correlation([1, 2, 3], [3, 1, 2]) == pytest.approx(-.5)
    assert audit.rank_correlation([1, 1, 1], [3, 1, 2]) is None


def test_forbidden_source_never_opened():
    with pytest.raises(RuntimeError, match="Forbidden input"):
        audit.record(Path("images/nonexistent.jpg"), "test")
    with pytest.raises(RuntimeError, match="Private split/source"):
        audit.record(Path("split_assignment.csv"), "test")
