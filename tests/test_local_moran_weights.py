"""local_moran_I.KNNWeights vs libpysal's KNN.from_array(...).sparse with transform 'r'.

KNNWeights stands in for the CSR in moran_I_all_genes: it must give the same column set per row
(including libpysal's tie-breaking through its KD-tree), the same sum and the same weights @ z.
"""

import numpy as np
import pytest
from conftest import assert_same_array
from libpysal.weights import KNN

from spoqc.metrics.transcript_density import local_moran_I

K = 30


def assert_knn_weights_equal(coords, k=K, seed=0):
    knn = KNN.from_array(coords, k=k, silence_warnings=True)
    knn.transform = "r"
    ref = knn.sparse
    new = local_moran_I.KNNWeights(coords, k)
    n = len(coords)
    assert ref.has_sorted_indices
    assert_same_array(
        np.asarray(ref.indptr, dtype=np.int64),
        np.arange(n + 1, dtype=np.int64) * k,
        "indptr",
    )
    assert_same_array(
        new.neighbours.ravel(), np.asarray(ref.indices, dtype=np.int64), "neighbours"
    )
    assert_same_array(np.full(n * k, new.weight), ref.data, "weights")
    assert_same_array(np.float64(new.sum()), np.float64(ref.sum()), "sum")
    z = np.random.default_rng(seed).normal(0, 1, (n, 7)).astype(np.float32)
    assert_same_array(new @ z, ref @ z, "weights @ z")


def test_random_points_match_libpysal():
    rng = np.random.default_rng(1)
    for n in (K + 1, K + 2, 57, 140):
        assert_knn_weights_equal(rng.uniform(0, 200, (n, 2)))


def test_lattice_ties_fall_back_to_the_kd_tree_and_match():
    # every point of a lattice has many neighbours at exactly the same distance
    xy = (
        np.stack(np.meshgrid(np.arange(8.0), np.arange(7.0)), axis=-1).reshape(-1, 2)
        * 10
    )
    assert not local_moran_I._unambiguous_knn(
        xy, K, np.empty((len(xy), K), dtype=np.int64)
    )
    assert_knn_weights_equal(xy)


@pytest.mark.parametrize("n_same", [2, K + 1, K + 3])
def test_coincident_points_match_libpysal(n_same):
    # up to and beyond k + 1 cells at one spot: libpysal's has_one_too_many self-drop
    rng = np.random.default_rng(2)
    xy = np.concatenate([np.full((n_same, 2), 50.0), rng.uniform(0, 200, (60, 2))])
    assert_knn_weights_equal(rng.permutation(xy))


def test_brute_force_path_is_taken_without_ties():
    xy = np.random.default_rng(3).uniform(0, 200, (80, 2))
    assert local_moran_I._unambiguous_knn(xy, K, np.empty((80, K), dtype=np.int64))


def test_mutant_leafsize_16_is_detected(monkeypatch):
    """The fallback must build libpysal's tree: with a lattice, scipy's default leaf size picks other ties."""
    xy = np.stack(np.meshgrid(np.arange(9.0), np.arange(9.0)), axis=-1).reshape(-1, 2)
    monkeypatch.setattr(local_moran_I, "KNN_LEAFSIZE", 16)
    with pytest.raises(AssertionError):
        assert_knn_weights_equal(xy)


def _near_tie_neighbourhood(ulps, seed=5):
    """Point 0 at the origin; K other points on a circle of radius 40, a (K + 1)-th whose squared
    distance is `ulps` ULPs larger, then far points: the gap at point 0's cut-off is `ulps` ULPs."""
    rng = np.random.default_rng(seed)
    angles = rng.permutation(np.linspace(0, 2 * np.pi, K + 1, endpoint=False))
    ring = np.column_stack((np.cos(angles), np.sin(angles))) * 40.0
    d2 = float((ring[-1] ** 2).sum())
    target = d2
    for _ in range(ulps):
        target = np.nextafter(target, np.inf)
    ring[-1] *= np.sqrt(target / d2)
    far = rng.uniform(150, 300, (40, 2))
    return np.concatenate([[[0.0, 0.0]], ring, far]), target - d2


@pytest.mark.parametrize("ulps", [1, 2, 5, 40])
def test_near_ties_at_the_cutoff_take_the_kd_tree_fallback(ulps):
    xy, gap = _near_tie_neighbourhood(ulps)
    assert gap > 0 or ulps == 0
    assert not local_moran_I._unambiguous_knn(xy, K, np.empty((len(xy), K), dtype=np.int64))
    assert_knn_weights_equal(xy)

