"""The nonzero-only accumulation must be BIT-identical to ovrlpy, not merely close.

Skipping a row whose blurred signal is exactly zero omits `0.0 * factor_c`, which is +-0.0
for finite loadings and leaves the accumulator unchanged. Retained rows go through the same
two rounding steps as ovrlpy's -- one multiply, one add, same values, no reassociation and
no FMA fusion -- so `np.array_equal` is the right assertion and `np.allclose` would be too
weak to catch a real regression.
"""
from __future__ import annotations

from queue import SimpleQueue

import numpy as np
import polars as pl
import pytest

ovrlpy = pytest.importorskip("ovrlpy")
from ovrlpy._utils import _calculate_embedding as ovrlpy_original  # noqa: E402

from spoqc._ovrlpy_fast import _calculate_embedding_sparse  # noqa: E402


def _items(n_genes, side, rng, *, n_min=2, n_max=120, z_center=0.5):
    items = []
    for gene in range(n_genes):
        n = int(rng.integers(n_min, n_max))
        items.append((gene, pl.DataFrame({
            "x_pixel": rng.integers(0, side, n),
            "y_pixel": rng.integers(0, side, n),
            "z": rng.random(n),
            "z_center": np.full(n, z_center),
        })))
    return items


def _run(fn, mask, components, items, **kwargs):
    queue: SimpleQueue = SimpleQueue()
    for item in items:
        queue.put(item)
    return fn(queue, mask, components, bandwidth=1.0, dtype=np.float32, **kwargs)


@pytest.mark.parametrize("components_dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "n_genes,side,n_components,sparsity",
    [(40, 60, 21, 0.3), (120, 90, 30, 0.3), (60, 80, 30, 0.9)],
)
def test_bit_identical_to_ovrlpy(n_genes, side, n_components, sparsity, components_dtype):
    """The headline claim, across dense and very sparse masks.

    float32 is the production dtype: ovrlpy fits its PCA on float32 pseudocells, so
    `pca.components_` is float32 and ovrlpy accumulates in float32.
    """
    rng = np.random.default_rng(0)
    mask = rng.random((side, side)) > sparsity
    components = rng.standard_normal((n_components, n_genes)).astype(components_dtype)
    items = _items(n_genes, side, rng)

    expected = _run(ovrlpy_original, mask, components, items)
    got = _run(_calculate_embedding_sparse, mask, components, items)

    for want, have in zip(expected, got):
        assert want.shape == have.shape
        assert want.dtype == have.dtype
        assert np.array_equal(want, have), (
            f"max abs diff {np.abs(want - have).max():.3e} over "
            f"{np.count_nonzero(want != have):,} of {want.size:,} entries"
        )


def test_empty_queue_returns_integer_sentinel():
    """ovrlpy's caller checks `isinstance(x, int) and x == 0`, so 0 must stay an int."""
    rng = np.random.default_rng(1)
    mask = rng.random((20, 20)) > 0.3
    top, bottom = _run(_calculate_embedding_sparse, mask, rng.standard_normal((5, 3)), [])
    assert isinstance(top, int) and top == 0
    assert isinstance(bottom, int) and bottom == 0


def test_single_transcript_genes_are_skipped_like_ovrlpy():
    """`len(df) < 2` genes are dropped by ovrlpy; a lone transcript is not negligible."""
    rng = np.random.default_rng(2)
    side = 40
    mask = rng.random((side, side)) > 0.3
    components = rng.standard_normal((8, 6))
    items = _items(6, side, rng, n_min=1, n_max=2)
    assert all(len(df) == 1 for _, df in items), "fixture must be single-transcript"

    assert _run(ovrlpy_original, mask, components, items) == (0, 0)
    assert _run(_calculate_embedding_sparse, mask, components, items) == (0, 0)


def test_z_equal_to_centre_is_dropped_from_both_sides():
    """ovrlpy filters `z > z_center` and `z < z_center`, so equality falls out of both."""
    side = 30
    mask = np.ones((side, side), dtype=bool)
    components = np.ones((4, 2))
    items = [(g, pl.DataFrame({
        "x_pixel": np.array([3, 9, 14]),
        "y_pixel": np.array([4, 8, 12]),
        "z": np.full(3, 0.5),
        "z_center": np.full(3, 0.5),
    })) for g in range(2)]

    assert _run(ovrlpy_original, mask, components, items) == (0, 0)
    assert _run(_calculate_embedding_sparse, mask, components, items) == (0, 0)


def test_all_zero_signal_rows_do_not_change_the_result():
    """A gene confined to one corner leaves most rows zero -- exactly the skipped case."""
    rng = np.random.default_rng(3)
    side = 70
    mask = np.ones((side, side), dtype=bool)
    components = rng.standard_normal((30, 4))
    items = [(g, pl.DataFrame({
        "x_pixel": rng.integers(0, 6, 20),
        "y_pixel": rng.integers(0, 6, 20),
        "z": rng.random(20),
        "z_center": np.full(20, 0.5),
    })) for g in range(4)]

    expected = _run(ovrlpy_original, mask, components, items)
    got = _run(_calculate_embedding_sparse, mask, components, items)
    for want, have in zip(expected, got):
        zero_rows = np.count_nonzero(~np.any(want != 0, axis=1))
        assert zero_rows > side * side // 2, "fixture must leave most rows zero"
        assert np.array_equal(want, have)


def test_non_finite_loading_is_rejected_not_silently_handled():
    """`0.0 * inf` is NaN, which ovrlpy propagates and skipping would not."""
    rng = np.random.default_rng(4)
    side = 25
    mask = np.ones((side, side), dtype=bool)
    components = rng.standard_normal((6, 3))
    components[2, 1] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        _run(_calculate_embedding_sparse, mask, components, _items(3, side, rng))


def _synthetic_transcripts(seed=0, n=60_000, n_genes=40, n_cells=200, side=300.0):
    """Clustered transcripts: cells of 5 types, 8 genes per type, so ovrlpy finds pseudocells."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, side, (n_cells, 2))
    cell_type = rng.integers(0, 5, n_cells)
    cell = rng.integers(0, n_cells, n)
    xy = centres[cell] + rng.normal(0, 3, (n, 2))
    gene = (cell_type[cell] * 8 + rng.integers(0, 8, n)) % n_genes
    return pd.DataFrame({
        "gene": [f"g{i}" for i in gene],
        "x": xy[:, 0],
        "y": xy[:, 1],
        "z": rng.uniform(0, 10, n),
    })


def _in_gene_order(fn):
    """Run `fn` on the patch's genes in gene-index order.

    ovrlpy 1.2.0 fills each patch's queue from `patch_df.group_by("gene")`, whose order
    polars does not maintain, so the float32 sum over genes -- stock or shim -- differs
    run to run in the last bits. Fixing the order leaves only the accumulation to differ.
    """
    def ordered(genes, mask, components, **kwargs):
        items = []
        while not genes.empty():
            items.append(genes.get())
        queue: SimpleQueue = SimpleQueue()
        for item in sorted(items, key=lambda item: item[0]):
            queue.put(item)
        return fn(queue, mask, components, **kwargs)
    return ordered


def test_compute_vsi_integrity_map_is_bit_identical_end_to_end(monkeypatch):
    """The whole compute_VSI stage on one fitted model: stock accumulation, then the shim.

    The model is fitted once, so `pca.components_` (float32, as in production) is shared;
    n_workers=1 and a fixed gene order fix the order the embeddings are summed in. The
    only difference between the two passes is the accumulation. The stock pass is run
    twice to prove the ordering makes it reproducible, so the comparison can be exact.
    """
    from ovrlpy import _ovrlp

    model = ovrlpy.Ovrlp(
        _synthetic_transcripts(), n_components=5, n_workers=1, random_state=0,
        patch_length=100,
    )
    model.process_coordinates(gridsize=1, n_iter=20)
    model.fit_transcripts(fit_umap=False)
    assert model.pca.components_.dtype == np.float32, "fixture must match production dtype"

    def integrity_map(fn):
        monkeypatch.setattr(_ovrlp, "_calculate_embedding", _in_gene_order(fn))
        model.compute_VSI()
        return model.integrity_map.copy()

    stock = integrity_map(ovrlpy_original)
    assert np.array_equal(stock, integrity_map(ovrlpy_original)), "stock must reproduce"
    assert np.count_nonzero(stock) > stock.size // 4, "fixture must produce signal"

    fast = integrity_map(_calculate_embedding_sparse)

    assert fast.dtype == stock.dtype
    assert np.array_equal(stock, fast), (
        f"max abs diff {np.abs(stock - fast).max():.3e} over "
        f"{np.count_nonzero(stock != fast):,} of {stock.size:,} pixels"
    )
