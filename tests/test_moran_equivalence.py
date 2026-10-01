"""Differential tests: core.moran against esda's Moran and the verbatim origin/dev callers.

tests/legacy/qc_model.py and tests/legacy/global_moran_I.py are origin/dev db00d98's
modules, unchanged. Every Moran attribute, its dtype and order, and the full global
numpy RNG state afterwards must match bit for bit.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from anndata import AnnData
from esda.moran import Moran
from libpysal.weights import Queen
import geopandas as gpd

from conftest import load_legacy
from spoqc.core import moran as core_moran
from spoqc.metrics.transcript_density import global_moran_I
from spoqc.subworkflows import qc_model

ATTRS = [
    "I",
    "EI",
    "VI_norm",
    "VI_rand",
    "z_norm",
    "z_rand",
    "p_norm",
    "p_rand",
    "z2ss",
    "z",
    "sim",
    "p_sim",
    "EI_sim",
    "seI_sim",
    "VI_sim",
    "z_sim",
    "p_z_sim",
]


def points(n, rng, clustered):
    if clustered:  # dense blobs plus background, like tissue
        centres = rng.uniform(0, 1000, (8, 2))
        xy = centres[rng.integers(0, 8, n)] + rng.normal(0, 25, (n, 2))
        xy[: n // 5] = rng.uniform(0, 1000, (n // 5, 2))
        return xy
    return rng.uniform(0, 1000, (n, 2))


def values(xy, rng, dtype):
    # a spatial trend plus mixed-magnitude noise, so summation order shows in the last bits
    y = np.sin(xy[:, 0] / 90.0) + rng.standard_normal(len(xy)) * 10.0 ** rng.integers(
        -3, 4, len(xy)
    )
    return y.astype(dtype)


def legacy_queen(xy):
    """origin/dev's weights: plain libpysal Queen from a points GeoDataFrame."""
    gdf = gpd.GeoDataFrame({"x": xy[:, 0], "y": xy[:, 1]}, geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]))
    return Queen.from_dataframe(gdf)


def rng_state():
    return np.random.get_state()


def assert_same_state(a, b):
    assert a[0] == b[0]
    assert np.array_equal(a[1], b[1]) and a[1].dtype == b[1].dtype
    assert a[2:] == b[2:]


def assert_same_moran(ref, new):
    for attr in ATTRS:
        r, t = np.asarray(getattr(ref, attr)), np.asarray(getattr(new, attr))
        assert r.dtype == t.dtype, attr
        assert r.shape == t.shape, attr
        assert np.array_equal(r, t), attr


@pytest.mark.parametrize("n", [9, 60, 1000, 6000])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("permutations", [1, 99, 999])
@pytest.mark.parametrize("predraws", [0, 3, 620])  # start mid-block and next to a twist
def test_moran_matches_esda(n, dtype, permutations, predraws):
    rng = np.random.default_rng(n + permutations)
    xy = points(n, rng, clustered=n > 100)
    y = values(xy, rng, dtype)

    np.random.seed(123)
    np.random.randint(0, 2**31, predraws)
    ref = Moran(y, legacy_queen(xy), permutations=permutations)
    ref_state = rng_state()

    np.random.seed(123)
    np.random.randint(0, 2**31, predraws)
    new = core_moran.moran(y, core_moran.queen_weights(xy), permutations=permutations)
    assert_same_state(ref_state, rng_state())
    assert_same_moran(ref, new)


def test_moran_without_permutations_matches_esda():
    rng = np.random.default_rng(1)
    xy = points(500, rng, True)
    y = values(xy, rng, np.float64)
    state = rng_state()
    ref = Moran(y, legacy_queen(xy), permutations=0)
    new = core_moran.moran(y, core_moran.queen_weights(xy), permutations=0)
    assert_same_state(state, rng_state())
    for attr in ["I", "z", "p_norm", "VI_rand"]:
        assert np.array_equal(getattr(ref, attr), getattr(new, attr))


def test_pca_loop_with_one_weights_matches_fresh_weights_per_pc():
    """origin/dev rebuilt the weights for every PC; one W reused gives the same numbers and draws."""
    rng = np.random.default_rng(7)
    xy = points(3000, rng, True)
    ys = [values(xy, rng, np.float64) for _ in range(4)]

    np.random.seed(123)
    ref = [Moran(y, legacy_queen(xy), permutations=199) for y in ys]
    ref_state = rng_state()

    np.random.seed(123)
    w = core_moran.queen_weights(xy)
    new = [core_moran.moran(y, w, permutations=199) for y in ys]
    assert_same_state(ref_state, rng_state())
    for r, t in zip(ref, new):
        assert_same_moran(r, t)


@pytest.mark.parametrize(
    "n", list(range(1, 300)) + [8191, 8192, 8193, 167779, 167780, 1_000_003]
)
def test_numpy_sum_replica(n):
    rng = np.random.default_rng(n)
    a = rng.standard_normal(n) * 10.0 ** rng.integers(-6, 7, n)
    assert core_moran._numpy_sum(a) == a.sum()


def capture_figures(module, monkeypatch):
    figures = []
    real = module.make_subplots

    def capturing(*args, **kwargs):
        fig = real(*args, **kwargs)
        figures.append(fig)
        return fig

    monkeypatch.setattr(module, "make_subplots", capturing)
    return figures


def test_spatial_variance_figures_match_legacy(monkeypatch, tmp_path):
    """The numbers qc_model plots (variance explained, Moran's I, permutation variance)."""
    legacy = load_legacy("qc_model", "spoqc.subworkflows")
    monkeypatch.setattr(go.Figure, "write_image", lambda *a, **k: None)
    monkeypatch.setattr(go.Figure, "write_html", lambda *a, **k: None)
    monkeypatch.setattr(qc_model, "save_figure", lambda *a, **k: None)

    rng = np.random.default_rng(11)
    n, n_pcs = 2500, 5
    xy = points(n, rng, True)
    df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]})
    for i in range(n_pcs):
        df[f"PC{i}"] = values(xy, rng, np.float32)
    table = AnnData(np.zeros((n, 1)))
    table.uns["pca"] = {"variance": rng.random(n_pcs + 3).astype(np.float32)}
    sdata = {"table": table}

    ref_figs = capture_figures(legacy, monkeypatch)
    np.random.seed(123)
    legacy.plot_spatial_vs_exression_variance(sdata, str(tmp_path), df, n_pcs)
    ref_state = rng_state()

    new_figs = capture_figures(qc_model, monkeypatch)
    np.random.seed(123)
    qc_model.plot_spatial_vs_exression_variance(sdata, str(tmp_path), df, n_pcs)
    assert_same_state(ref_state, rng_state())

    assert len(ref_figs) == len(new_figs) == 2
    for rf, nf in zip(ref_figs, new_figs):
        for rt, nt in zip(rf.data, nf.data):
            assert list(rt.x) == list(nt.x)
            r, t = np.asarray(rt.y), np.asarray(nt.y)
            assert r.dtype == t.dtype and np.array_equal(r, t)


def test_global_moran_weights_match_legacy():
    legacy = load_legacy("global_moran_I", "spoqc.metrics.transcript_density")
    rng = np.random.default_rng(3)
    coords = points(2000, rng, True)
    gdf = legacy.gpd.GeoDataFrame(
        {"x": coords[:, 0], "y": coords[:, 1]},
        geometry=legacy.gpd.points_from_xy(coords[:, 0], coords[:, 1]),
    )
    ref = legacy.Queen.from_dataframe(gdf)
    ref.transform = "r"
    new = core_moran.queen_weights(coords)
    new.transform = "r"
    for attr in ["indptr", "indices", "data"]:
        assert np.array_equal(getattr(ref.sparse, attr), getattr(new.sparse, attr))
    X = rng.poisson(2.0, (2000, 6)).astype(np.float64)
    assert np.array_equal(
        legacy.moran_I_all_genes(X, ref.sparse),
        global_moran_I.moran_I_all_genes(X, new.sparse),
        equal_nan=True,
    )
