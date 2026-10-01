"""Model QC: Moran's I and its permutation variance per PC, vs the verbatim origin/dev loop.

origin/dev (tests/legacy/qc_model.py, db00d98) rebuilt the same Queen weights for every PC
and ran esda's serial 999-permutation loop. The new code builds the weights once and runs
the permutations through spoqc.core.moran, which reproduces esda's global-RNG draws in
order. The acceptance bar is bit-identity of every number the step reports (moran_I,
spatial_variance, variance_explained) and of the global RNG state it leaves behind.
"""

import types

import numpy as np
import pandas as pd
import pytest
from esda.moran import Moran
from libpysal.weights import Queen
import geopandas as gpd

from spoqc import helperfuncs
from spoqc.core import moran as core_moran
from spoqc.subworkflows import qc_model

from conftest import load_legacy

legacy = load_legacy("qc_model", "spoqc.subworkflows")

N_POINTS = 2_500
N_PCS = 4


def _frame(dtype, seed=0):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 5_000, N_POINTS), rng.uniform(0, 5_000, N_POINTS)
    df = pd.DataFrame({"x": x, "y": y})
    for i in range(N_PCS):
        signal = np.sin(x / (300 + 100 * i)) * np.cos(y / 400)
        df[f"PC{i}"] = (signal * (i % 2) + rng.normal(size=N_POINTS)).astype(dtype)
    return df


class _Figure:
    """Stands in for make_subplots(); records what the step would plot."""

    def __init__(self, record):
        self.record = record

    def add_trace(self, trace, secondary_y):
        self.record.append((trace.name, np.asarray(trace.y).copy()))

    def update_layout(self, **kw):
        pass

    def update_yaxes(self, **kw):
        pass

    def write_html(self, *a, **kw):
        pass

    def write_image(self, *a, **kw):
        pass


def _run(module, monkeypatch, df, seed, **kw):
    """The step's reported traces and the global RNG state it leaves."""
    record = []
    monkeypatch.setattr(module, "make_subplots", lambda **k: _Figure(record))
    monkeypatch.setattr(helperfuncs, "apply_general_plotly_layout", lambda fig, b: None)
    if hasattr(module, "save_figure"):
        monkeypatch.setattr(module, "save_figure", lambda *a, **k: None)
    sdata = {
        "table": types.SimpleNamespace(uns={"pca": {"variance": np.linspace(3, 1, 10)}})
    }
    np.random.seed(seed)
    module.plot_spatial_vs_exression_variance(sdata, "unused", df.copy(), N_PCS, **kw)
    return record, np.random.get_state()


def _assert_same(a, b):
    (rec_a, st_a), (rec_b, st_b) = a, b
    assert [n for n, _ in rec_a] == [n for n, _ in rec_b]
    for (name, ya), (_, yb) in zip(rec_a, rec_b):
        assert ya.dtype == yb.dtype, name
        np.testing.assert_array_equal(ya, yb, err_msg=name, strict=True)
        assert ya.tobytes() == yb.tobytes(), name  # bit-identical, incl. signed zeros
    assert st_a[0] == st_b[0] and st_a[2:] == st_b[2:]
    np.testing.assert_array_equal(st_a[1], st_b[1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("seed", [123, 7])
def test_step_numbers_and_rng_state_match_origin_dev(monkeypatch, dtype, seed):
    df = _frame(dtype)
    old = _run(legacy, monkeypatch, df, seed)
    new = _run(qc_model, monkeypatch, df, seed)
    _assert_same(old, new)


def test_single_thread_matches_too(monkeypatch):
    import numba

    df = _frame(np.float32, seed=3)
    threads = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        _assert_same(_run(legacy, monkeypatch, df, 123), _run(qc_model, monkeypatch, df, 123))
    finally:
        numba.set_num_threads(threads)


def test_one_moran_matches_esda_attributes():
    """I and VI_sim against esda's own object, with a chunk count that is not a multiple."""
    df = _frame(np.float32, seed=5)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    np.random.seed(11)
    ref = Moran(df["PC1"], Queen.from_dataframe(gdf), permutations=999)
    np.random.seed(11)
    new = core_moran.moran(df["PC1"], core_moran.queen_weights(df[["x", "y"]].to_numpy()), 999)
    assert new.I == ref.I and new.VI_sim == ref.VI_sim


def test_queen_weights_consume_no_rng():
    """Hoisting Queen out of the PC loop is only exact if building it draws nothing."""
    df = _frame(np.float64)
    np.random.seed(1)
    Queen.from_dataframe(gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y)))
    after = np.random.get_state()[1].copy()
    np.random.seed(1)
    np.testing.assert_array_equal(after, np.random.get_state()[1])


# ----------------------------------------------------------------- mutants must fail


def _mutant_scaled_z(y, w, permutations):
    y = np.asarray(y, dtype=np.float64)
    return _real_moran(y / y.std(), w, permutations)


def _mutant_fewer_permutations(y, w, permutations):
    return _real_moran(y, w, permutations - 1)


_real_moran = core_moran.moran


@pytest.mark.parametrize("mutant", [_mutant_scaled_z, _mutant_fewer_permutations])
def test_mutants_are_caught(monkeypatch, mutant):
    df = _frame(np.float32)
    old = _run(legacy, monkeypatch, df, 123)
    monkeypatch.setattr(qc_model, "moran_test", mutant)
    new = _run(qc_model, monkeypatch, df, 123)
    with pytest.raises(AssertionError):
        _assert_same(old, new)


def test_mutant_skipped_draw_is_caught(monkeypatch):
    """Drawing one permutation too many shifts every later PC's RNG stream."""
    df = _frame(np.float32)
    old = _run(legacy, monkeypatch, df, 123)
    real = qc_model.moran_test

    def extra_draw(*a, **k):
        np.random.permutation(3)
        return real(*a, **k)

    monkeypatch.setattr(qc_model, "moran_test", extra_draw)
    with pytest.raises(AssertionError):
        _assert_same(old, _run(qc_model, monkeypatch, df, 123))
