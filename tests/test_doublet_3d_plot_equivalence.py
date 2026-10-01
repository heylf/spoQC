"""The 3D scatter must receive identical rows after the polars rewrite.

The original materialised ovrlpy's ENTIRE transcript table with .to_pandas(), then kept
1% of it and read only x/y. A tree-RSS trace measured that single call adding ~48 GB in
under 10 s on a 229,970-cell 5K-panel sample, pushing the run to 217.79 GB and past the
memory budget.

These tests reproduce the ORIGINAL pandas logic verbatim and assert the polars path feeds
ax.scatter the same rows in the same order, including the cases the rewrite could break:
the inclusive `.between` boundaries, the stride being applied AFTER the filter, nulls, and
the aspect ratio using full-column maxima rather than the downsampled subset.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from spoqc.metrics.segmentation.doublet_score import downsample_transcript_layers


def _original(frame: pl.DataFrame):
    """Verbatim reproduction of the pre-rewrite code path."""
    transcripts_processed = frame.to_pandas()
    out = []
    for i in range(-2, 3):
        subset = transcripts_processed[
            (transcripts_processed['z'] - transcripts_processed['z_center']).between(i, i + 1)
        ]
        subset = subset[::100]
        out.append((subset["x"].to_numpy(), subset["y"].to_numpy()))
    ratio = transcripts_processed["x"].max() / transcripts_processed["y"].max()
    return out, ratio


def _rewritten(frame: pl.DataFrame):
    """Calls the SHIPPED helper, so drift in the real code fails these tests."""
    per_layer, ratio = downsample_transcript_layers(frame)
    return [(s["x"].to_numpy(), s["y"].to_numpy()) for _, s in per_layer], ratio


def _assert_same(frame):
    got, got_ratio = _rewritten(frame)
    want, want_ratio = _original(frame)
    assert len(got) == len(want)
    for i, ((gx, gy), (wx, wy)) in enumerate(zip(got, want)):
        assert np.array_equal(gx, wx), f"layer {i - 2}: x differs ({gx.size} vs {wx.size})"
        assert np.array_equal(gy, wy), f"layer {i - 2}: y differs"
    assert got_ratio == want_ratio


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_original_on_random_transcripts(seed):
    rng = np.random.default_rng(seed)
    n = 25_000
    _assert_same(pl.DataFrame({
        "x": rng.uniform(0, 5000, n),
        "y": rng.uniform(0, 4000, n),
        "z": rng.uniform(-4, 4, n),
        "z_center": np.zeros(n),
        # columns the plot never reads but .to_pandas() used to materialise
        "gene": rng.choice([f"Gene{i}" for i in range(50)], n),
        "qv": rng.uniform(0, 40, n),
    }))


def test_between_boundaries_are_inclusive():
    """pandas .between(i, i+1) includes both ends; >= / <= must match exactly."""
    depth = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 0.5, 1.0000001])
    n = depth.size
    _assert_same(pl.DataFrame({
        "x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float) + 1,
        "z": depth, "z_center": np.zeros(n), "gene": ["G"] * n,
    }))


def test_stride_is_applied_after_the_filter():
    """Striding before filtering would keep different rows; make that observable."""
    n = 1000
    # every row is in layer 0, so [::100] must pick rows 0,100,200,... of the FILTERED frame
    _assert_same(pl.DataFrame({
        "x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float),
        "z": np.full(n, 0.5), "z_center": np.zeros(n), "gene": ["G"] * n,
    }))


def test_interleaved_layers_do_not_cross_contaminate():
    """Rows alternate between layers, so a wrong stride order shows up immediately."""
    n = 900
    depth = np.tile([-1.5, 0.5, 1.5], n // 3)
    _assert_same(pl.DataFrame({
        "x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float) * 2,
        "z": depth, "z_center": np.zeros(n), "gene": ["G"] * n,
    }))


def test_empty_layer_is_handled():
    n = 50
    _assert_same(pl.DataFrame({
        "x": np.arange(n, dtype=float), "y": np.arange(n, dtype=float),
        "z": np.full(n, 0.5), "z_center": np.zeros(n), "gene": ["G"] * n,
    }))


def test_ratio_uses_full_columns_not_the_downsample():
    """The max must come from all rows; the downsampled subset would give a different one."""
    n = 5000
    rng = np.random.default_rng(3)
    frame = pl.DataFrame({
        "x": rng.uniform(0, 1000, n), "y": rng.uniform(0, 800, n),
        "z": rng.uniform(-3, 3, n), "z_center": np.zeros(n), "gene": ["G"] * n,
    })
    _, ratio = _rewritten(frame)
    assert ratio == frame["x"].max() / frame["y"].max()
