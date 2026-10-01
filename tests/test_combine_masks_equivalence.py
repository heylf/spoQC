"""combine_masks vs the verbatim origin/dev function (tests/legacy/combine_masks.py, db00d98).

The step writes only figures, so its numbers are what it hands to the figure code: every
array given to plot_pixels and ax.hist, the Venn subset sizes and the title's uncovered
share. These tests record all of them from both implementations on synthetic masks
(written in the on-disk formats hqcr/hqpr/hqtr use) and assert exact equality of values,
dtypes and call order.
"""

import os

import dask.dataframe as dd
import matplotlib

matplotlib.use("Agg")
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from spoqc import helperfuncs
from spoqc.hqr import combine_masks

from conftest import load_legacy

legacy = load_legacy("combine_masks", "spoqc.hqr")

DIM_X, DIM_Y = 70, 90
STAINING = 0
_REAL_TITLE = plt.title
IMAGEDIM = helperfuncs.ImageDimStruct(0, 0, DIM_Y, DIM_X)


def _write_inputs(tmp, rng, binary=True, n_parts=7, nan_beliefs=False):
    n = DIM_X * DIM_Y

    def beliefs(values):
        """With nan_beliefs, NaN (stored as null) in scattered pixels and in the last rows only (one part)."""
        if nan_beliefs:
            values[rng.random(n) < 0.01] = np.nan
            values[-50:] = np.nan
        return values

    def mask(dtype):
        values = rng.integers(0, 2, n) if binary else rng.integers(-1, 3, n)
        return values.astype(dtype)

    for suf, bdtype, mdtype in [
        ("", np.float64, np.uint8),
        ("_smoothed", np.float32, np.int8),
    ]:
        pd.DataFrame(
            {
                f"hqcr_beliefs{suf}": beliefs(rng.random(n).astype(bdtype)),
                f"hqcr_mask{suf}": mask(mdtype),
            }
        ).to_parquet(f"{tmp}/hqcr_output_mask{suf}_raw.parquet")
        for prefix in [f"hqpr_{STAINING}", "hqtr"]:
            frame = pd.DataFrame(
                {
                    "cluster": rng.integers(0, 5, n).astype(np.int32),
                    f"{prefix}_beliefs{suf}": beliefs(rng.random(n)),
                    f"{prefix}_mask{suf}": mask(np.int64),
                }
            )
            # uneven partitions, more than 9 of them: part.10 sorts after part.1 lexically
            dd.from_pandas(frame, npartitions=n_parts, sort=False).to_parquet(
                f"{tmp}/{prefix}_output_mask{suf}_raw", write_index=False
            )


def _cmap_key(cmap):
    """A colormap by value (a LinearSegmentedColormap's repr is its address)."""
    return cmap if isinstance(cmap, str) else (cmap.name, cmap(np.linspace(0, 1, 16)).tobytes())


def _record(module, monkeypatch, tmp, fig_path, threads=None):
    calls = []

    def plot_pixels(
        figure_path, image, imagedim, suffix, title, cmap, axis_off, baroff, **kw
    ):
        image = np.asarray(image)
        calls.append(
            (
                "pixels",
                suffix,
                title,
                _cmap_key(cmap),
                axis_off,
                baroff,
                str(kw),
                image.shape,
                image.dtype,
                image.tobytes(),
            )
        )

    def hist(self, x, *a, **kw):
        x = np.asarray(x)
        calls.append(("hist", x.dtype, x.tobytes(), a, str(kw)))

    def venn3(subsets, set_labels):
        calls.append(
            (
                "venn",
                sorted((k, type(v).__name__, float(v)) for k, v in subsets.items()),
                set_labels,
            )
        )

    def title(t, *a, **kw):
        calls.append(("title", t))
        return _REAL_TITLE(t, *a, **kw)

    monkeypatch.setattr(helperfuncs, "plot_pixels", plot_pixels)
    monkeypatch.setattr(matplotlib.axes.Axes, "hist", hist)
    monkeypatch.setattr(module, "venn3", venn3)
    monkeypatch.setattr(plt, "title", title)
    monkeypatch.setattr(plt.Figure, "savefig", lambda *a, **k: None)
    if hasattr(module, "save_figure"):
        monkeypatch.setattr(module, "save_figure", lambda *a, **k: None)
    kw = {} if threads is None else {"threads": threads}
    module.start_combining_masks(
        fig_path,
        str(tmp),
        IMAGEDIM,
        DIM_X,
        DIM_Y,
        STAINING,
        celltype_refined=False,
        **kw,
    )
    plt.close("all")
    return calls


@pytest.fixture
def inputs(tmp_path):
    _write_inputs(tmp_path, np.random.default_rng(0))
    return tmp_path


def _compare(monkeypatch, tmp, threads):
    os.makedirs(tmp / "fig", exist_ok=True)
    old = _record(legacy, monkeypatch, tmp, str(tmp / "fig"))
    new = _record(combine_masks, monkeypatch, tmp, str(tmp / "fig"), threads=threads)
    return old, new


@pytest.mark.parametrize("threads", [1, 4])
def test_every_number_handed_to_the_figures_matches(monkeypatch, inputs, threads):
    old, new = _compare(monkeypatch, inputs, threads)
    assert len(old) == len(new) > 0
    for i, (a, b) in enumerate(zip(old, new)):
        assert a == b, f"call {i}: {a[:2]}"


def test_nan_beliefs_match_too(monkeypatch, tmp_path):
    """Null beliefs (NaN in the frames hqcr/hqpr/hqtr write) reach the figures as NaN, as in origin/dev."""
    _write_inputs(tmp_path, np.random.default_rng(6), nan_beliefs=True)
    old, new = _compare(monkeypatch, tmp_path, 4)
    assert len(old) == len(new) > 0
    for i, (a, b) in enumerate(zip(old, new)):
        assert a == b, f"call {i}: {a[:2]}"
    hists = [np.frombuffer(c[2], dtype=c[1]) for c in new if c[0] == "hist"]
    assert hists and all(np.isnan(h).any() for h in hists), "every belief histogram must see NaN"


def test_non_binary_masks_match_too(monkeypatch, tmp_path):
    """origin/dev's Venn expressions are arithmetic, not set logic; values outside {0, 1} must agree too."""
    _write_inputs(tmp_path, np.random.default_rng(1), binary=False)
    with pytest.raises(AssertionError, match="Venn diagram error"):
        _record(legacy, monkeypatch, tmp_path, str(tmp_path))
    with pytest.raises(AssertionError, match="Venn diagram error"):
        _record(combine_masks, monkeypatch, tmp_path, str(tmp_path), threads=4)


@pytest.mark.parametrize("chunk", [1 << 20, 1000, 7])
def test_venn_counts_match_on_arbitrary_integers(monkeypatch, chunk):
    monkeypatch.setattr(combine_masks, "_CHUNK", chunk)
    rng = np.random.default_rng(2)
    n = 300_001
    c, p, t = (rng.integers(-2, 4, n).astype(d) for d in (np.uint8, np.int64, np.int64))
    ref = pd.DataFrame({"c": c, "p": p, "t": t})
    expected = [
        np.sum(((ref["c"] - ref["p"] - ref["t"]) == 1).astype(np.uint8)),
        np.sum(((ref["p"] - ref["c"] - ref["t"]) == 1).astype(np.uint8)),
        np.sum(((ref["t"] - ref["p"] - ref["c"]) == 1).astype(np.uint8)),
        np.sum(((ref["c"] + ref["p"] - ref["t"]) == 2).astype(np.uint8)),
        np.sum(((ref["c"] - ref["p"] + ref["t"]) == 2).astype(np.uint8)),
        np.sum(((-ref["c"] + ref["p"] + ref["t"]) == 2).astype(np.uint8)),
        np.sum(((ref["c"] + ref["p"] + ref["t"]) == 3).astype(np.uint8)),
        np.sum(((ref["c"] + ref["p"] + ref["t"]) == 0).astype(np.uint8)),
    ]
    got = combine_masks._venn_counts(c, p, t, threads=4)
    assert [int(x) for x in got] == [int(x) for x in expected]


# ----------------------------------------------------------------- mutants must fail


@pytest.mark.parametrize(
    "mutant",
    [
        lambda c, p, t, threads: combine_masks_venn(
            p, c, t, threads
        ),  # modalities swapped
        lambda c, p, t, threads: combine_masks_venn(
            c[: len(c) // 2], p[: len(c) // 2], t[: len(c) // 2], threads
        ),  # half the pixels lost (one pixel is below the figures' 3-decimal rounding)
    ],
)
def test_venn_mutants_are_caught(monkeypatch, inputs, mutant):
    monkeypatch.setattr(combine_masks, "_venn_counts", mutant)
    try:
        old, new = _compare(monkeypatch, inputs, 4)
    except AssertionError as e:  # the step's own Venn sanity check trips: caught too
        assert "Venn diagram error" in str(e)
        return
    assert old != new


def test_beliefs_mutant_is_caught(monkeypatch, inputs):
    real = combine_masks._mean_of_three

    def mutant(a, b, c, threads):
        return real(a, c, b, threads)  # (a + c) + b rounds differently from (a + b) + c

    monkeypatch.setattr(combine_masks, "_mean_of_three", mutant)
    old, new = _compare(monkeypatch, inputs, 4)
    assert old != new


combine_masks_venn = combine_masks._venn_counts


def test_lost_chunk_mutant_is_caught(monkeypatch):
    rng = np.random.default_rng(4)
    c, p, t = (rng.integers(0, 2, 10_000).astype(d) for d in (np.uint8, np.int64, np.int64))
    monkeypatch.setattr(combine_masks, "_CHUNK", 1000)
    good = combine_masks._venn_counts(c, p, t, threads=4)
    real = combine_masks._chunks
    monkeypatch.setattr(combine_masks, "_chunks", lambda n: real(n)[:-1])
    assert list(combine_masks._venn_counts(c, p, t, threads=4)) != list(good)


@pytest.mark.parametrize("chunk", [1 << 20, 1000, 7])
def test_mean_of_three_matches_pandas(monkeypatch, chunk):
    monkeypatch.setattr(combine_masks, "_CHUNK", chunk)
    rng = np.random.default_rng(5)
    a = rng.random(20_001).astype(np.float32)
    b, c = rng.random(20_001), rng.random(20_001)
    ref = pd.Series(a) + pd.Series(b) + pd.Series(c)
    ref /= 3
    got = combine_masks._mean_of_three(a, b, c, threads=4)
    assert got.dtype == ref.dtype and got.tobytes() == ref.to_numpy().tobytes()
