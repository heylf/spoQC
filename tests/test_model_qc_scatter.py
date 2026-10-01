"""Model QC's PC scatter: the vectorised hue colouring against seaborn's own scatterplot."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import seaborn as sns  # noqa: E402

from spoqc.subworkflows import qc_model  # noqa: E402


def frame(n, dtype, seed):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 5000, n), rng.uniform(0, 5000, n)
    return pd.DataFrame({"x": x, "y": y, "PC0": (np.sin(x / 300) + rng.normal(size=n)).astype(dtype)})


def draw(df, s, new):
    fig = plt.figure(figsize=(4, 4))
    if new:
        qc_model.scatter_grey_hue(df, "PC0", s)
    else:
        sns.scatterplot(data=df, x="x", y="y", hue="PC0", s=s, palette="grey")
    legend = plt.legend(bbox_to_anchor=(1.15, 1), loc="upper left", borderaxespad=0.0, markerscale=1)
    ax = plt.gca()
    fig.canvas.draw()
    points = ax.collections[0]
    out = dict(
        offsets=np.asarray(points.get_offsets()),
        facecolors=points.get_facecolors(),
        edgecolors=points.get_edgecolors(),
        linewidths=points.get_linewidths(),
        sizes=points.get_sizes(),
        legend=[t.get_text() for t in legend.get_texts()],
        legend_colors=[(h.get_markerfacecolor(), h.get_markeredgecolor(), h.get_markersize()) for h in legend.legend_handles],
        labels=(ax.get_xlabel(), ax.get_ylabel()),
        limits=(ax.get_xlim(), ax.get_ylim()),
        png=np.asarray(fig.canvas.buffer_rgba()).copy(),
    )
    plt.close(fig)
    return out


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("s", [1, 10])
def test_scatter_matches_seaborn(dtype, s):
    df = frame(3000, dtype, seed=s)
    ref, new = draw(df, s, new=False), draw(df, s, new=True)
    for key in ["offsets", "facecolors", "edgecolors", "linewidths", "sizes"]:
        assert ref[key].dtype == new[key].dtype and np.array_equal(ref[key], new[key]), key
    for key in ["legend", "legend_colors", "labels", "limits"]:
        assert ref[key] == new[key], key
    # the two extreme points are drawn twice, identically; only their antialiased edges may differ
    differing = np.any(ref["png"] != new["png"], axis=-1).mean()
    assert differing < 1e-3, differing


def test_wrong_palette_is_caught(monkeypatch):
    df = frame(500, np.float32, seed=2)
    ref = draw(df, 1, new=False)
    monkeypatch.setattr(qc_model.sns, "color_palette", lambda *a, **k: plt.get_cmap("viridis"))
    new = draw(df, 1, new=True)
    assert not np.array_equal(ref["facecolors"], new["facecolors"])
