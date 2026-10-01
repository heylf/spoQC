"""final_report without an annotation file.

cli.py runs the analysis step (analysis/overview, analysis/cluster) only when an annotation file
is given, but origin/dev's report read those figures unconditionally and crashed without one.
The report now includes those sections only when annotated=True. The tests check three things:
- With an annotation, report.html and report_p*.html are byte-identical to the pre-change
  function (tests/legacy/final_report.py, origin/dev db00d98's report code).
- Without an annotation, the report completes.
- With an annotation, a missing analysis figure still raises.
"""

import os
import shutil

import pytest

from spoqc.subworkflows import final_report

from conftest import load_legacy

legacy = load_legacy("final_report", "spoqc.subworkflows")

STAININGS = ["DAPI", "boundary"]

# Figures whose presence origin/dev checks by listing a folder or with os.path.exists, so the
# fill loop below never discovers them. Present them explicitly to exercise those branches.
LISTED = [
    "analysis/overview/scatterplot/scatterplot_leiden_cluster_0.png",
    "analysis/overview/scatterplot/scatterplot_leiden_cluster_1.png",
    "analysis/overview/scatterplot/scatterplot_annotation_Tcell.png",
    "analysis/overview/umap/umap_plot_hqpr_0_filtered_out.png",
    "analysis/overview/scatterplot/scatterplot_hqpr_0_filtered_out.png",
    "analysis/overview/boxplot/boxplot_hqcr_beliefs.png",
    "analysis/overview/boxplot/boxplot_hqpr_0_beliefs.png",
    "analysis/overview/boxplot/boxplot_hqtr_beliefs.png",
    "cellcycleqc/barplot_sample_cellcycle_fractions.png",
]
SUBCLUSTER = [
    "analysis/cluster/funkyheatmap/funkyheatmap_1.png",
    "analysis/cluster/scatterplot/scatterplot_leiden_cluster_0.png",
]


def _write(root, rel):
    """A distinct file per path, so a figure shown in the wrong place changes the HTML."""
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(
            b"\x89PNG-" + rel.encode()
            if rel.endswith(".png")
            else f"<div>{rel}</div>".encode()
        )


def _report_dir(tmp_path, subcluster):
    """A figure folder holding every file origin/dev's report reads, found by running it."""
    root = tmp_path / "report"
    os.makedirs(root / "combine_masks" / "0")
    os.makedirs(root / "combine_masks" / "1")
    for rel in LISTED + (SUBCLUSTER if subcluster else []):
        _write(root, rel)
    for _ in range(500):
        try:
            legacy.create_final_report(str(root), STAININGS, True)
            break
        except FileNotFoundError as e:
            _write(root, os.path.relpath(e.filename, root))
    else:
        raise AssertionError("fixture never completed")
    for f in os.listdir(root):
        if f.startswith("report"):
            os.remove(root / f)
    return root


def _outputs(root):
    return {
        f: (root / f).read_bytes()
        for f in sorted(os.listdir(root))
        if f.startswith("report")
    }


@pytest.mark.parametrize("subcluster", [False, True])
def test_with_annotation_the_report_is_unchanged(tmp_path, subcluster):
    root = _report_dir(tmp_path, subcluster)
    legacy.create_final_report(str(root), STAININGS, True)
    old = _outputs(root)
    for f in old:
        os.remove(root / f)
    final_report.create_final_report(str(root), STAININGS, True, True)
    new = _outputs(root)
    assert list(new) == list(old) and len(old) > 1
    for f in old:
        assert new[f] == old[f], f


def test_without_annotation_the_report_completes(tmp_path):
    root = _report_dir(tmp_path, subcluster=True)
    stale = tmp_path / "stale_analysis"
    shutil.move(root / "analysis", stale)  # no annotation: the analysis step never ran
    with pytest.raises(FileNotFoundError, match="analysis/overview"):
        legacy.create_final_report(str(root), STAININGS, True)  # origin/dev's crash
    final_report.create_final_report(str(root), STAININGS, True, False)
    html = (root / "report.html").read_text()
    assert "Combined beliefs" in html and "High quality regions (HQRs)" in html
    for gone in [
        "Summary of spoQC",
        "Traffic Light System HQCR",
        "Individual HQR filters",
        "Subcluster analysis",
        "Spatial plots annotation clusters",
    ]:
        assert gone not in html, gone
    assert sorted(f for f in os.listdir(root) if f.startswith("report_p")) == [
        "report_p0.html",
        "report_p1.html",
    ]


def test_without_annotation_stale_analysis_figures_are_not_shown(tmp_path):
    root = _report_dir(tmp_path, subcluster=True)
    final_report.create_final_report(str(root), STAININGS, False, False)
    assert "analysis/" not in (root / "report.html").read_text()
    assert b"analysis/" not in b"".join(_outputs(root).values())


@pytest.mark.parametrize(
    "missing",
    [
        "analysis/overview/funkyheatmap/funkyheatmap_1.png",
        "analysis/overview/umap/umap_plot_leiden.html",
        "analysis/overview/fractions/fractions_traffic_light_leiden.png",
        "analysis/overview/umap/umap_plot_hqtr_filtered_out.png",
    ],
)
def test_with_annotation_a_missing_figure_still_raises(tmp_path, missing):
    root = _report_dir(tmp_path, subcluster=False)
    os.remove(root / missing)
    with pytest.raises(FileNotFoundError):
        final_report.create_final_report(str(root), STAININGS, False, True)
