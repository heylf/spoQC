"""Differential tests: each converted call site against its verbatim origin/dev module (tests/legacy/)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from conftest import IMAGE_SHAPE, assert_same_array, cli_sdata, load_legacy

from spoqc import helperfuncs
from spoqc.core import transcripts
from spoqc.metrics.segmentation import void
from spoqc.metrics.transcript_density import (
    ac_image,
    local_moran_I,
    qv_image,
    transcript_density_image,
)
from spoqc.subworkflows import qc_transcript

DENSITY = "spoqc.metrics.transcript_density"
IMAGEDIM = helperfuncs.ImageDimStruct(0, 0, IMAGE_SHAPE[1], IMAGE_SHAPE[0])
IMAGE_ARGS = (IMAGEDIM, "morphology_focus", "scale0")


def test_transcript_density_image(sdata):
    legacy = load_legacy("transcript_density_image", DENSITY)
    expected = legacy.generate_transcript_density_image(sdata, None, *IMAGE_ARGS)
    got = transcript_density_image.generate_transcript_density_image(
        sdata, None, *IMAGE_ARGS
    )
    assert expected.sum() > 0
    assert_same_array(got, expected, "density image")


def test_transcript_quality_density_image(sdata):
    legacy = load_legacy("qv_image", DENSITY)
    expected = legacy.generate_transcript_quality_density_image(
        sdata, None, *IMAGE_ARGS
    )
    got = qv_image.generate_transcript_quality_density_image(sdata, None, *IMAGE_ARGS)
    assert expected.sum() > 0
    assert_same_array(got, expected, "qv image")


def _legacy_local_moran_I(monkeypatch):
    """The origin/dev module, with the helperfuncs.points_within_radius it called (since
    replaced by core.spatial.neighbour_lists) restored verbatim for it."""
    from test_core_spatial_neighbours import original_points_within_radius

    monkeypatch.setattr(helperfuncs, "points_within_radius", original_points_within_radius, raising=False)
    return load_legacy("local_moran_I", DENSITY)


def test_local_moran_I_values(sdata, monkeypatch):
    legacy = _legacy_local_moran_I(monkeypatch)
    expected = legacy.calculate_local_moran_I_values(sdata, 2)
    got = local_moran_I.calculate_local_moran_I_values(sdata, 2)
    assert len(np.unique(expected)) > 10, "degenerate synthetic data"
    assert_same_array(got, expected, "local Moran's I")


def _global_ambient():
    # a repeated gene (later row wins), a gene absent from the transcripts, a NaN Moran's I
    return pd.DataFrame(
        {
            "genes": [
                "SEC11C",
                "ACTA2",
                "EPCAM",
                "KRT7",
                "SEC11C",
                "CD4",
                "NegControlCodeword_0502",
            ],
            "morans_I": [0.25, -0.5, 0.9, np.nan, 0.75, 0.125, 0.3],
        }
    )


def test_transcript_ambient_density_image(sdata, monkeypatch):
    legacy = load_legacy("ac_image", DENSITY)
    monkeypatch.setattr(legacy, "local_moran_I", _legacy_local_moran_I(monkeypatch))
    expected = legacy.generate_transcript_ambient_density_image(
        sdata, None, 2, IMAGEDIM, _global_ambient(), *IMAGE_ARGS[1:]
    )
    got = ac_image.generate_transcript_ambient_density_image(
        sdata, None, 2, IMAGEDIM, _global_ambient(), *IMAGE_ARGS[1:]
    )
    assert np.count_nonzero(expected) > 0
    assert_same_array(got, expected, "ac image")


class _Stop(Exception):
    pass


def _void_transcript_counts(
    module, sdata, figure_path, tmp, contaminants, n_transcript_calls
):
    """Run calc_void up to the nuclei count and return what each transcript count received."""
    calls = []
    real = module.count_stuff_in_triangles_via_delaunay

    def record(delaunay, stuff):
        if len(calls) == n_transcript_calls:
            raise _Stop
        calls.append(np.array(stuff, copy=True))
        return real(delaunay, stuff)

    module.count_stuff_in_triangles_via_delaunay = record
    try:
        with pytest.raises(_Stop):
            module.calc_void(sdata, str(figure_path), str(tmp), 1, contaminants)
    finally:
        module.count_stuff_in_triangles_via_delaunay = real
    return calls


@pytest.mark.parametrize(
    "crop", [None, (10, 10, 60, 50)], ids=["full", "dev_test_crop"]
)
def test_void_transcript_counts(synthetic_zarr, crop, tmp_path, monkeypatch):
    sdata = cli_sdata(synthetic_zarr, crop)
    # figures are not compared; kaleido needs a browser the test host may not allow
    monkeypatch.setattr(go.Figure, "write_image", lambda *args, **kwargs: None)
    frame = sdata.points["transcripts"].compute()
    doublet = pd.DataFrame(
        {
            "doublet": (frame["x"] < 12).to_numpy(),
            "wdoublet": (frame["x"] < 12).to_numpy().astype(int),
        },
        index=frame.index,
    )
    helperfuncs.df_to_parquet(doublet, "doublet", str(tmp_path), [], "transcripts")
    contaminants = ["ACTA2", "EPCAM"]  # EPCAM has no transcripts
    legacy = load_legacy("void", "spoqc.metrics.segmentation")
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    expected = _void_transcript_counts(
        legacy, sdata, tmp_path / "old", tmp_path, contaminants, 3
    )
    got = _void_transcript_counts(
        void, sdata, tmp_path / "new", tmp_path, contaminants, 3
    )
    assert len(got) == len(expected) == 3 and min(len(c) for c in expected) > 0
    for i, (g, e) in enumerate(zip(got, expected)):
        assert_same_array(g, e, f"count call {i}")


def test_negativeprobeqc_plots_same_points(sdata, monkeypatch):
    captured = []
    monkeypatch.setattr(
        helperfuncs, "plot_scatter_density_df", lambda df, *args: captured.append(df)
    )
    load_legacy("qc_transcript", "spoqc.subworkflows").negativeprobeqc(
        sdata, "unused", "transcripts"
    )
    qc_transcript.negativeprobeqc(sdata, "unused")
    expected, got = captured
    assert len(expected) > 0
    for column in ["x", "y", "neg_probes"]:
        assert_same_array(got[column], expected[column], column)


def test_low_qc_transcript_count(sdata, monkeypatch):
    monkeypatch.setattr(
        helperfuncs, "plot_scatter_density", lambda *args, **kwargs: None
    )
    # the function merges its string cell ids on an obs level named "index"
    sdata["table"].obs.index = pd.Index(
        sdata["table"].obs.index.astype(str), name="index"
    )
    qc_transcript.get_low_qc_transcript_count(
        sdata["transcripts"].compute(), sdata, 20, "unused"
    )
    expected = sdata["table"].obs["num_low_qc_transcript"].to_numpy()
    frame = transcripts.load_transcripts(sdata, ["qv", "cell_id"]).to_pandas()
    qc_transcript.get_low_qc_transcript_count(frame, sdata, 20, "unused")
    assert expected.sum() > 0
    assert_same_array(
        sdata["table"].obs["num_low_qc_transcript"], expected, "num_low_qc_transcript"
    )


def test_void_rejects_a_stale_doublet_parquet(sdata, tmp_path, monkeypatch):
    monkeypatch.setattr(go.Figure, "write_image", lambda *args, **kwargs: None)
    frame = sdata.points["transcripts"].compute()
    stale = pd.DataFrame(
        {"doublet": np.zeros(len(frame), bool), "wdoublet": np.zeros(len(frame), int)},
        index=frame.index + 1,  # same length, monotonic, other rows
    )
    helperfuncs.df_to_parquet(stale, "doublet", str(tmp_path), [], "transcripts")
    with pytest.raises(AssertionError):
        void.calc_void(sdata, str(tmp_path), str(tmp_path), 1, [])


def _write_gtf(path):
    import gzip

    lines = [
        f'chr1\tsrc\tgene\t1\t2\t.\t+\t.\tgene_name "{name}"; gene_type "{kind}";'
        for name, kind in [
            ("SEC11C", "protein_coding"),
            ("ACTA2", "protein_coding"),
            ("KRT7", "lncRNA"),
        ]
    ]
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")


def _capture_plots(monkeypatch):
    captured = []
    monkeypatch.setattr(go.Figure, "write_image", lambda *args, **kwargs: None)
    monkeypatch.setattr(go.Figure, "write_html", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        helperfuncs,
        "plot_scatter_density_by_category_df",
        lambda df, *a, **k: captured.append(df),
    )
    monkeypatch.setattr(
        helperfuncs, "plot_scatter_density_df", lambda df, *a, **k: captured.append(df)
    )
    return captured


def test_transcriptqc_uses_the_same_transcripts(sdata, tmp_path, monkeypatch):
    gtf = tmp_path / "ref.gtf.gz"
    _write_gtf(gtf)
    captured = _capture_plots(monkeypatch)
    load_legacy("qc_transcript", "spoqc.subworkflows").transcriptqc(
        sdata, str(tmp_path), str(gtf), "transcripts"
    )
    n_old = len(captured)
    qc_transcript.transcriptqc(sdata, str(tmp_path), str(gtf))
    old, new = captured[:n_old], captured[n_old:]
    assert len(old) == len(new) == 3
    for e, g in zip(old, new):
        for column in ["x", "y", "qv", "cell_id", "location", "feature_type"]:
            assert_same_array(g[column], e[column], column)


def test_transcriptz_uses_the_same_transcripts(sdata, tmp_path, monkeypatch):
    captured = _capture_plots(monkeypatch)
    load_legacy("qc_transcript", "spoqc.subworkflows").transcriptz(
        sdata, str(tmp_path), "transcripts"
    )
    qc_transcript.transcriptz(sdata, str(tmp_path))
    old, new = captured
    for column in ["x", "y", "z", "sample"]:
        assert_same_array(new[column], old[column], column)
