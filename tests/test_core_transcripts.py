"""core.transcripts equals the dask element cli.py sets up, and reads each column once."""

from __future__ import annotations

import gc
import weakref
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import spatialdata as sd
from spatialdata.transformations import (
    Affine,
    Identity,
    MapAxis,
    Scale,
    Sequence,
    Translation,
    set_transformation,
)
from conftest import assert_same_array, cli_sdata

from spoqc.core import transcripts

COLUMNS = [
    "x",
    "y",
    "z",
    "feature_name",
    "cell_id",
    "transcript_id",
    "overlaps_nucleus",
    "qv",
]


def test_load_equals_cli_dask_compute(sdata):
    expected = sdata.points["transcripts"].compute()
    got = transcripts.load_transcripts(sdata, COLUMNS).to_pandas()
    assert expected.index.equals(got.index), "row order is not the deduplicated index"
    for column in COLUMNS:
        if column == "feature_name":
            assert list(got[column].cat.categories) == list(
                expected[column].cat.categories
            )
            assert (
                "NaN" in got[column].cat.categories
                and "UNUSED" not in got[column].cat.categories
            )
            assert_same_array(got[column].cat.codes, expected[column].cat.codes, column)
        else:
            assert_same_array(got[column], expected[column], column)
    assert (got["cell_id"] == -1).any() and (got["cell_id"] >= 0).any()


def test_frame_is_one_chunk_and_enum(sdata):
    frame = transcripts.load_transcripts(sdata, ["x", "feature_name"])
    assert frame.n_chunks() == 1
    assert isinstance(frame.schema["feature_name"], pl.Enum)


def test_each_column_is_computed_once(sdata, monkeypatch):
    reads = []
    real = transcripts._read
    monkeypatch.setattr(
        transcripts,
        "_read",
        lambda sdata, columns: reads.append(columns) or real(sdata, columns),
    )
    transcripts.load_transcripts(sdata, ["x", "y"])
    transcripts.load_transcripts(sdata, ["y", "qv"])
    transcripts.load_transcripts(sdata, ["qv", "x", "y"])
    assert reads == [["x", "y"], ["qv"]]
    transcripts.release(sdata)
    transcripts.load_transcripts(sdata, ["x"])
    assert reads[-1] == ["x"]


def test_global_coordinates_equal_get_centroids(sdata):
    expected = sd.get_centroids(
        sdata["transcripts"], coordinate_system="global"
    ).compute()
    frame = transcripts.global_coordinates(sdata)
    assert transcripts.global_coordinates(sdata) is frame, "computed once per run"
    got = frame.to_pandas()
    assert list(got.columns) == list(expected.columns)
    for column in expected.columns:
        assert_same_array(got[column], expected[column], column)
    assert_same_array(
        got.astype(int).to_numpy(), expected.astype(int).to_numpy(), "astype(int)"
    )


def test_cropped_sdata_equals_its_dask_compute(synthetic_zarr):
    cropped = cli_sdata(synthetic_zarr, crop=(10, 10, 60, 50))
    assert cropped.path is None
    expected = cropped.points["transcripts"].compute()
    got = transcripts.load_transcripts(cropped, COLUMNS).to_pandas()
    assert 0 < len(got) < 6_000
    for column in COLUMNS:
        if column == "feature_name":
            assert list(got[column].cat.categories) == list(
                expected[column].cat.categories
            )
            assert_same_array(got[column].cat.codes, expected[column].cat.codes, column)
        else:
            assert_same_array(got[column], expected[column], column)
    expected_xyz = sd.get_centroids(
        cropped["transcripts"], coordinate_system="global"
    ).compute()
    got_xyz = transcripts.global_coordinates(cropped).to_pandas()
    for column in expected_xyz.columns:
        assert_same_array(got_xyz[column], expected_xyz[column], column)


@pytest.mark.parametrize(
    "transformation",
    [
        Identity(),
        Translation([3.5, -2.25], axes=("x", "y")),
        MapAxis({"x": "y", "y": "x", "z": "z"}),
        Scale([0.3, 7.1, 2.0], axes=("x", "y", "z")),
    ],
    ids=lambda t: type(t).__name__,
)
def test_global_coordinates_exact_for_elementwise_transformations(
    sdata, transformation
):
    set_transformation(sdata.points["transcripts"], transformation, "global")
    expected = sd.get_centroids(
        sdata["transcripts"], coordinate_system="global"
    ).compute()
    got = transcripts.global_coordinates(sdata).to_pandas()
    for column in expected.columns:
        assert_same_array(got[column], expected[column], column)


@pytest.mark.parametrize(
    "transformation",
    [
        Affine(
            np.array([[0.9, 0.1, 3.0], [-0.2, 1.1, 1.0], [0.0, 0.0, 1.0]]),
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        ),
        Sequence(
            [
                Scale([2.0, 2.0], axes=("x", "y")),
                Translation([1.0, 1.0], axes=("x", "y")),
            ]
        ),
    ],
    ids=lambda t: type(t).__name__,
)
def test_global_coordinates_refuses_matrix_transformations(sdata, transformation):
    set_transformation(sdata.points["transcripts"], transformation, "global")
    with pytest.raises(NotImplementedError, match="bit-identical"):
        transcripts.global_coordinates(sdata)


def test_caches_die_with_their_sdata(synthetic_zarr):
    sdata = cli_sdata(synthetic_zarr)
    transcripts.load_transcripts(sdata, ["x"])
    transcripts.global_coordinates(sdata)
    ref = weakref.ref(sdata)
    del sdata
    gc.collect()
    assert ref() is None
    assert (
        len(transcripts._frames)
        == len(transcripts._indexes)
        == len(transcripts._global_coordinates)
        == 0
    )


def test_transcript_index_is_the_element_index(synthetic_zarr):
    for crop in (None, (10, 10, 60, 50)):
        sdata = cli_sdata(synthetic_zarr, crop)
        expected = sdata.points["transcripts"].index.compute().to_numpy()
        transcripts.load_transcripts(sdata, ["x"])
        assert_same_array(
            transcripts.transcript_index(sdata), expected, f"index, crop={crop}"
        )
        fresh = cli_sdata(synthetic_zarr, crop)
        assert_same_array(
            transcripts.transcript_index(fresh),
            expected,
            f"index without a load, crop={crop}",
        )


def test_lookup_by_code_maps_names_through_enum_codes():
    feature = pl.Series(["b", "a", "c", "a"], dtype=pl.Enum(["a", "b", "c"]))
    lut = transcripts.lookup_by_code(feature, {"a": 5, "c": 7, "zz": 9}, -1, np.int64)
    assert_same_array(lut, np.array([5, -1, 7], dtype=np.int64), "lut")
    assert_same_array(
        lut[feature.to_physical().to_numpy()],
        np.array([-1, 5, 7, 5], dtype=np.int64),
        "gather",
    )


def test_no_deprecated_polars_category_api():
    root = Path(transcripts.__file__).parents[1]
    offenders = [
        str(p) for p in root.rglob("*.py") if "get_categories" in p.read_text()
    ]
    assert offenders == []
