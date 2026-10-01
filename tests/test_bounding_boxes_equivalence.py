"""Differential test: spoqc.core.raster.read_pixel_columns vs dask.dataframe.read_parquet, the
reader it replaces, on dask-written part directories, single files and nulls.
"""

import os

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from legacy.parquet_writer import ddf_to_parquet  # origin/dev's writer, the reference for core.parquet
from spoqc.core import raster


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.float32])
def test_read_pixel_columns_matches_dask_on_a_many_part_directory(tmp_path, dtype):
    values = np.random.default_rng(3).integers(0, 100, 5003).astype(dtype)
    frame = pd.DataFrame({"other": np.arange(len(values)), "v": values})
    ddf = dd.from_pandas(
        frame, npartitions=13
    )  # part.10 sorts before part.2 lexicographically
    ddf_to_parquet(ddf, "p", str(tmp_path), [], "mask_smoothed_raw")
    path = f"{tmp_path}/p_output_mask_smoothed_raw"
    assert len(os.listdir(path)) >= 13
    expected = (
        dd.read_parquet(path, columns=["v"], engine="pyarrow")["v"].compute().to_numpy()
    )
    actual = raster.read_pixel_columns(path, ["v"], len(values), 3)["v"]
    assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()


def test_read_pixel_columns_reads_every_row_group_of_a_file(tmp_path):
    values = np.arange(10_007, dtype=np.int64) * 3
    pq.write_table(
        pa.Table.from_arrays([pa.array(values)], names=["d"]),
        f"{tmp_path}/f.parquet",
        row_group_size=1000,
    )
    assert pq.ParquetFile(f"{tmp_path}/f.parquet").metadata.num_row_groups == 11
    assert (
        raster.read_pixel_columns(f"{tmp_path}/f.parquet", ["d"], len(values), 3)["d"].tobytes()
        == values.tobytes()
    )


def test_read_pixel_columns_raises_on_an_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="no .parquet part files"):
        raster.read_pixel_columns(str(tmp_path), ["v"], 2, 3)


@pytest.mark.parametrize("type_", [pa.int8(), pa.uint8(), pa.int32(), pa.int64(), pa.float32(), pa.float64()])
@pytest.mark.parametrize("null_part", [0, 1])
def test_read_pixel_columns_matches_dask_on_nulls(tmp_path, type_, null_part):
    """Nulls in one part only: NaN for floats; integers become float64 everywhere, as dask returns them."""
    parts = [[1, 0, 1, 1], [0, 1, 1]]
    parts[null_part][1] = None
    for i, values in enumerate(parts):
        pq.write_table(
            pa.table({"v": pa.array(values, type=type_)}), f"{tmp_path}/part.{i}.parquet", row_group_size=2
        )
    expected = dd.read_parquet(str(tmp_path), columns=["v"], engine="pyarrow")["v"].compute().to_numpy()
    assert np.isnan(expected).sum() == 1
    actual = raster.read_pixel_columns(str(tmp_path), ["v"], len(expected), 3)["v"]
    assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()


def test_read_pixel_columns_raises_on_nulls_in_a_bool_column(tmp_path):
    """dask would return an object array of True/False/None; no pixel column is boolean."""
    pq.write_table(pa.table({"v": pa.array([True, None], type=pa.bool_())}), f"{tmp_path}/part.0.parquet")
    with pytest.raises(ValueError, match="1 nulls in the bool column"):
        raster.read_pixel_columns(str(tmp_path), ["v"], 2, 3)


def test_read_pixel_columns_raises_on_parts_with_different_types(tmp_path):
    pq.write_table(pa.table({"v": pa.array([1, 2], type=pa.int8())}), f"{tmp_path}/part.0.parquet")
    pq.write_table(pa.table({"v": pa.array([3, 4], type=pa.int64())}), f"{tmp_path}/part.1.parquet")
    with pytest.raises(ValueError, match="disagree"):
        raster.read_pixel_columns(str(tmp_path), ["v"], 4, 3)
