"""Differential tests: the per-pixel parquet writer (core.parquet.write_parts) vs origin/dev's
helperfuncs.ddf_to_parquet (tests/legacy/parquet_writer.py), byte for byte.
"""

import os

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from conftest import parquet_rows

from legacy.parquet_writer import ddf_to_parquet  # origin/dev's writer, the reference for core.parquet
from spoqc.core import parquet


def assert_same_dir(a, b):
    assert sorted(os.listdir(a)) == sorted(os.listdir(b))
    for name in os.listdir(b):
        assert open(f"{a}/{name}", "rb").read() == open(f"{b}/{name}", "rb").read(), (
            name
        )


class TestPrior:
    def test_write_parts_replaces_a_previous_directory(self, tmp_path):
        path = tmp_path / "hqtr_output_prior"
        path.mkdir()
        (path / "part.99.parquet").write_bytes(b"stale")
        values = np.arange(10.0)
        parquet.write_parts(str(path), 10, lambda a, b: {"v": values[a:b]}, range(0, 10, 4), 2)
        assert sorted(os.listdir(path)) == [
            "part.0.parquet",
            "part.1.parquet",
            "part.2.parquet",
        ]
        assert dd.read_parquet(str(path), calculate_divisions=True).divisions == (
            0,
            4,
            8,
            9,
        )


class TestWriteParts:
    @pytest.mark.parametrize("n", [9_999, 10_000, 10_001, 20_000, 29_999, 30_001])
    def test_nan_columns_match_dask_bytes(self, tmp_path, n):
        """dask writes through pa.Table.from_pandas, which stores NaN as null."""
        rng = np.random.default_rng(n)
        columns = {name: rng.normal(0, 1, n) for name in ("a_density", "d_a_density", "norm_p_a_density")}
        for values in columns.values():
            values[rng.random(n) < 0.05] = np.nan
        names = list(columns)
        ddf = dd.from_dask_array(da.from_array(columns[names[0]], chunks=10_000), columns=[names[0]])
        ddf = ddf.assign(**{k: dd.from_dask_array(da.from_array(columns[k], chunks=10_000)) for k in names[1:]})
        ddf_to_parquet(ddf, "hqtr", str(tmp_path), [], "old")
        parquet.write_parts(
            str(tmp_path / "hqtr_output_new"), n, parquet.columns_of(columns), range(0, n, 10_000), 3
        )
        assert_same_dir(tmp_path / "hqtr_output_new", tmp_path / "hqtr_output_old")


    @pytest.mark.parametrize("n", [1_000, 10_001, 30_653, 7, 5])
    @pytest.mark.parametrize("part_rows", [3, 4_000, 1 << 22])
    def test_mask_smoothed_raw_rows_match_origin_dev(self, tmp_path, n, part_rows):
        """The refinement's mask_smoothed_raw: origin/dev wrote dd.from_pandas(npartitions=
        ceil(n / 10,000)) of in-memory columns; part_rows parts hold the same rows, bytes and schema."""
        rng = np.random.default_rng(n)
        columns = {
            "hqtr_beliefs": rng.random(n),
            "hqtr_beliefs_smoothed": rng.random(n).astype(np.float32),
            "hqtr_mask_smoothed": rng.integers(0, 2, n).astype(np.int8),
        }
        ddf = dd.from_pandas(pd.DataFrame(columns), npartitions=-(-n // 10_000))
        ddf_to_parquet(ddf, "hqtr", str(tmp_path), [], "old")
        parquet.write_parts(str(tmp_path / "hqtr_output_new"), n, parquet.columns_of(columns), range(0, n, part_rows), 2)
        assert len(os.listdir(tmp_path / "hqtr_output_new")) == -(-n // part_rows)
        old, new = parquet_rows(str(tmp_path / "hqtr_output_old")), parquet_rows(str(tmp_path / "hqtr_output_new"))
        assert old.schema.equals(new.schema, check_metadata=True)
        assert old.equals(new)

    def test_starts_must_ascend_from_zero(self, tmp_path):
        with pytest.raises(ValueError, match="ascend"):
            parquet.write_parts(str(tmp_path / "p"), 10, lambda a, b: {"v": np.zeros(b - a)}, [0, 5, 5], 1)
