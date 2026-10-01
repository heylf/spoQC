"""Differential tests: the per-pixel parquet writer, the histogram and Moran's I vs the verbatim origin/dev code
they replace.

- the parquet writer (core.parquet): helperfuncs.ddf_to_parquet, byte for byte;
- the threaded histogram (helperfuncs.histogram): np.histogram;
- Moran's I: tests/legacy/global_moran_I.py and tests/legacy/local_moran_I.py.
Every comparison is exact: values, dtype and order.
"""

import os

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from conftest import assert_same_array, load_legacy, parquet_rows
from libpysal.weights import KNN

from legacy.parquet_writer import ddf_to_parquet  # origin/dev's writer, the reference for core.parquet
from spoqc import helperfuncs
from spoqc.core import parquet
from spoqc.metrics.transcript_density import (
    global_moran_I,
    local_moran_I,
)


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


class TestHistogram:
    def test_counts_are_numpys_over_non_nan_values(self):
        values = np.random.default_rng(2).normal(0, 1, 100_001)
        values[::97] = np.nan
        counts, edges = helperfuncs.histogram(values, 100)
        finite = values[~np.isnan(values)]
        expected_counts, expected_edges = np.histogram(finite, bins=100)
        assert_same_array(counts, expected_counts, "counts")
        assert_same_array(edges, expected_edges, "edges")


class TestMoransI:
    def test_global_matches_origin(self):
        rng = np.random.default_rng(3)
        X = rng.poisson(1.0, (300, 12)).astype(np.float32)
        X[:, 4] = 2.0  # zero variance: NaN
        w = KNN.from_array(rng.uniform(0, 100, (300, 2)), k=6)
        w.transform = "r"
        legacy = load_legacy("global_moran_I", "spoqc.metrics.transcript_density")
        assert_same_array(
            global_moran_I.moran_I_all_genes(X, w.sparse),
            legacy.moran_I_all_genes(X, w.sparse),
            "global",
        )

    def test_local_matches_origin(self):
        rng = np.random.default_rng(4)
        coords = rng.uniform(0, 100, (90, 2))
        X = rng.poisson(1.0, (90, 12)).astype(np.float32)
        X[:, 7] = 0.0  # zero variance: -1
        legacy = load_legacy("local_moran_I", "spoqc.metrics.transcript_density")
        expected = legacy.moran_I_all_genes(X, KNN.from_array(coords, k=30))
        got = global_moran_I.moran_I_all_genes(
            X, local_moran_I.KNNWeights(coords, 30), fill=-1.0, dtype=np.float32
        )
        assert_same_array(got, expected, "local")


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
