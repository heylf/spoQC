"""Differential tests: the hqtr/ambient primitives vs the verbatim origin/dev code they replace.

- pixel grids (core.groupreduce): pandas groupby(["x", "y"]) + reindex onto the MultiIndex grid;
- disk_density: np.flipud(scipy.ndimage.convolve(...)) of the whole image;
- the qv/ac prior and its parquet (priors.hqtr.ac_or_qv + core.parquet): tests/legacy/ac_or_qv.py
  and helperfuncs.ddf_to_parquet, byte for byte;
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
from scipy.ndimage import convolve

from legacy.parquet_writer import ddf_to_parquet  # origin/dev's writer, the reference for core.parquet
from spoqc import helperfuncs
from spoqc.core import groupreduce, parquet
from spoqc.metrics.transcript_density import (
    global_moran_I,
    local_moran_I,
    transcript_density_image,
)
from spoqc.priors.hqtr import ac_or_qv

X0, X1, Y0, Y1 = 3, 40, 2, 31  # grid x in [X0, X1), y in [Y0, Y1)


def origin_grid(df, column, how):
    """qv_image.py / ac_image.py at origin/dev: groupby, reindex onto the pixel grid, fillna(0.0)."""
    gm = getattr(df.groupby(["x", "y"])[column], how)()
    grid = pd.MultiIndex.from_tuples(
        [(x, y) for y in range(Y0, Y1) for x in range(X0, X1)], names=["x", "y"]
    )
    return gm.reindex(grid).fillna(0.0).to_numpy().astype("float64")


def origin_counts(df):
    """transcript_density_image.py at origin/dev."""
    counts = df.value_counts(subset=["x", "y"]).rename("count")
    grid = pd.MultiIndex.from_tuples(
        [(x, y) for y in range(Y0, Y1) for x in range(X0, X1)], names=["x", "y"]
    )
    idxer = counts.index.get_indexer(grid)
    return np.where(idxer >= 0, counts.to_numpy()[idxer], 0)


@pytest.fixture
def points():
    """Truncated integer pixels with off-grid points on every side, crowded pixels, and hard values."""
    rng = np.random.default_rng(0)
    n = 30_000
    x = rng.normal(20, 8, n).astype(np.float32)
    y = rng.normal(15, 7, n).astype(np.float32)
    x[:50] = X1  # the grid's exclusive end
    y[50:100] = -0.7  # truncates to 0, below Y0
    x[100:5000] = 17.3  # a few crowded pixels
    y[100:5000] = rng.integers(10, 13, 4900)
    df = pd.DataFrame({"x": x, "y": y}).astype(int)
    qv = rng.uniform(0, 40, n).astype(np.float32)
    qv[100:5000:3] = np.float32(
        3.0e4
    )  # large and small values: a plain float32 sum would drift
    qv[100:5000:7] = np.float32(1.0e-3)
    qv[5000:5040] = np.nan  # NaN is skipped
    df.loc[5000:5039, ["x", "y"]] = [8, 8]  # ... and a pixel with only NaN
    qv[6000] = np.inf
    df["qv"] = qv
    df["morans_I"] = rng.normal(0, 1, n)
    df.loc[7000:7009, ["x", "y"]] = [9, 9]
    df.loc[
        7000:7009, "morans_I"
    ] = -2.0  # a pixel whose values are all negative stays negative
    df["local"] = np.where(rng.random(n) < 0.3, -1.0, rng.normal(0, 1, n)).astype(
        np.float32
    )
    return df


def new_groups(df):
    return groupreduce.pixel_groups(
        df["x"].to_numpy(), df["y"].to_numpy(), (X0, X1), (Y0, Y1)
    )


N_PIXELS = (X1 - X0) * (Y1 - Y0)


class TestPixelGrids:
    def test_counts_match_value_counts(self, points):
        pixels, _, offsets = new_groups(points)
        assert_same_array(
            groupreduce.to_grid(pixels, np.diff(offsets), N_PIXELS, 0),
            origin_counts(points),
            "counts",
        )

    def test_mean_matches_pandas_groupby_mean(self, points):
        pixels, rows, offsets = new_groups(points)
        mean = groupreduce.group_mean(points["qv"].to_numpy(), rows, offsets)
        got = groupreduce.to_grid(pixels, mean, N_PIXELS, 0.0).astype("float64")
        assert_same_array(got, origin_grid(points, "qv", "mean"), "qv mean")

    def test_mean_data_is_sensitive_to_the_summation(self, points):
        """Mutant check: without pandas' compensated sum the crowded pixels come out different."""
        pixels, rows, offsets = new_groups(points)
        values = points["qv"].to_numpy()
        plain = np.array(
            [
                np.float32(values[rows[a:b]][~np.isnan(values[rows[a:b]])].sum())
                / np.float32(max(np.count_nonzero(~np.isnan(values[rows[a:b]])), 1))
                for a, b in zip(offsets[:-1], offsets[1:])
            ],
            dtype=np.float32,
        )
        got = groupreduce.to_grid(pixels, plain, N_PIXELS, 0.0).astype("float64")
        assert got.tobytes() != origin_grid(points, "qv", "mean").tobytes()

    @pytest.mark.parametrize("column", ["morans_I", "local"])
    def test_max_matches_pandas_groupby_max(self, points, column):
        pixels, rows, offsets = new_groups(points)
        maximum = groupreduce.group_max(points[column].to_numpy(), rows, offsets)
        got = groupreduce.to_grid(pixels, maximum, N_PIXELS, 0.0).astype("float64")
        assert_same_array(got, origin_grid(points, column, "max"), column)

    def test_rows_within_a_pixel_keep_their_order(self, points):
        _, rows, offsets = new_groups(points)
        assert all(
            (np.diff(rows[a:b]) > 0).all() for a, b in zip(offsets[:-1], offsets[1:])
        )

    def test_no_points_on_the_grid(self):
        empty = pd.DataFrame({"x": [0, 100], "y": [0, 100], "qv": np.float32([1, 2])})
        pixels, rows, offsets = new_groups(empty)
        mean = groupreduce.group_mean(empty["qv"].to_numpy(), rows, offsets)
        assert_same_array(
            groupreduce.to_grid(pixels, mean, N_PIXELS, 0.0).astype("float64"),
            origin_grid(empty, "qv", "mean"),
            "empty",
        )


class TestDiskDensity:
    @pytest.mark.parametrize("shape", [(1, 9), (5, 7), (61, 83), (200, 13)])
    @pytest.mark.parametrize("dtype", [np.int64, np.float64])
    @pytest.mark.parametrize("workers", [1, 3, 8])
    def test_matches_whole_image_convolve(self, shape, dtype, workers):
        rng = np.random.default_rng(1)
        image = (
            (rng.random(shape) * 7).astype(dtype)
            if dtype == np.int64
            else rng.normal(0, 1, shape)
        )
        y, x = np.ogrid[-3:4, -3:4]
        kernel = ((x**2 + y**2) <= 9).astype(image.dtype)
        expected = np.flipud(convolve(image, kernel, mode="constant", cval=0))
        got = transcript_density_image.disk_density(image, 3, workers)
        assert got.flags.c_contiguous
        assert_same_array(got, expected, "disk density")


def legacy_prior(values, tmp, col, thresh, std, tail):
    """qv_image/ac_image at origin/dev: from_dask_array, the dask prior, ddf_to_parquet."""
    ac_or_qv_legacy = load_legacy("ac_or_qv", "spoqc.priors.hqtr")
    image_ddf = dd.from_dask_array(da.from_array(values, chunks=1000), columns=[col])
    image_ddf = ac_or_qv_legacy.calc_prob_pixel_stuff_v2(
        image_ddf, str(tmp), thresh, std, tail, col
    )
    norm_p = image_ddf[f"norm_p_{col}"].compute().to_numpy()
    ddf_to_parquet(image_ddf, "hqtr", str(tmp), [], "prior")
    return norm_p


def assert_same_dir(a, b):
    assert sorted(os.listdir(a)) == sorted(os.listdir(b))
    for name in os.listdir(b):
        assert open(f"{a}/{name}", "rb").read() == open(f"{b}/{name}", "rb").read(), (
            name
        )


class TestPrior:
    @pytest.mark.parametrize(
        "n,thresh,std,tail",
        [
            (12_345, 20.0, 3, "left"),
            (5_000, 0.4, 1, "left"),
            (7_001, 0.4, 1, "right"),
            (3_000, 1.0, 0.5, None),
        ],
    )
    def test_prior_and_parquet_match_origin(
        self, tmp_path, monkeypatch, n, thresh, std, tail
    ):
        monkeypatch.setattr(ac_or_qv, "ROWS_PER_TASK", 1024)  # many slices
        rng = np.random.default_rng(n)
        values = np.where(rng.random(n) < 0.4, 0.0, rng.gamma(2.0, thresh, n))
        (tmp_path / "old").mkdir()
        (tmp_path / "new").mkdir()
        expected = legacy_prior(
            values, tmp_path / "old", "x_density", thresh, std, tail
        )
        norm_p, part_columns = ac_or_qv.calc_prob_pixel_stuff_v2(
            values, str(tmp_path / "new"), thresh, std, tail, "x_density", 3
        )
        # the merged Gaussian prior (priors.gaussian) differs from origin/dev in the last bits
        assert np.max(np.abs(norm_p - expected)) <= 16 * np.finfo(np.float64).eps
        new_dir = str(tmp_path / "new" / "hqtr_output_prior")
        parquet.write_parts(new_dir, n, part_columns, range(0, n, 3_000), 3)
        # the new layout: 3,000-row parts, only the columns readers use; the same values
        new = dd.read_parquet(new_dir, calculate_divisions=True)
        old = dd.read_parquet(str(tmp_path / "old" / "hqtr_output_prior")).compute()
        assert list(new.columns) == ["x_density", "norm_p_x_density"]
        assert new.divisions == (*range(0, n, 3_000), n - 1)
        new = new.compute()
        assert_same_array(new["x_density"].to_numpy(), old["x_density"].to_numpy(), "x_density")
        assert_same_array(new["norm_p_x_density"].to_numpy(), norm_p, "norm_p_x_density")
        assert new.index.equals(old.index)

    def test_constant_image_scales_by_one(self, tmp_path):
        values = np.zeros(2_500)
        expected = legacy_prior(values, tmp_path, "qv_density", 20.0, 3, "left")
        norm_p, _ = ac_or_qv.calc_prob_pixel_stuff_v2(
            values, str(tmp_path), 20.0, 3, "left", "qv_density", 2
        )
        assert_same_array(norm_p, expected, "constant")

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
