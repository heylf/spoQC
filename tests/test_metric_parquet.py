"""Per-pixel metric parquets: helperfuncs.nparr_to_parquet writes a directory of parts through
core.parquet.write_parts (dask_index=False) instead of one pyarrow file.

Differential against the verbatim previous writer and reader:
the values every reader returns are bit-identical, and each part is byte for byte the file the
previous writer writes for that part's rows. Mutants of the new code must fail the comparison.
"""

import inspect
import os
import types

import dask.dataframe as dd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spoqc import helperfuncs
from spoqc.core import parquet, raster
from spoqc.image_analysis import pixel_scoring_dask

N = 5_003  # rows; parts of PART_ROWS rows, the last one short
PART_ROWS = 400  # 13 parts: part.10 .. part.12 sort before part.2 lexicographically


# ---- verbatim previous code (helperfuncs.py) -----------------------------------------------
def old_nparr_to_parquet(np_arr, prefix, spoqc_tmp_folder, suffix):
    outfile = f"{spoqc_tmp_folder}/{prefix}_output_{suffix}.parquet"
    table = pa.Table.from_arrays([pa.array(np_arr)], names=[prefix])
    pq.write_table(table, outfile)


def old_read_pixel_features(tmp_files, threads):
    import concurrent.futures

    n_rows = pq.ParquetFile(tmp_files[0]).metadata.num_rows
    features = np.empty((n_rows, len(tmp_files)), dtype=np.float32, order="F")
    with concurrent.futures.ThreadPoolExecutor(threads) as pool:
        for j, tmp_file in enumerate(tmp_files):
            metadata = pq.ParquetFile(tmp_file).metadata
            starts = np.cumsum(
                [0]
                + [
                    metadata.row_group(i).num_rows
                    for i in range(metadata.num_row_groups)
                ]
            )

            def read_row_group(i, tmp_file=tmp_file, starts=starts, j=j):
                values = (
                    pq.ParquetFile(tmp_file)
                    .read_row_group(i, use_threads=False)
                    .column(0)
                    .to_numpy()
                )
                features[starts[i] : starts[i + 1], j] = values

            list(pool.map(read_row_group, range(metadata.num_row_groups)))
    return features


# ---- data ------------------------------------------------------------------------------------
def metric_arrays(rng):
    """The dtypes structure analysis writes: float64 (with NaN and +-inf, e.g. log of 0),
    float32 (texture metrics), uint8 (relevance), int64 (transcript density)."""
    f64 = rng.gamma(2.0, 1.0, N)
    f64[[3, 400, 4999]] = np.nan
    f64[[7, 801]] = np.inf
    f64[9] = -np.inf
    f64[10] = -0.0
    return {
        "lbp": f64,
        "entropy": rng.random(N).astype(np.float32),
        "relevance": rng.integers(0, 2, N).astype(np.uint8),
        "transcript_density": rng.integers(0, 50, N).astype(np.int64),
    }


@pytest.fixture
def arrays():
    return metric_arrays(np.random.default_rng(11))


@pytest.fixture
def small_parts(monkeypatch):
    monkeypatch.setattr(parquet, "PART_ROWS", PART_ROWS)
    monkeypatch.setattr(helperfuncs, "PIXEL_FEATURES", {})


def write_both(tmp_path, arrays, module=helperfuncs):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    for name, values in arrays.items():
        old_nparr_to_parquet(values, name, str(old), "hqtr")
        module.nparr_to_parquet(values, name, str(new), "hqtr")
    return old, new


def bits(a):
    return a.dtype.str, a.tobytes()


def part_files(path):
    names = sorted(os.listdir(path), key=lambda n: int(n.split(".")[1]))
    assert names == [f"part.{i}.parquet" for i in range(len(names))]
    return [os.path.join(path, n) for n in names]


def assert_same_values(old, new, arrays):
    """Every reader returns bit-identical values from the old file and the new directory."""
    for name, values in arrays.items():
        o, n = f"{old}/{name}_output_hqtr.parquet", f"{new}/{name}_output_hqtr.parquet"
        assert os.path.isfile(o) and os.path.isdir(n)
        # the stored column: same arrow type, no nulls (NaN stays NaN), same bits
        new_column = pa.concat_arrays(
            [
                chunk
                for f in part_files(n)
                for chunk in pq.read_table(f).column(name).chunks
            ]
        )
        old_column = pq.read_table(o).column(name).combine_chunks()
        assert (
            new_column.type == old_column.type
            and new_column.null_count == old_column.null_count == 0
        )
        assert (
            bits(new_column.to_numpy()) == bits(old_column.to_numpy()) == bits(values)
        ), name
        # readers: core.raster (load_intensity_image, combine masks, bounding boxes) and dask
        # (additional_analysis.analysis_funcs)
        assert bits(raster.read_pixel_columns(n, [name], N, 3)[name]) == bits(
            raster.read_pixel_columns(o, [name], N, 3)[name]
        ), name
        assert bits(dd.read_parquet(n)[name].compute().to_numpy()) == bits(
            dd.read_parquet(o)[name].compute().to_numpy()
        ), name
        assert raster.pixel_rows(n) == pq.ParquetFile(o).metadata.num_rows == N


class TestMetricParquet:
    def test_every_reader_returns_the_previous_values(
        self, tmp_path, arrays, small_parts
    ):
        old, new = write_both(tmp_path, arrays)
        assert_same_values(old, new, arrays)

    def test_each_part_is_the_file_the_previous_writer_writes_for_its_rows(
        self, tmp_path, arrays, small_parts
    ):
        _, new = write_both(tmp_path, arrays)
        for name, values in arrays.items():
            files = part_files(f"{new}/{name}_output_hqtr.parquet")
            assert len(files) == -(-N // PART_ROWS)
            for i, file in enumerate(files):
                ref = tmp_path / f"ref_{name}_{i}"
                ref.mkdir()
                old_nparr_to_parquet(
                    values[i * PART_ROWS : (i + 1) * PART_ROWS], name, str(ref), "hqtr"
                )
                with (
                    open(file, "rb") as a,
                    open(f"{ref}/{name}_output_hqtr.parquet", "rb") as b,
                ):
                    assert a.read() == b.read(), (name, i)

    @pytest.mark.parametrize("threads", [1, 4])
    def test_read_pixel_features_from_parquet_matches_the_previous_reader(
        self, tmp_path, arrays, small_parts, threads
    ):
        old, new = write_both(tmp_path, arrays)
        helperfuncs.PIXEL_FEATURES.clear()  # a separate clustering run: read from disk
        files = lambda root: [f"{root}/{name}_output_hqtr.parquet" for name in arrays]
        expected = old_read_pixel_features(files(old), threads)
        assert bits(helperfuncs.read_pixel_features(files(new), threads)) == bits(
            expected
        )
        assert bits(helperfuncs.read_pixel_features(files(old), threads)) == bits(
            expected
        )

    def test_in_memory_handoff_is_the_float32_column(
        self, tmp_path, arrays, small_parts
    ):
        _, new = write_both(tmp_path, arrays)
        for name, values in arrays.items():
            column = helperfuncs.PIXEL_FEATURES[
                os.path.abspath(f"{new}/{name}_output_hqtr.parquet")
            ]
            assert bits(column) == bits(np.asarray(values, dtype=np.float32)), name
            if values.dtype == np.float32:
                assert column is values  # no copy, as np.asarray made none

    def test_pixel_feature_files_finds_the_directories(self, tmp_path, small_parts):
        for name in pixel_scoring_dask.PIXEL_FEATURE_NAMES["hqtr"]:
            helperfuncs.nparr_to_parquet(np.zeros(10), name, str(tmp_path), "hqtr")
        files = pixel_scoring_dask.pixel_feature_files(str(tmp_path), "hqtr", "hqtr")
        assert all(os.path.isdir(f) for f in files) and len(files) == 8

    def test_rewriting_over_a_previous_single_file_replaces_it(
        self, tmp_path, arrays, small_parts
    ):
        old_nparr_to_parquet(arrays["lbp"][:7], "lbp", str(tmp_path), "hqtr")
        helperfuncs.nparr_to_parquet(arrays["lbp"], "lbp", str(tmp_path), "hqtr")
        path = f"{tmp_path}/lbp_output_hqtr.parquet"
        assert bits(raster.read_pixel_columns(path, ["lbp"], N, 2)["lbp"]) == bits(arrays["lbp"])

    def test_directory_of_the_production_part_size(self, tmp_path, monkeypatch):
        monkeypatch.setattr(helperfuncs, "PIXEL_FEATURES", {})
        values = np.arange(3 * parquet.PART_ROWS // 2, dtype=np.float64)
        helperfuncs.nparr_to_parquet(values, "energy", str(tmp_path), "hqtr")
        files = part_files(f"{tmp_path}/energy_output_hqtr.parquet")
        assert [pq.ParquetFile(f).metadata.num_rows for f in files] == [
            len(values) - len(values) // 3,
            len(values) // 3,
        ]


# ---- mutants: each must break the comparison ---------------------------------------------------
def mutant(module, old, new):
    source = inspect.getsource(module)
    assert source.count(old) == 1, old
    clone = types.ModuleType(f"{module.__name__}_mutant")
    clone.__package__ = module.__package__
    clone.__file__ = module.__file__
    exec(compile(source.replace(old, new), module.__file__, "exec"), clone.__dict__)
    return clone


# Each mutant is a list of (old, new) source replacements in core/parquet.py.
PARQUET_MUTANTS = {
    "NaN written as null": [(
        "table = pa.Table.from_arrays([pa.array(v) for v in columns.values()], names=list(columns))",
        "table = pa.Table.from_arrays([pa.array(v, from_pandas=True) for v in columns.values()], names=list(columns))",
    )],
    "parts off by one row": [(
        "columns = first if i == 0 else part_columns(starts[i], stops[i])",
        "columns = first if i == 0 else part_columns(starts[i] + 1, stops[i] + 1)",
    )],
    "dask index written": [
        ("    if dask_index:\n        empty", "    if True:\n        empty"),
        ("        if dask_index:\n            # from_pandas", "        if True:\n            # from_pandas"),
    ],
}  # fmt: skip


def parquet_mutant(replacements):
    source = inspect.getsource(parquet)
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    clone = types.ModuleType("spoqc.core.parquet_mutant")
    clone.__package__ = parquet.__package__
    exec(compile(source, parquet.__file__, "exec"), clone.__dict__)
    clone.PART_ROWS = parquet.PART_ROWS  # small_parts' part size, not the production one
    return clone


@pytest.mark.parametrize("name", PARQUET_MUTANTS)
def test_parquet_mutant_is_caught(name, tmp_path, arrays, small_parts, monkeypatch):
    monkeypatch.setattr(helperfuncs, "parquet", parquet_mutant(PARQUET_MUTANTS[name]))
    old, new = write_both(tmp_path, arrays)
    with pytest.raises(AssertionError):
        assert_same_values(old, new, arrays)
        for metric, values in arrays.items():  # the byte-level part check
            for i, file in enumerate(part_files(f"{new}/{metric}_output_hqtr.parquet")):
                ref = tmp_path / f"ref_{metric}_{i}"
                ref.mkdir()
                old_nparr_to_parquet(values[i * PART_ROWS : (i + 1) * PART_ROWS], metric, str(ref), "hqtr")
                with open(file, "rb") as a, open(f"{ref}/{metric}_output_hqtr.parquet", "rb") as b:
                    assert a.read() == b.read(), (metric, i)


def test_lexicographic_part_order_is_caught(tmp_path, arrays, small_parts, monkeypatch):
    clone = mutant(raster, "key=natural_sort_key)", "key=None)")
    old, new = write_both(tmp_path, arrays)
    with pytest.raises(AssertionError):
        assert bits(
            clone.read_pixel_columns(f"{new}/lbp_output_hqtr.parquet", ["lbp"], N, 2)["lbp"]
        ) == bits(arrays["lbp"])


def test_float32_handoff_cast_mutant_is_caught(
    tmp_path, arrays, small_parts, monkeypatch
):
    clone = mutant(
        helperfuncs,
        "column = np.empty(len(np_arr), dtype=np.float32)",
        "column = np.empty(len(np_arr), dtype=np.float16)",
    )
    with pytest.raises(AssertionError):
        write_both(tmp_path, arrays, module=clone)
        for name, values in arrays.items():
            column = clone.PIXEL_FEATURES[
                os.path.abspath(f"{tmp_path}/new/{name}_output_hqtr.parquet")
            ]
            assert bits(column) == bits(np.asarray(values, dtype=np.float32)), name
