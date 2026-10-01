"""core.raster.read_pixel_columns vs the reader it replaced, verbatim below.

The old reader opened every part twice per column (a serial metadata pass, then a per-row-group
thread pool); the new one reads the footers once, then decodes every row group once for all
requested columns on a thread pool. Both are compared byte for byte, with their dtypes, on the
layouts spoQC writes (core.parquet's dask-identical parts, uneven dask partitions, a
many-row-group single file, parts of several row groups) and on nulls. Mutants of the new
reader must fail.
"""
import inspect
import os
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dask.utils import natural_sort_key

from spoqc.core import parquet, raster


def read_pixel_column(path, column, threads):
    """Read one column of a per-pixel parquet (a single file, or a dask directory of part.N.parquet)
    into a 1-D numpy array in the row order dask.dataframe.read_parquet returns.

    Row groups are decoded in parallel into one preallocated array. Nulls come out as dask 2026.1
    returns them: NaN in a float column, and a null anywhere in an integer column promotes the whole
    column to float64 with NaN (the parquet statistics' null counts decide this up front). Raises when
    there are no part files, when the parts disagree on the column's type, or when a non-numeric
    column holds nulls (dask would return an object array).
    """
    if os.path.isdir(path):
        # dask orders its part files naturally (part.2 before part.10), not lexicographically.
        names = sorted(
            (n for n in os.listdir(path) if n.endswith(".parquet")),
            key=natural_sort_key,
        )
        files = [os.path.join(path, n) for n in names]
        if not files:
            raise FileNotFoundError(f"no .parquet part files in {path}")
    else:
        files = [path]
    pieces, sizes, types, nulls = [], [], set(), 0
    for file in files:
        meta = pq.ParquetFile(file)
        types.add(meta.schema_arrow.field(column).type)
        index = meta.schema_arrow.get_field_index(column)
        for group in range(meta.metadata.num_row_groups):
            pieces.append((file, group))
            sizes.append(meta.metadata.row_group(group).num_rows)
            nulls += meta.metadata.row_group(group).column(index).statistics.null_count
    if len(types) != 1:
        raise ValueError(f"parts of {path} disagree on the type of {column}: {sorted(map(str, types))}")
    dtype = np.dtype(types.pop().to_pandas_dtype())
    if nulls and dtype.kind in "iu":
        dtype = np.dtype(np.float64)  # pandas' integer-with-NaN promotion, as dask applies it
    elif nulls and dtype.kind != "f":
        raise ValueError(f"{nulls} nulls in the {dtype} column {column} of {path}")
    starts = np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)])
    out = np.empty(int(starts[-1]), dtype=dtype)

    def read(i):
        file, group = pieces[i]
        values = (
            pq.ParquetFile(file)
            .read_row_group(group, columns=[column], use_threads=False)
            .column(0)
        )
        out[starts[i] : starts[i + 1]] = values.to_numpy()  # nulls: NaN (float64 for integers)

    with ThreadPoolExecutor(threads) as executor:
        list(executor.map(read, range(len(pieces))))
    return out


def write_pixel_parts(path, n_rows, starts, seed):
    """A mask_raw-like directory through core.parquet (the writer of every spoQC pixel directory)."""
    rng = np.random.default_rng(seed)
    columns = {
        "mask": rng.integers(0, 2, n_rows).astype(np.int8),
        "beliefs": rng.random(n_rows),
        "beliefs_smoothed": rng.random(n_rows).astype(np.float32),
        "cluster": rng.integers(0, 100, n_rows).astype(np.int32),
        "label": rng.integers(0, 3, n_rows).astype(np.int64),
    }
    parquet.write_parts(path, n_rows, parquet.columns_of(columns), starts, 3)
    return columns


def assert_same_as_old(path, columns, n_rows, threads=3):
    new = raster.read_pixel_columns(path, columns, n_rows, threads)
    assert list(new) == list(columns)
    for column in columns:
        old = read_pixel_column(path, column, 4)
        assert new[column].dtype == old.dtype, column
        assert new[column].tobytes() == old.tobytes(), column


ALL = ["mask", "beliefs", "beliefs_smoothed", "cluster", "label"]


@pytest.mark.parametrize("columns", [["mask", "beliefs"], ["beliefs_smoothed"], ALL, ["label", "mask"]])
def test_many_small_parts_like_hqtr(tmp_path, columns):
    """hqtr's layout: equal parts (here 500 rows; 10,000 in production), more than 10 of them."""
    n_rows = 23 * 500 + 137
    write_pixel_parts(f"{tmp_path}/d", n_rows, range(0, n_rows, 500), seed=1)
    assert len(os.listdir(f"{tmp_path}/d")) == 24
    assert_same_as_old(f"{tmp_path}/d", columns, n_rows)


def test_uneven_dask_partitions(tmp_path):
    n_rows = 10_007
    write_pixel_parts(f"{tmp_path}/d", n_rows, [*range(0, 13 * 770, 770)][:13], seed=2)  # 12 x 770 + 767
    assert_same_as_old(f"{tmp_path}/d", ALL, n_rows)


@pytest.mark.parametrize("threads", [1, 2, 7])
def test_parts_of_several_row_groups_in_any_thread_count(tmp_path, threads):
    """spoQC's layout: parts of several row groups (core.parquet.PART_ROWS rows of 1,048,576-row
    groups in production); row groups finish in any order on the pool, rows stay in order."""
    n_rows = 5 * 700 + 311
    path = f"{tmp_path}/d"
    os.makedirs(path)
    rng = np.random.default_rng(3)
    columns = {"mask": rng.integers(0, 2, n_rows).astype(np.int8), "beliefs": rng.random(n_rows)}
    for i, start in enumerate(range(0, n_rows, 700)):
        part = {name: values[start : start + 700] for name, values in columns.items()}
        pq.write_table(pa.table(part), f"{path}/part.{i}.parquet", row_group_size=128)
    assert pq.ParquetFile(f"{path}/part.0.parquet").metadata.num_row_groups == 6
    assert_same_as_old(path, ["beliefs", "mask"], n_rows, threads)
    new = raster.read_pixel_columns(path, ["beliefs", "mask"], n_rows, threads)
    assert all(new[name].tobytes() == columns[name].tobytes() for name in new)


def test_single_file_with_many_row_groups_like_hqcr(tmp_path):
    rng = np.random.default_rng(4)
    n_rows = 50_003
    table = pa.table({"hqcr_mask": rng.integers(0, 2, n_rows).astype(np.int8), "hqcr_beliefs": rng.random(n_rows)})
    pq.write_table(table, f"{tmp_path}/hqcr.parquet", row_group_size=1_000)
    assert pq.ParquetFile(f"{tmp_path}/hqcr.parquet").metadata.num_row_groups == 51
    assert_same_as_old(f"{tmp_path}/hqcr.parquet", ["hqcr_mask", "hqcr_beliefs"], n_rows)


@pytest.mark.parametrize("type_", [pa.int8(), pa.int64(), pa.uint16(), pa.float32(), pa.float64()])
@pytest.mark.parametrize("null_part", [0, 7, 11])
def test_nulls_in_one_part(tmp_path, type_, null_part):
    """A null in an early or a late part: NaN for floats, float64 with NaN for the whole integer column."""
    rng = np.random.default_rng(5)
    for i in range(12):
        values = rng.integers(0, 50, 40).tolist()
        if i == null_part:
            values[3] = values[17] = None
        pq.write_table(
            pa.table({"v": pa.array(values, type=type_), "w": pa.array(rng.random(40))}),
            f"{tmp_path}/part.{i}.parquet",
            row_group_size=16,
        )
    new = raster.read_pixel_columns(str(tmp_path), ["v", "w"], 12 * 40, 3)
    for column in ("v", "w"):
        old = read_pixel_column(str(tmp_path), column, 3)
        assert new[column].dtype == old.dtype and new[column].tobytes() == old.tobytes()
    assert np.isnan(new["v"]).sum() == 2


def test_out_is_filled_by_casting_assignment(tmp_path):
    """read_pixel_features' call: one column into a column of an F-ordered float32 matrix, the
    values the old reader returns cast to float32 (NaN kept); other columns are still allocated."""
    n_rows = 3 * 500 + 17
    write_pixel_parts(f"{tmp_path}/d", n_rows, range(0, n_rows, 500), seed=9)
    features = np.zeros((n_rows, 3), dtype=np.float32, order="F")
    got = raster.read_pixel_columns(
        f"{tmp_path}/d", ["beliefs", "label"], n_rows, 3, out={"beliefs": features[:, 1]}
    )
    assert np.shares_memory(got["beliefs"], features)
    assert features[:, 1].tobytes() == read_pixel_column(f"{tmp_path}/d", "beliefs", 3).astype(np.float32).tobytes()
    assert not features[:, [0, 2]].any()
    assert got["label"].tobytes() == read_pixel_column(f"{tmp_path}/d", "label", 3).tobytes()


@pytest.mark.parametrize("type_", [pa.float32(), pa.float64()])
def test_float_nulls_into_out_are_nan(tmp_path, type_):
    pq.write_table(pa.table({"v": pa.array([1.5, None, 3.0], type=type_)}), f"{tmp_path}/f.parquet")
    out = np.empty(3, dtype=np.float32)
    raster.read_pixel_columns(f"{tmp_path}/f.parquet", ["v"], 3, 2, out={"v": out})
    assert out[0] == 1.5 and np.isnan(out[1]) and out[2] == 3.0


@pytest.mark.parametrize("type_", [pa.int8(), pa.int64(), pa.uint16()])
def test_integer_nulls_into_out_raise(tmp_path, type_):
    pq.write_table(pa.table({"v": pa.array([1, None, 3], type=type_)}), f"{tmp_path}/i.parquet")
    with pytest.raises(ValueError, match="nulls in the integer column v"):
        raster.read_pixel_columns(f"{tmp_path}/i.parquet", ["v"], 3, 2, out={"v": np.empty(3, dtype=np.float32)})


@pytest.mark.parametrize("shape", [(2,), (4,), (3, 1)])
def test_out_of_the_wrong_shape_raises(tmp_path, shape):
    pq.write_table(pa.table({"v": pa.array([1.0, 2.0, 3.0])}), f"{tmp_path}/s.parquet")
    with pytest.raises(ValueError, match="has shape"):
        raster.read_pixel_columns(f"{tmp_path}/s.parquet", ["v"], 3, 2, out={"v": np.empty(shape)})


@pytest.mark.parametrize("n_rows", [12 * 40 - 1, 12 * 40 + 1])
def test_a_wrong_row_count_raises(tmp_path, n_rows):
    write_pixel_parts(f"{tmp_path}/d", 12 * 40, range(0, 12 * 40, 40), seed=6)
    with pytest.raises(ValueError, match=f"holds {12 * 40} rows, {n_rows} expected"):
        raster.read_pixel_columns(f"{tmp_path}/d", ["mask"], n_rows, 3)


def mutant(old, new):
    source = inspect.getsource(raster)
    assert source.count(old) == 1, old
    clone = types.ModuleType("raster_mutant")
    clone.__package__ = raster.__package__
    exec(compile(source.replace(old, new), raster.__file__, "exec"), clone.__dict__)
    return clone


MUTANTS = {
    "lexicographic part order": ("key=natural_sort_key)", "key=str)"),
    "row groups out of order": (
        "[(i, group) for group in range(meta.num_row_groups)]",
        "[(i, group) for group in reversed(range(meta.num_row_groups))]",
    ),
    "first row group only": ("for group in range(meta.num_row_groups)]", "for group in range(1)]"),
    "no row count check": ("    if starts[-1] != n_rows:\n", "    if starts[-1] > n_rows:\n"),
    "no type check": ("            if schema.field(column).type != type_:", "            if False:"),
    "one dtype for all": ("np.dtype(type_.to_pandas_dtype())", "np.dtype(np.float64)"),
    "no integer null promotion": ("        if rows:  # pandas'", "        if False:  # pandas'"),
    "nulls left as fill values": ("out[column][np.concatenate(rows)] = np.nan", "pass"),
    "out ignored": (
        "column: given[column] if column in given else np.empty(n_rows, dtype=dtypes[column])",
        "column: np.empty(n_rows, dtype=dtypes[column])",
    ),
    "integer nulls cast into out": ('if kind in "iu" and column in given:', "if False:"),
}


def mutant_differs(tmp_path, module):
    """True when the module's reader differs from the old one on a spoQC-like directory (parts
    of several row groups, more than 10 parts) or on nulls, or fails to raise on too few rows
    or on parts that disagree on a type, or mishandles `out` (columns of a preallocated matrix)."""
    n_rows = 2 * 3_001 + 29 * 500
    os.makedirs(f"{tmp_path}/d")
    rng = np.random.default_rng(8)
    for i, (start, stop) in enumerate(zip([0, 3_001, *range(6_002, n_rows, 500)], [3_001, 6_002, *range(6_502, n_rows, 500), n_rows])):
        part = {"mask": rng.integers(0, 2, stop - start).astype(np.int8), "beliefs": rng.random(stop - start)}
        pq.write_table(pa.table(part), f"{tmp_path}/d/part.{i}.parquet", row_group_size=1_000)
    os.makedirs(f"{tmp_path}/n")
    for i in range(12):
        values = list(range(40))
        if i == 7:
            values[3] = None
        pq.write_table(pa.table({"v": pa.array(values, type=pa.int32())}), f"{tmp_path}/n/part.{i}.parquet")
    for path, columns, n in ((f"{tmp_path}/d", ["mask", "beliefs"], n_rows), (f"{tmp_path}/n", ["v"], 480)):
        try:
            new = module.read_pixel_columns(path, columns, n, 3)
        except Exception:
            return True
        for column in columns:
            old = read_pixel_column(path, column, 4)
            if new[column].dtype != old.dtype or new[column].tobytes() != old.tobytes():
                return True
    # out: columns of an F-ordered float32 matrix, filled by casting assignment
    features = np.full((n_rows, 2), -1, dtype=np.float32, order="F")
    out = {"mask": features[:, 0], "beliefs": features[:, 1]}
    try:
        got = module.read_pixel_columns(f"{tmp_path}/d", ["mask", "beliefs"], n_rows, 3, out=out)
    except Exception:
        return True
    for j, column in enumerate(["mask", "beliefs"]):
        expected = read_pixel_column(f"{tmp_path}/d", column, 4).astype(np.float32)
        if got[column] is not out[column] or features[:, j].tobytes() != expected.tobytes():
            return True
    try:  # an integer column with nulls cannot be promoted in place
        module.read_pixel_columns(f"{tmp_path}/n", ["v"], 480, 3, out={"v": np.empty(480, dtype=np.float32)})
    except ValueError:
        pass
    else:
        return True
    os.makedirs(f"{tmp_path}/t")
    pq.write_table(pa.table({"v": pa.array([1, 2], type=pa.int8())}), f"{tmp_path}/t/part.0.parquet")
    pq.write_table(pa.table({"v": pa.array([3, 4], type=pa.int64())}), f"{tmp_path}/t/part.1.parquet")
    for path, columns, n in ((f"{tmp_path}/d", ["mask"], n_rows + 1), (f"{tmp_path}/t", ["v"], 4)):
        try:
            module.read_pixel_columns(path, columns, n, 3)
        except ValueError:
            continue
        except Exception:
            return True
        return True
    return False


def test_the_mutant_check_passes_the_real_reader(tmp_path):
    assert not mutant_differs(tmp_path, raster)


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_caught(tmp_path, name):
    assert mutant_differs(tmp_path, mutant(*MUTANTS[name])), f"mutant {name!r} survived"
