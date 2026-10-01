"""Whole-image raster primitives: read a per-pixel parquet column.

Every function is exact (bit-identical to the dask code it replaces) and splits its work
over `threads` (parquet row groups).
"""

import os

import numpy as np
import pyarrow.parquet as pq
from dask.utils import natural_sort_key

from .threads import map_slices


def _part_files(path):
    """The parquet files of a per-pixel parquet (a single file, or a directory of part.N.parquet)
    in the order dask.dataframe.read_parquet reads them."""
    if not os.path.isdir(path):
        return [path]
    # dask orders its part files naturally (part.2 before part.10), not lexicographically.
    names = sorted((n for n in os.listdir(path) if n.endswith(".parquet")), key=natural_sort_key)
    if not names:
        raise FileNotFoundError(f"no .parquet part files in {path}")
    return [os.path.join(path, n) for n in names]


def read_pixel_columns(path, columns, n_rows, threads, *, out=None):
    """Read columns of a per-pixel parquet (a single file, or a dask directory of part.N.parquet)
    into 1-D numpy arrays of n_rows values each, in the row order dask.dataframe.read_parquet
    returns, as {column: array}.

    out: {column: preallocated 1-D array of n_rows values} (any subset of `columns`, e.g. a column
    of an F-ordered matrix) to fill instead of allocating; values are cast into it by assignment,
    and the returned dict holds those arrays.

    The footers are read on `threads` threads, then every row group is decoded once for all
    `columns` (pyarrow releases the GIL while decoding) and copied into the preallocated arrays,
    one row group per task on `threads` threads.

    Nulls come out as dask 2026.1 returns them: NaN in a float column, and a null anywhere in an
    integer column promotes the whole column to float64 with NaN. Raises when there are no part
    files, when a part's type for a column differs from the first part's, when a non-numeric
    column holds nulls (dask would return an object array), when the files do not hold exactly
    n_rows rows, or when an integer column read into `out` holds nulls (it cannot be promoted in
    place).
    """
    files = _part_files(path)
    metas = map_slices(lambda i: pq.read_metadata(files[i.start]), len(files), 1, threads)
    types = {column: metas[0].schema.to_arrow_schema().field(column).type for column in columns}
    pieces = []  # (file index, row group)
    for i, meta in enumerate(metas):
        schema = meta.schema.to_arrow_schema()
        for column, type_ in types.items():
            if schema.field(column).type != type_:
                raise ValueError(
                    f"parts of {path} disagree on the type of {column}: "
                    f"{type_} in {files[0]}, {schema.field(column).type} in {files[i]}"
                )
        pieces += [(i, group) for group in range(meta.num_row_groups)]
    sizes = [metas[i].row_group(group).num_rows for i, group in pieces]
    starts = np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)])
    if starts[-1] != n_rows:
        raise ValueError(f"{path} holds {starts[-1]} rows, {n_rows} expected")
    given = {} if out is None else out
    for column, array in given.items():
        if column not in types or array.shape != (n_rows,):
            raise ValueError(f"out[{column!r}] has shape {array.shape}, not a ({n_rows},) column of {columns}")
    dtypes = {column: np.dtype(type_.to_pandas_dtype()) for column, type_ in types.items()}
    out = {
        column: given[column] if column in given else np.empty(n_rows, dtype=dtypes[column])
        for column in columns
    }
    null_rows = {column: [] for column in columns}  # integer columns: rows to set NaN at the end

    def read(piece):
        k = piece.start
        i, group = pieces[k]
        table = pq.ParquetFile(files[i], metadata=metas[i], pre_buffer=False).read_row_group(
            group, columns=columns, use_threads=False
        )
        start, stop = starts[k], starts[k + 1]
        for column in columns:
            values = table.column(column)
            if values.null_count:
                kind = dtypes[column].kind
                if kind in "iu" and column in given:
                    raise ValueError(
                        f"{values.null_count} nulls in the integer column {column} of {path}, "
                        f"read into a preallocated {out[column].dtype} array"
                    )
                if kind in "iu":
                    null_rows[column].append(
                        start + np.flatnonzero(values.is_null().to_numpy(zero_copy_only=False))
                    )
                    values = values.fill_null(0)
                elif kind != "f":
                    raise ValueError(
                        f"{values.null_count} nulls in the {types[column]} column {column} of {path}"
                    )
            out[column][start:stop] = values.to_numpy()  # float nulls: NaN; cast into `out`

    map_slices(read, len(pieces), 1, threads)
    for column, rows in null_rows.items():
        if rows:  # pandas' integer-with-NaN promotion, as dask applies it
            out[column] = out[column].astype(np.float64)
            out[column][np.concatenate(rows)] = np.nan
    return out
