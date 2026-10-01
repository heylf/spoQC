"""The writer of spoQC's per-pixel parquet directories.

`write_parts` writes, byte for byte, the files dask's
    ddf.to_parquet(path, engine="pyarrow", write_index=True, overwrite=True)
writes for a frame of in-memory columns with a default (0..n-1) index split into partitions
starting at `starts` (the call origin/dev's helperfuncs.ddf_to_parquet made), without a dask
graph: part.{i}.parquet holds rows starts[i] up to starts[i + 1] with their row positions as the
int64 index '__null_dask_index__', so dd.read_parquet(path, calculate_divisions=True) sees the
same partitions and divisions. The parts are built and written on `threads` threads.
With dask_index=False each part is instead the file pq.write_table writes for
pa.Table.from_arrays of the part's columns alone: no index, no pandas metadata, NaN kept as NaN
(spoQC's per-pixel metric columns, written by helperfuncs.nparr_to_parquet).
"""

import os
import shutil
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .threads import map_slices

INDEX_NAME = "__null_dask_index__"  # what dask names an unnamed index it writes
# Rows per part of spoQC's per-pixel parquet directories (hqtr/hqpr mask_raw and mask_smoothed_raw,
# and the metric parquets helperfuncs.nparr_to_parquet writes); for the masks:
# 218 parts at 913 Mpx, each written as four 1,048,576-row row groups (pq.write_table's default),
# which raster.read_pixel_columns decodes in parallel. At 10,000-row parts (origin/dev, 91,296
# files) the read was bound by per-file Python overhead; from 1M rows up it is decode-bound
# (docs/perf/mask_layout.md).
PART_ROWS = 1 << 22


def write_parts(
    path: str,
    n_rows: int,
    part_columns: Callable,
    starts: Sequence[int],
    threads: int,
    *,
    dask_index: bool = True,
) -> None:
    """
    Writes rows 0..n_rows-1 as dask's partitioned parquet directory at `path` (replacing it, or
    a single parquet file of that name).

    starts: the first row of each part, ascending from 0 (e.g. range(0, n_rows, rows_per_part)).
    part_columns(start, stop) returns the part's columns, in order, as a dict of 1-D numpy
    arrays of length stop - start; every call must give the same names and dtypes.
    """
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)
    os.makedirs(path)
    if n_rows == 0:
        return
    starts = [int(s) for s in starts]
    if (
        starts[0] != 0
        or any(b <= a for a, b in zip(starts, starts[1:]))
        or starts[-1] >= n_rows
    ):
        raise ValueError(
            f"part starts must ascend from 0 below {n_rows}: {starts[:3]} ... {starts[-3:]}"
        )
    stops = [*starts[1:], n_rows]
    first = part_columns(0, stops[0])
    if dask_index:
        empty = pd.DataFrame({name: values[:0] for name, values in first.items()})
        empty.index = pd.Index(np.arange(0, dtype=np.int64), name=INDEX_NAME)
        schema = pa.Table.from_pandas(empty, nthreads=1, preserve_index=True).schema

    def write(parts):
        i = parts.start
        columns = first if i == 0 else part_columns(starts[i], stops[i])
        if dask_index:
            # from_pandas: NaN becomes null, as pa.Table.from_pandas (dask's path) makes it
            arrays = [pa.array(values, from_pandas=True) for values in columns.values()]
            arrays.append(pa.array(np.arange(starts[i], stops[i], dtype=np.int64)))
            table = pa.Table.from_arrays(arrays, schema=schema)
        else:
            table = pa.Table.from_arrays([pa.array(v) for v in columns.values()], names=list(columns))
        pq.write_table(table, f"{path}/part.{i}.parquet", compression="snappy")

    map_slices(write, len(starts), 1, threads)


def columns_of(columns: dict) -> Callable:
    """part_columns for in-memory full-length arrays: their rows start..stop-1."""
    return lambda start, stop: {
        name: values[start:stop] for name, values in columns.items()
    }
