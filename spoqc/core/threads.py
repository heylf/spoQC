"""The single thread budget of a spoQC run.

`configure(n)` must run before the libraries below are imported: polars sizes its
pool at first import, numba reads NUMBA_NUM_THREADS at import, OpenBLAS/MKL/OpenMP
read their variables when the library loads (pyarrow's CPU pool follows
OMP_NUM_THREADS), dask and zarr read DASK_* / ZARR_* variables into their config at
import, and pyarrow reads ARROW_IO_THREADS when its I/O pool starts.
OpenCV and numcodecs' blosc keep their own pools, which are set by call; numcodecs
does not follow BLOSC_NTHREADS (measured), so it is set explicitly.
Pool sizes elsewhere in spoQC come from `N`.
"""

from __future__ import annotations

import os
import sys

ENV_VARS = (
    "POLARS_MAX_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLOSC_NTHREADS",
    "DASK_NUM_WORKERS",
    "ZARR_THREADING__MAX_WORKERS",
    "ARROW_IO_THREADS",
)
_FIXED_AT_IMPORT = (
    "numpy",
    "numba",
    "polars",
    "dask",
    "ovrlpy",
    "pyarrow",
    "zarr",
    "numcodecs",
    "cv2",
)

N: int | None = None


def configure(n: int) -> None:
    global N
    if n < 1:
        raise ValueError(f"thread count must be >= 1, got {n}")
    imported = [m for m in _FIXED_AT_IMPORT if m in sys.modules]
    if imported:
        raise RuntimeError(f"threads.configure must run before importing {imported}")
    for var in ENV_VARS:
        os.environ[var] = str(n)
    # imported only now, so numpy (which both load) starts under the variables above
    import cv2
    import numcodecs.blosc

    cv2.setNumThreads(n)  # OpenCV otherwise starts one thread per host CPU
    numcodecs.blosc.set_nthreads(n)
    N = n


def budget() -> int:
    """N, for code without a threads argument; raises if configure() never ran."""
    if N is None:
        raise RuntimeError("spoqc.core.threads.configure(n) has not run, so there is no thread budget")
    return N


def map_slices(fn, n: int, step: int, workers: int) -> list:
    """[fn(s) for s in slice(0, step), slice(step, 2 * step), ... up to n], run on `workers` threads.

    For whole-array numpy work split by rows: numpy and scipy release the GIL inside their loops,
    and elementwise or row-local work gives the same values whatever the split.
    """
    from concurrent.futures import ThreadPoolExecutor

    slices = [slice(start, min(start + step, n)) for start in range(0, n, step)]
    with ThreadPoolExecutor(workers) as executor:
        return list(executor.map(fn, slices))


def map_rows(fn, arrays: tuple, workers: int):
    """fn(*arrays) for an elementwise (row-local) fn, computed on blocks of rows on `workers` threads."""
    import numpy as np  # not at module level: configure() must run before numpy loads

    n_rows = arrays[0].shape[0]
    out = np.empty(arrays[0].shape, dtype=fn(*(a[:1] for a in arrays)).dtype)

    def block(rows):
        out[rows] = fn(*(a[rows] for a in arrays))

    map_slices(block, n_rows, max(-(-n_rows // (4 * workers)), 1), workers)
    return out
