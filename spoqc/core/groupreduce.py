"""Reductions over ragged groups stored as one flat array.

A group layout is given by `offsets` (length n_groups + 1): group g owns the
flat elements offsets[g]:offsets[g + 1].
"""

from typing import Callable

import numba
import numpy as np
import polars as pl

# Rows per partial sum in group_sum. Fixed, so the summation order never depends on the thread count.
GROUP_SUM_CHUNK = 1 << 20


def group_offsets(group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    """Offsets of each group in a flat array whose group_idx is sorted."""
    return np.searchsorted(group_idx, np.arange(n_groups + 1))


def cyclic_shift(offsets: np.ndarray, shift: int) -> np.ndarray:
    """Flat index of the element `shift` places after each element, wrapping around within its group."""
    sizes = np.diff(offsets)
    start = np.repeat(offsets[:-1], sizes)
    return start + (np.arange(offsets[-1]) - start + shift) % np.repeat(sizes, sizes)


def group_count(group_idx: np.ndarray, mask: np.ndarray, n_groups: int) -> np.ndarray:
    """Number of elements per group where mask is True."""
    return np.bincount(group_idx[mask], minlength=n_groups)


def ragged_reduce(values: np.ndarray, offsets: np.ndarray, reducer: Callable, out: np.ndarray) -> np.ndarray:
    """
    Writes reducer(values[offsets[g]:offsets[g + 1]]) into out[g] for every non-empty group.

    Groups of equal size k are reduced together as one (groups, k) matrix with
    reducer(..., axis=1); numpy reduces each row exactly as it reduces the 1-D
    group, so the result is bit-identical to calling reducer per group.
    Empty groups keep their value in `out`.
    """
    sizes = np.diff(offsets)
    for k in np.unique(sizes[sizes > 0]):
        groups = np.flatnonzero(sizes == k)
        out[groups] = reducer(values[offsets[groups][:, None] + np.arange(k)], axis=1)
    return out


def ragged_lists(values: list, offsets: np.ndarray) -> list:
    """One Python list per group."""
    return [values[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


@numba.njit(parallel=True)
def _chunk_group_sums(group_idx, values, n_groups, chunk):
    n_chunks = (len(group_idx) + chunk - 1) // chunk
    sums = np.zeros((n_chunks, n_groups))
    counts = np.zeros((n_chunks, n_groups), dtype=np.int64)
    for c in numba.prange(n_chunks):
        for i in range(c * chunk, min(len(group_idx), (c + 1) * chunk)):
            sums[c, group_idx[i]] += values[i]
            counts[c, group_idx[i]] += 1
    return sums, counts


def group_sum(group_idx: np.ndarray, values: np.ndarray, n_groups: int):
    """Per-group float64 sum of values and element count, in a fixed order.

    Each GROUP_SUM_CHUNK rows are summed in row order, then the chunk sums are added in chunk
    order, so the result is identical for any thread count and run to run.
    """
    sums, counts = _chunk_group_sums(group_idx, values, n_groups, GROUP_SUM_CHUNK)
    total, count = sums[0].copy(), counts[0].copy()
    for c in range(1, len(sums)):
        total += sums[c]
        count += counts[c]
    return total, count


def pixel_groups(x: np.ndarray, y: np.ndarray, x_range: tuple, y_range: tuple):
    """
    Groups points by the integer pixel (x, y) they sit on, for the grid x0 <= x < x1, y0 <= y < y1.

    Pixel p = (y - y0) * (x1 - x0) + (x - x0) is the row-major position in the (y1 - y0, x1 - x0)
    image, the order of MultiIndex.from_product([range(y0, y1), range(x0, x1)]). Points off the
    grid are dropped, as a reindex onto that grid drops them.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: `pixels`, the occupied pixels ascending;
        `rows`, the positions of the on-grid points sorted by pixel, then position; `offsets`,
        so that pixel pixels[g] holds rows[offsets[g]:offsets[g + 1]].
    """
    (x0, x1), (y0, y1) = x_range, y_range
    n = len(x)
    if (x1 - x0) * (y1 - y0) * max(n, 1) >= 2**63:
        raise OverflowError("pixel * n_points does not fit in int64")
    rows = np.flatnonzero((x >= x0) & (x < x1) & (y >= y0) & (y < y1))
    # (pixel, position) keys are unique, so any sort (polars sorts on all its threads) gives this order.
    key = pl.Series((y[rows] - y0) * (x1 - x0) + (x[rows] - x0)) * n + pl.Series(rows)
    flat, rows = np.divmod(key.sort().to_numpy(), n)
    if len(flat) == 0:
        return flat, rows, np.zeros(1, dtype=np.int64)
    starts = np.flatnonzero(np.diff(flat)) + 1
    pixels = flat[np.concatenate(([0], starts))]
    offsets = np.concatenate(([0], starts, [len(flat)]))
    return pixels, rows, offsets



@numba.njit(parallel=True, fastmath=False)
def _group_mean(values, rows, offsets, zero, single, out):
    for g in numba.prange(len(offsets) - 1):
        total = zero
        compensation = zero
        count = 0
        for k in range(offsets[g], offsets[g + 1]):
            value = values[rows[k]]
            if value == value:
                count += 1
                y = value - compensation
                t = total + y
                compensation = t - total - y
                if compensation != compensation:
                    compensation = zero
                total = t
        if count == 0:
            out[g] = np.nan
        elif single:
            out[g] = total / np.float32(count)
        else:
            out[g] = total / np.float64(count)


def group_mean(values: np.ndarray, rows: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """
    pandas' groupby(...).mean() of float `values` over the groups rows[offsets[g]:offsets[g + 1]]:
    a compensated (Kahan) sum in the values' dtype over the group's rows in the given order,
    skipping NaN, divided by the count; NaN for a group with no values. Groups run in parallel.
    """
    if values.dtype not in (np.float32, np.float64):
        raise TypeError(f"group_mean needs float32 or float64 values, not {values.dtype}")
    out = np.empty(len(offsets) - 1, dtype=values.dtype)
    _group_mean(values, rows, offsets, values.dtype.type(0), values.dtype == np.float32, out)
    return out


@numba.njit(parallel=True, fastmath=False)
def _group_max(values, rows, offsets, out):
    for g in numba.prange(len(offsets) - 1):
        found = False
        best = -np.inf
        for k in range(offsets[g], offsets[g + 1]):
            value = values[rows[k]]
            if value == value:
                found = True
                if value > best:
                    best = value
        out[g] = best if found else np.nan


def group_max(values: np.ndarray, rows: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """pandas' groupby(...).max() of float `values`: the maximum, skipping NaN; NaN for a group with no values."""
    if values.dtype not in (np.float32, np.float64):
        raise TypeError(f"group_max needs float32 or float64 values, not {values.dtype}")
    out = np.empty(len(offsets) - 1, dtype=values.dtype)
    _group_max(values, rows, offsets, out)
    return out


def to_grid(pixels: np.ndarray, group_values: np.ndarray, n_pixels: int, fill=0) -> np.ndarray:
    """A flat image of n_pixels with group_values at pixels and `fill` elsewhere and where a value is NaN
    (reindex onto the grid, then fillna(fill)); in group_values' dtype."""
    grid = np.full(n_pixels, fill, dtype=group_values.dtype)
    grid[pixels] = group_values
    if grid.dtype.kind == "f":
        grid[pixels[np.isnan(group_values)]] = fill
    return grid
