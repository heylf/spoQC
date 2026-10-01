"""Reductions over ragged groups stored as one flat array.

A group layout is given by `offsets` (length n_groups + 1): group g owns the
flat elements offsets[g]:offsets[g + 1].
"""

from typing import Callable

import numpy as np


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
