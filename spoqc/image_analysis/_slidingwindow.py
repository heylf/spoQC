from collections.abc import Callable
from typing import Any

# from typing import overload
import numpy as np
from numba import njit, prange
from numpy.typing import DTypeLike

type Array2D[T: np.generic] = np.ndarray[tuple[int, int], np.dtype[T]]
type Array3D[T: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[T]]
type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.integer]]


## Utils
@njit
def occurrence_probability(
    x: IntArray,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """Relative occurrence of each non-negative integer"""
    # assert x.size > 0
    counts = np.bincount(x.ravel())
    return counts / x.size


## Metrics
@njit
def entropy(x: IntArray) -> np.floating:
    p = occurrence_probability(x)
    # removing zeros is faster than using nansum
    p = p[p > 0]
    return -(p * np.log(p)).sum()


@njit
def kl_divergence_uniform(x: IntArray) -> np.number:
    """Kullback-Leibler divergence against a uniform distribution"""
    p = occurrence_probability(x)
    # removing zeros is faster than using nansum
    p = p[p > 0]

    # uniform distribution: probability for each element
    # if there are less observations than potential levels truncate
    q = max(1 / x.size, 1 / (np.iinfo(x.dtype).max + 1))

    return ( -(p * np.log(p / q)) ).sum()


@njit
def homogeneity(x: IntArray) -> np.floating:
    counts = np.bincount(x.ravel())
    m, n = x.shape
    v = x[m // 2, n // 2]  # central value

    counts[v] -= 1  # remove central pixel
    p = counts / (x.size - 1)
    # np.arange(len(p)) gets the corresponding value to each probability p
    abs_diff = np.fabs(np.arange(len(p)) - v)
    # If the absolute center pixel difference of the intensieties is large then the homogenity is low and vice versa.
    # If all values are the same then the homogenity is 1.
    return (p / (abs_diff + 1)).sum()


## Sliding window
@njit(parallel=True)
def _sliding_window[T: np.generic](
    func: Callable[[Array2D[T]], Any], x: Array2D[T], r: int, out: Array2D
):
    s = 2 * r + 1
    windows = np.lib.stride_tricks.sliding_window_view(x, (s, s))
    assert windows.shape[:2] == out.shape

    m, n = out.shape
    for i in prange(m):
        for j in prange(n):
            out[i, j] = func(windows[i, j])


def sliding_window[T: np.generic](
    func: Callable[[Array2D[T]], Any],
    x: Array2D[T],
    r: int,
    *,
    dtype: DTypeLike = np.float32,
) -> Array2D:
    """
    Calculate `func` across sliding windows of `x` with radius `r`.

    If the input has shape n x m the output will be n-2r x m-2r.

    Parameters
    ----------
    x : numpy.ndarray[tuple[int, int], numpy.dtype]
        2D array of non-negative integers.
    r : int
        Radius of the sliding window i.e. the size will be 2r+1 in each dimension.
    func : collections.abc.Callable
        Function that takes an 2D-Array view as input (sliding window) and returns a scalar.
        The function should not mutate its input.
    dtype : numpy.typing.DTypeLike
        The dtype of the output array.

    Returns
    -------
    out : numpy.ndarray[tuple[int, int], numpy.dtype]
    """
    out_shape = (x.shape[0] - 2 * r, x.shape[1] - 2 * r)
    out = np.empty(out_shape, dtype=dtype)
    _sliding_window(func, x, r, out)
    return out


def sliding_window_padded[T: np.generic](
    func: Callable[[Array2D[T]], Any],
    x: Array2D[T],
    r: int,
    *,
    dtype: DTypeLike = np.float32,
    mode: str | Callable = "symmetric",
    **kwargs,
) -> Array2D:
    """
    Calculate `func` across sliding windows of `x` with radius `r`.

    Parameters
    ----------
    x : numpy.ndarray[tuple[int, int], numpy.dtype]
        2D array of non-negative integers.
    r : int
        Radius of the sliding window i.e. the size will be 2r+1 in each dimension.
    func : collections.abc.Callable
        Function that takes an 2D-Array view as input (sliding window) and returns a scalar.
        The function should not mutate its input.
    dtype : numpy.typing.DTypeLike
        The dtype of the output array.
    mode : str | collections.abc.Callable
        A valid padding mode for :py:func:`numpy.pad`
    kwargs
        Other keyword arguments are passed to :py:func:`numpy.pad`

    Returns
    -------
    out : numpy.ndarray[tuple[int, int], numpy.dtype]
    """
    x_padded = np.pad(x, pad_width=r, mode=mode, **kwargs)  # type: ignore
    out = sliding_window(func, x_padded, r, dtype=dtype)
    return out
