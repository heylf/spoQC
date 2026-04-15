from collections.abc import Callable
from typing import Any

# from typing import overload
import numpy as np
from numba import njit, prange
from numpy.typing import DTypeLike

type Array2D[T: np.generic] = np.ndarray[tuple[int, int], np.dtype[T]]
type Array3D[T: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[T]]
type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.integer]]

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
