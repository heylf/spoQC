

import numpy as np

from numba import njit
from numpy.typing import DTypeLike

type Array2D[T: np.generic] = np.ndarray[tuple[int, int], np.dtype[T]]
type Array3D[T: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[T]]
type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.integer]]

from spoqc.image_analysis._slidingwindow import sliding_window_padded
from ... import helperfuncs

@njit
def occurrence_probability(
    x: IntArray,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """Relative occurrence of each non-negative integer"""
    # assert x.size > 0
    counts = np.bincount(x.ravel())
    return counts / x.size

@njit
def entropy(x: IntArray) -> np.floating:
    p = occurrence_probability(x)
    # removing zeros is faster than using nansum
    p = p[p > 0]
    return -(p * np.log(p)).sum()

def pixel_entropy(figure_path, img, window_size, imagedim, mode="reflect"):
    timer = helperfuncs.Timer()

    # numba.set_threads(threads)
    radius = (window_size - 1) // 2

    timer.start()
    print("... Parallel processing")
    entropy_image = sliding_window_padded(entropy, img, radius, mode=mode)
    timer.stop()

    print("... Create plot")
    helperfuncs.plot_pixels(
        figure_path,
        entropy_image,
        imagedim,
        "entropy",
        "Pixel Entropy",
        "hot",
        False,
        False,
    )

    return entropy_image.flatten()