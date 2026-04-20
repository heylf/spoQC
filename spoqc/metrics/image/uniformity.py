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
def kl_divergence_uniform(x: IntArray) -> np.number:
    """Kullback-Leibler divergence against a uniform distribution"""
    p = occurrence_probability(x)
    # removing zeros is faster than using nansum
    p = p[p > 0]

    # uniform distribution: probability for each element
    # if there are less observations than potential levels truncate
    q = max(1 / x.size, 1 / (np.iinfo(x.dtype).max + 1))

    return ( -(p * np.log(p / q)) ).sum()

def pixel_uniformity(figure_path, img, window_size, imagedim, mode="reflect"):
    timer = helperfuncs.Timer()

    # numba.set_threads(threads)
    radius = (window_size - 1) // 2

    timer.start()
    print(f"... Parallel processing")
    uniformity_image = -sliding_window_padded(kl_divergence_uniform, img, radius, mode=mode)
    timer.stop()
    
    helperfuncs.plot_pixels(
        figure_path,
        uniformity_image,
        imagedim,
        "uniformity",
        "Pixel Uniformity",
        "hot",
        False,
        False,
    )

    return uniformity_image.flatten()