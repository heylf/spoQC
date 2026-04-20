import numpy as np

from numba import njit
from numpy.typing import DTypeLike

type Array2D[T: np.generic] = np.ndarray[tuple[int, int], np.dtype[T]]
type Array3D[T: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[T]]
type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.integer]]

from spoqc.image_analysis._slidingwindow import sliding_window_padded
from ... import helperfuncs

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

def pixel_homogeneity(figure_path, img, imagedim, window_size, mode="reflect"):
    timer = helperfuncs.Timer()

    # numba.set_threads(threads)
    radius = (window_size - 1) // 2

    timer.start()
    print("... Parallel processing")
    homogeneity_image = sliding_window_padded(homogeneity, img, radius, mode=mode)
    timer.stop()

    helperfuncs.plot_pixels(
        figure_path,
        homogeneity_image,
        imagedim,
        "homogeneity",
        "Pixel Homogeneity",
        "hot",
        False,
        False,
    )

    return homogeneity_image.flatten()