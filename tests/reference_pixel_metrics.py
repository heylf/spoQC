"""Verbatim per-pixel metric code before this change, for differential tests.

Each section is the file named in its banner, copied unchanged except that
"from ... import helperfuncs" becomes "from spoqc import helperfuncs" and the
sliding_window_padded import is dropped (the verbatim copy is in this file).
start_image_struc_analyis is verbatim; its module-level imports are replaced by
a `metrics` shim that points at the verbatim wrappers in this file.
"""
# ruff: noqa
# ===== spoqc/image_analysis/_slidingwindow.py
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


# ===== spoqc/metrics/image/entropy.py


import numpy as np

from numba import njit
from numpy.typing import DTypeLike

type Array2D[T: np.generic] = np.ndarray[tuple[int, int], np.dtype[T]]
type Array3D[T: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[T]]
type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.integer]]

from spoqc import helperfuncs

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

# ===== spoqc/metrics/image/uniformity.py
import numpy as np

from numba import njit
from numpy.typing import DTypeLike

type Array2D[T: np.generic] = np.ndarray[tuple[int, int], np.dtype[T]]
type Array3D[T: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[T]]
type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.integer]]

from spoqc import helperfuncs

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

# ===== spoqc/metrics/image/homogenity.py
import numpy as np

from numba import njit
from numpy.typing import DTypeLike

type Array2D[T: np.generic] = np.ndarray[tuple[int, int], np.dtype[T]]
type Array3D[T: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[T]]
type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.integer]]

from spoqc import helperfuncs

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

# ===== spoqc/metrics/image/energy.py
import numpy as np

from scipy.ndimage import gaussian_filter

from spoqc import helperfuncs

def pixel_energy(figure_path, xy_intensities, window_size, imagedim):
    """
    Calculate the energy in a local neighborhood for each pixel in a grayscale image.

    Args:
        img: Provided image data.
        window_size (int): Size of the sliding window (must be odd).

    Returns:
        energy_image (ndarray): Image of energy values for each pixel.
    """
    
    # Square the pixel intensities
    squared_image = xy_intensities.astype(np.float64) ** 2

    # Compute the local energy using a sliding window (mean of squared values)
    energy_image = gaussian_filter(squared_image, sigma=1, radius=window_size)

    log10_energy_image = np.log10(energy_image + 1)

    helperfuncs.plot_pixels(
        figure_path,
        log10_energy_image,
        imagedim,
        'energy', 
        'Log10 Pixel Energy', 
        'hot',
        False,
        False
    )

    return log10_energy_image.flatten()

# ===== spoqc/metrics/image/edge_strength.py
import numpy as np
import cv2

from spoqc import helperfuncs

def pixel_edge_strength(figure_path, xy_intensities, imagedim):

    # This I have to do else it breaks.
    xy_intensities = xy_intensities.astype(np.uint16)
    
    # Compute gradients in the x and y directions using Sobel filters
    grad_x = cv2.Sobel(xy_intensities, cv2.CV_64F, 1, 0, ksize=3)  # Gradient along x
    grad_y = cv2.Sobel(xy_intensities, cv2.CV_64F, 0, 1, ksize=3)  # Gradient along y

    # Compute the magnitude of the gradient (edge strength)
    # edge_strength = np.sqrt(grad_x**2 + grad_y**2)
    log10_edge_strength_image = np.log10( ( np.sqrt(grad_x**2 + grad_y**2) + 1 ) )

    helperfuncs.plot_pixels(
        figure_path,
        log10_edge_strength_image,
        imagedim,
        'edge_strength', 
        'Edge Strength', 
        'hot',
        False,
        False
    )

    return log10_edge_strength_image.flatten()

# ===== spoqc/metrics/image/relevance.py
import numpy as np
import cv2

from spoqc import helperfuncs

def pixel_relevance(figure_path, xy_intensities, background_intensity, imagedim):
    """
    Determines if each pixel belongs to a segmented region and visualizes relevance.

    Args:
        image_path (str): Path to the input grayscale image.
        threshold (int): Threshold value for segmentation (0-255).

    Returns:
        segmented_image (ndarray): Binary segmented image (0 or 255).
        relevance_map (ndarray): Map showing relevance of each pixel to the segmented region.
    """
    timer = helperfuncs.Timer()

    # This I have to do else it breaks.
    xy_intensities = xy_intensities.astype(np.uint16)

    # Apply blur
    print("[NOTE] Gaussian Blur")
    timer.start()
    blur = cv2.GaussianBlur(xy_intensities,(5,5),0)
    timer.stop()

    # Apply binary thresholding for segmentation
    print("[NOTE] Otsu thresholding")
    timer.start()
    _, segmented_image = cv2.threshold(blur, background_intensity, 
                                       max(xy_intensities.flatten()), cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    timer.stop()

    # Compute relevance map: pixels in the segmented region have value 1, others have 0
    relevance_map_image = (segmented_image > 0).astype(np.uint8)

    print("[NOTE] Plotting")
    timer.start()
    helperfuncs.plot_pixels(
        figure_path,
        relevance_map_image,
        imagedim,
        'relevance', 
        'Pixel Relevance', 
        'hot',
        False,
        True,
        legend_dict={"high rel.": "#FFFFFF", "low rel.": "#000000"}
    )
    timer.stop()

    return relevance_map_image.flatten()

# ===== spoqc/metrics/image/lbp.py
from skimage.feature import local_binary_pattern

from spoqc import helperfuncs

def pixel_lbp(figure_path, xy_intensities, n_points, radius, imagedim):
    """
    Calculate the Local Binary Pattern (LBP) of a grayscale image.

    Args:
        img Provided image data.
        radius (int): Radius of the circle for LBP computation.
        n_points (int): Number of points in the circular neighborhood.

    Returns:
        lbp_image (ndarray): Image of LBP values.
    """
    # Load the image in grayscale

    timer = helperfuncs.Timer()

    # Calculate LBP using skimage's local_binary_pattern
    print("[NOTE] LBP calculation")
    timer.start()
    lbp_image = local_binary_pattern(xy_intensities, n_points, radius, method="uniform")
    timer.stop()

    print("[NOTE] Plotting")
    timer.start()
    helperfuncs.plot_pixels(
        figure_path,
        lbp_image,
        imagedim,
        'lbp', 
        'Local Binary Pattern (LBP)', 
        'hot',
        False,
        False
    )
    timer.stop()

    return lbp_image.flatten()

# ===== spoqc/metrics/image/utility.py
import numpy as np
import plotly.express as px
import dask.array as da

from spoqc import helperfuncs
from spoqc.core.figures import save_figure

def turn_into_uint8(arr):
    # normalize to 0–1 if needed
    if int(arr.min()) != 0 or int(arr.max()) != 1:
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    # scale to 0–255 and convert to uint8
    uint8_arr = (arr * 255).astype(np.uint8)
    return uint8_arr

def pixel_intensity_qc(figure_path, intensities, background_intensity, hist, bin_edges, dim_x, dim_y, imagedim):

    timer = helperfuncs.Timer()

    figures = []

    # When you plot a histogram via plotly, it stores all the orginal data in the json file 
    # and makes the bins and counts on the javascript side. 
    # Thus the plot get quite large.
    # Use therefore the precomupted histogram data from numpy.
    bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    print("[NOTE] Barplot")
    timer.start()
    fig = px.bar(x=bins, y=hist, labels={'x':'intensity', 'y':'count'})
    fig.update_layout(
        title=f"Total distribution intensity with backkground intensity {background_intensity}"
    )
    timer.stop()
    helperfuncs.apply_general_plotly_layout(fig, True)
    figures.append(fig)
    save_figure(fig, f"{figure_path}/histogram_intensity.png", f"{figure_path}/histogram_intensity.pdf", scale=3)

    with open(f'{figure_path}/histogram_intensity.html', 'w') as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    
    signal_noise_ratio_log2fc = np.log2( (intensities + 1) / background_intensity )

    helperfuncs.plot_pixels(
        figure_path,
        np.array(signal_noise_ratio_log2fc).reshape(dim_x, dim_y),
        imagedim,
        'snr', 
        'Log2 Signal-Noise-Ratio', 
        'hot',
        False,
        False
    )
    
    return signal_noise_ratio_log2fc


def estimate_background_intensity_dask(sdata, image_type, resolution, staining, nbins=100, range_=None):
    """
    nbins: number of histogram bins
    range_: optional (min, max); if None, computed lazily with dask
    """
    intensities = sdata[image_type][resolution].image.data[int(staining)]
    intensities.ravel()

    if not hasattr(intensities, "chunks"):
        raise TypeError("Pass a dask.array for the Dask implementation.")

    # Compute min/max lazily if not supplied (cheap: just scalars)
    if range_ is None:
        vmin = da.nanmin(intensities)
        vmax = da.nanmax(intensities)
        vmin, vmax = da.compute(vmin, vmax)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            raise ValueError("Non-finite min/max encountered.")
        if vmin == vmax:
            vmax = vmin + 1.0
        range_ = (float(vmin), float(vmax))

    # Dask builds the histogram in a reduction; result is tiny (nbins) -> safe to .compute()
    hist, bin_edges = da.histogram(intensities, bins=nbins, range=range_)
    hist, bin_edges = da.compute(hist, bin_edges)

    max_bin_idx = int(np.argmax(hist))
    # center of the winning bin
    background = np.round((bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) * 0.5, 3)
    return background, hist, bin_edges

# ===== spoqc/image_analysis/structure_analysis.py (its 'metrics' resolves to the shim below)
import os
import sys
from types import SimpleNamespace

import spoqc.metrics.transcript_density

metrics = SimpleNamespace(
    image=SimpleNamespace(
        utility=SimpleNamespace(
            pixel_intensity_qc=pixel_intensity_qc,
            estimate_background_intensity_dask=estimate_background_intensity_dask,
        ),
        lbp=SimpleNamespace(pixel_lbp=pixel_lbp),
        edge_strength=SimpleNamespace(pixel_edge_strength=pixel_edge_strength),
        energy=SimpleNamespace(pixel_energy=pixel_energy),
        relevance=SimpleNamespace(pixel_relevance=pixel_relevance),
        entropy=SimpleNamespace(pixel_entropy=pixel_entropy),
        uniformity=SimpleNamespace(pixel_uniformity=pixel_uniformity),
        homogenity=SimpleNamespace(pixel_homogeneity=pixel_homogeneity),
    ),
    transcript_density=spoqc.metrics.transcript_density,
)

def start_image_struc_analyis(
        sdata,
        figure_path_base,
        spoqc_tmp_folder,
        modality,
        image_type,
        resolution,
        imagedim,
        dim_x,
        dim_y,
        overwrite,
        *,
        staining=None
):

    timer = helperfuncs.Timer()

    print(f'[NOTE] start structural analysis with {image_type} and {resolution} for {modality}')
    
    tmp_suffix = modality
    if ( staining ):
        spoqc_tmp_folder = f'{spoqc_tmp_folder}/metrices/{modality}/{staining}/'
        figure_path = f'{figure_path_base}/{modality}/{modality}_metrices/{staining}/'
        tmp_suffix = f'{modality}_{staining}'
    else:
        figure_path = f'{figure_path_base}/{modality}/{modality}_metrices/'
        spoqc_tmp_folder = f'{spoqc_tmp_folder}/metrices/{modality}'

    xy_intensities = None
    intensities = None
    # Quantized copy used only by the entropy/uniformity/homogeneity sliding-window metrics below:
    # each window's np.bincount allocates max(window_values)+1 elements, so raw uint16 image intensities 
    # (full 0-65535 dynamic range) make every window allocation far larger than the handful of values it summarizes.
    # Transcript density values (hqtr) are already small integers and don't need this.
    texture_intensities = None
    if ( modality == 'hqtr' ):
        # Intensities already flipped
        intensities = metrics.transcript_density.transcript_density_image.generate_transcript_density_image(
            sdata,
            figure_path,
            imagedim,
            image_type,
            resolution
        )
        xy_intensities = intensities.reshape(dim_x, dim_y)

        # Plot transcript point plot
        helperfuncs.plot_scatter_by_category(
            sdata.points['transcripts'].compute(),
            None, 
            figure_path, 
            'transcript_points',
            'transcript_points',
            None,
            pointsize=0.5
        )

        helperfuncs.nparr_to_parquet(intensities, 'transcript_density', spoqc_tmp_folder, tmp_suffix)
        texture_intensities = xy_intensities
    else:
        xy_intensities = sdata[image_type][resolution].image.values[int(staining)]
        xy_intensities = np.flipud(xy_intensities)
        intensities = xy_intensities.flatten()

        n_bins = 256
        texture_intensities = np.floor(
            (xy_intensities.astype(np.float64) - xy_intensities.min())
            / max(xy_intensities.max() - xy_intensities.min(), 1) * (n_bins - 1)
        ).astype(np.uint8)

    # Plot intensities
    name = 'input'
    if ( modality == 'hqtr' ):
        name = f'{name}_transcript_densities'
    elif ( modality == 'hqpr' ):
        name = f'{name}_pixel_intensities'
    else:
        sys.exit('[ERROR] Modality not supported')

    helperfuncs.plot_pixels(
        figure_path,
        np.log10(xy_intensities + 1),
        imagedim,
        name,
        name,
        'gray',
        False,
        False
    )

    steps = []

    for step in ['intensity', 'edge_strength', 'lbp', 'energy', 
                 'relevance', 'homogenity', 'entropy', 'uniformity',
                 'cluster']:
        if ( overwrite or not os.path.exists(f"{spoqc_tmp_folder}/{step}_output_{modality}.parquet") ):
            if ( not ( step == 'intensity' and modality == 'hqtr' ) ):
                steps.append(step)
                print(f"[NOTE] {step} will be performed")

    background_intensity = 0.0  # this is valid for hqtr
    hist = None
    bin_edges = None

    if ( modality == 'hqpr' ):
        background_intensity, hist, bin_edges = metrics.image.utility.estimate_background_intensity_dask(
            sdata,
            image_type,
            resolution,
            staining
        )

    step = 'intensity'
    if ( step in steps and modality == 'hqpr' ):
        # General Singal/Noise ratio. Is the pixel noise or true positive?
        # Not valid for hqtr because the background is a constant of 0.0.
        print('[NOTE] Evaluate pixel intensity')
        timer.start()
        signal_noise_ratio_log2fc = metrics.image.utility.pixel_intensity_qc(figure_path, intensities, 
                                                                 background_intensity, hist, bin_edges, 
                                                                 dim_x, dim_y, imagedim)
        timer.stop()
        helperfuncs.nparr_to_parquet(signal_noise_ratio_log2fc, step, spoqc_tmp_folder, tmp_suffix)

    step = 'lbp'
    if ( step in steps ):
        # Pixel pattern information. Does a pixel live in a specific pattern?
        # Mostly useful to identify if windows have specific patterns you want to cluster.
        print('[NOTE] Investigate local binary patterns')
        timer.start()
        lbp = metrics.image.lbp.pixel_lbp(figure_path, xy_intensities, 100, 3, imagedim)
        timer.stop()
        helperfuncs.nparr_to_parquet(lbp, step, spoqc_tmp_folder, tmp_suffix)

    #######################
    ###### Structure ######
    #######################

    step = 'edge_strength'
    if ( step in steps ):
        # EDGE STRENGTH - Edge detection
        print('[NOTE] Calculate edge strength')
        timer.start()
        edge_strength = metrics.image.edge_strength.pixel_edge_strength(figure_path, xy_intensities, imagedim)
        timer.stop()
        helperfuncs.nparr_to_parquet(edge_strength, step, spoqc_tmp_folder, tmp_suffix)

    step = 'energy'
    if ( step in steps ):
        # How much inforamtion has a pixel?
        print('[NOTE] Calculate pixel energy')
        timer.start()
        pixel_energy = metrics.image.energy.pixel_energy(figure_path, xy_intensities, 5, imagedim)
        timer.stop()
        helperfuncs.nparr_to_parquet(pixel_energy, step, spoqc_tmp_folder, tmp_suffix)

    step = 'relevance'
    if ( step in steps ):
        # Just check which pixel are have intensities bigger than background.
        print('[NOTE] Investigate pixel relevance')
        timer.start()
        pixel_relevance = metrics.image.relevance.pixel_relevance(figure_path, xy_intensities, 
                                                                        background_intensity, imagedim)
        timer.stop()
        helperfuncs.nparr_to_parquet(pixel_relevance, step, spoqc_tmp_folder, tmp_suffix)

    step = "entropy"
    if step in steps:
        # General Pixel Information. How much information contributes a pixel?
        # Computational expensive.
        print("[NOTE] Calculate pixel entropy")
        timer.start()
        pixel_entropy = metrics.image.entropy.pixel_entropy(figure_path, texture_intensities, 5, imagedim)
        timer.stop()
        helperfuncs.nparr_to_parquet(pixel_entropy, step, spoqc_tmp_folder, tmp_suffix)

    ############################
    ###### Anti structure ######
    ############################

    step = "uniformity"
    if step in steps:
        # Is the pixel in a noisy region?
        print("[NOTE] Calculate pixel uniformity with")
        timer.start()
        pixel_uniformity = metrics.image.uniformity.pixel_uniformity(figure_path, texture_intensities, 5, imagedim)
        timer.stop()
        helperfuncs.nparr_to_parquet(pixel_uniformity, step, spoqc_tmp_folder, tmp_suffix)

    step = "homogenity"
    if step in steps:
        # How much does a pixel disrupt the local neighbourhood?
        # Or how homogenous is the pixel around the region?
        # Computational expensive.
        print("[NOTE] Calculate pixel homogeneity")
        timer.start()
        pixel_homogeneity = metrics.image.homogenity.pixel_homogeneity(figure_path, texture_intensities, imagedim, 5)
        timer.stop()
        helperfuncs.nparr_to_parquet(pixel_homogeneity, step, spoqc_tmp_folder, tmp_suffix)
