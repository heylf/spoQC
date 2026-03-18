import numpy as np
import cv2
import plotly.express as px
import dask.array as da

from skimage.feature import local_binary_pattern
from scipy.ndimage import gaussian_filter

from .. import helperfuncs

from spoqc.image_analysis._slidingwindow import (
    sliding_window_padded,
    entropy,
    homogeneity,
    kl_divergence_uniform
)

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
    fig.write_image(f"{figure_path}/histogram_intensity.png", scale=3)

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


def estimate_background_intensity(intensities):
    nbins = 100

    hist, bin_edges = np.histogram(intensities, bins=nbins)

    # TODO so far I assume that bin with the highest density is background because
    # most of the image is background. Is this correct?

    # Find the bin with the highest count
    max_count = np.max(hist)
    max_bin_index = np.argmax(hist)
    max_bin_range = (bin_edges[max_bin_index], bin_edges[max_bin_index + 1])

    background_intensity = np.round(np.mean(max_bin_range), 3)

    return (background_intensity, hist, bin_edges)


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


def pixel_lbp(figure_path, xy_intensities, n_points, radius, imagedim):
    """
    #TODO change description

    Calculate the Local Binary Pattern (LBP) of a grayscale image.

    Args:
        img #TODO
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


def pixel_energy(figure_path, xy_intensities, window_size, imagedim):
    """
    # TODO adjust description
    Calculate the energy in a local neighborhood for each pixel in a grayscale image.

    Args:
        img: #TODO
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


def pixel_relevance(figure_path, xy_intensities, background_intensity, imagedim):
    """
    # TODO adjust description
    Determines if each pixel belongs to a segmented region and visualizes relevance.

    Args:
        image_path (str): Path to the input grayscale image.
        threshold (int): Threshold value for segmentation (0–255).

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