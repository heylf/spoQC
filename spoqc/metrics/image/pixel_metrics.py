"""Per-pixel image metrics and the one driver that plots and flattens them.

Every kernel takes a 2D image and returns a tuple of 2D metric images.
`pixel_metric` runs a kernel once, plots each result and returns them flattened.
The window-histogram kernel (entropy, uniformity, homogeneity) is
`image_analysis._slidingwindow.texture_metrics`.
"""

import cv2
import numpy as np
from numba import njit, prange
from scipy.ndimage import gaussian_filter

from ... import helperfuncs


def pixel_metric(kernel, image, figure_path, imagedim, plots, **kernel_args):
    """Run `kernel(image, **kernel_args)` once; plot each 2D result and return them flattened
    (as views where the result is C-contiguous, so no second full-image copy is made).

    plots: one (suffix, title, legend_dict) per kernel result; legend_dict None means no legend.
    """
    results = kernel(image, **kernel_args)
    flat = []
    for result, (suffix, title, legend_dict) in zip(results, plots, strict=True):
        helperfuncs.plot_pixels(
            figure_path,
            result,
            imagedim,
            suffix,
            title,
            "hot",
            False,
            legend_dict is not None,
            legend_dict=legend_dict,
        )
        flat.append(result.ravel())
    return flat


def signal_noise_ratio(xy_intensities, background_intensity):
    """Log2 signal-noise ratio: is the pixel noise or true positive?"""
    return (np.log2((xy_intensities + 1) / background_intensity),)


@njit(parallel=True)
def _lbp_uniform(image, rp, cp, output):
    """skimage 0.26 _texture.pyx _local_binary_pattern, method 'U', rows in parallel.

    Same arithmetic in the same order as skimage's bilinear_interpolation (interpolation.pxd,
    mode 'C', cval 0) at GLOBAL (r + rp[i], c + cp[i]), so the tie test
    texture[i] - image[r, c] >= 0 decides exactly as it does there.
    """
    rows, cols = image.shape
    P = rp.shape[0]
    for r in prange(rows):
        signed_texture = np.empty(P, dtype=np.int8)
        for c in range(cols):
            center = image[r, c]
            for i in range(P):
                rr = r + rp[i]
                cc = c + cp[i]
                minr = np.int64(np.floor(rr))
                minc = np.int64(np.floor(cc))
                maxr = np.int64(np.ceil(rr))
                maxc = np.int64(np.ceil(cc))
                dr = rr - minr
                dc = cc - minc
                top_left = image[minr, minc] if 0 <= minr < rows and 0 <= minc < cols else 0.0
                top_right = image[minr, maxc] if 0 <= minr < rows and 0 <= maxc < cols else 0.0
                bottom_left = image[maxr, minc] if 0 <= maxr < rows and 0 <= minc < cols else 0.0
                bottom_right = image[maxr, maxc] if 0 <= maxr < rows and 0 <= maxc < cols else 0.0
                top = (1 - dc) * top_left + dc * top_right
                bottom = (1 - dc) * bottom_left + dc * bottom_right
                texture = (1 - dr) * top + dr * bottom
                signed_texture[i] = 1 if texture - center >= 0 else 0
            changes = 0
            for i in range(P - 1):
                changes += (signed_texture[i] - signed_texture[i + 1]) != 0
            lbp = 0.0
            if changes <= 2:
                for i in range(P):
                    lbp += signed_texture[i]
            else:
                lbp = P + 1
            output[r, c] = lbp


def lbp(xy_intensities, n_points, radius):
    """Local binary pattern ("uniform"), bit-identical to skimage.feature.local_binary_pattern.

    Threads: numba's pool, set from CONST.THREADS in cli.py.
    """
    # skimage's circle points: local position of texture elements, rounded to 5 decimals
    rp = np.round(-radius * np.sin(2 * np.pi * np.arange(n_points, dtype=np.float64) / n_points), 5)
    cp = np.round(radius * np.cos(2 * np.pi * np.arange(n_points, dtype=np.float64) / n_points), 5)
    image = np.ascontiguousarray(xy_intensities, dtype=np.float64)
    output = np.zeros(image.shape, dtype=np.float64)
    _lbp_uniform(image, rp, cp, output)
    return (output,)


def edge_strength(xy_intensities):
    """Log10 Sobel gradient magnitude."""
    # This I have to do else it breaks.
    xy_intensities = xy_intensities.astype(np.uint16)
    grad_x = cv2.Sobel(xy_intensities, cv2.CV_64F, 1, 0, ksize=3)  # Gradient along x
    grad_y = cv2.Sobel(xy_intensities, cv2.CV_64F, 0, 1, ksize=3)  # Gradient along y
    return (np.log10(np.sqrt(grad_x**2 + grad_y**2) + 1),)


def energy(xy_intensities, window_size):
    """Log10 local energy: gaussian-weighted mean of squared intensities (radius window_size)."""
    squared_image = xy_intensities.astype(np.float64) ** 2
    energy_image = gaussian_filter(squared_image, sigma=1, radius=window_size)
    return (np.log10(energy_image + 1),)


def relevance(xy_intensities, background_intensity):
    """1 where the blurred pixel is above the Otsu threshold, else 0."""
    # This I have to do else it breaks.
    xy_intensities = xy_intensities.astype(np.uint16)
    blur = cv2.GaussianBlur(xy_intensities, (5, 5), 0)
    _, segmented_image = cv2.threshold(
        blur,
        background_intensity,
        xy_intensities.max(),
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return ((segmented_image > 0).astype(np.uint8),)

