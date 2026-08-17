import numpy as np

from scipy.ndimage import gaussian_filter

from ... import helperfuncs

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