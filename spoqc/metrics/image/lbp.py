from skimage.feature import local_binary_pattern

from ... import helperfuncs

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