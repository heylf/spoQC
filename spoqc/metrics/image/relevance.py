import numpy as np
import cv2

from ... import helperfuncs

def pixel_relevance(figure_path, xy_intensities, background_intensity, imagedim):
    """
    # TODO adjust description
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