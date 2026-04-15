import numpy as np
import cv2

from ... import helperfuncs

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