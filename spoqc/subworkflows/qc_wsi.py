import cv2  # -> opencv-python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import functools

from typing import List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor

from .. import helperfuncs

def measure_stripe_thickness_and_black_area(image_path: str, 
                                            background_color: np.ndarray[3, np.dtype[np.int_]],
                                            output_path: str) -> float:
    """
    Function to measure stripe thickness and black area between contour, i.e., 
    it can be used to detect thickness/thinness of areas/domains in an image.

    Assumptions:
    1) There is a background color of the image (default = white)s.
    2) Edges in the images are the domain contours.
    3) Area (black) that is not background and not edges are domains.
    The thickness score is definite as:
    score 1.0 = infinite thickness (one domain that covers the entire image)
    score 0.0 = infinite thinness (infinite domains, edges cover the entire image)

    Example:
        >>> measure_stripe_thickness_and_black_area('./../image.jpg')
        4.5

    Args:
        image_path (str): Input path of the figure that you want to analyse.
        brackground_color (RGB): RBG color of the background.
        output_path (str): Where to save the image with the domain thickness score.

    Return
        float: Normalized adjusted black area between edges (thickness).
    
    """

    lower_bound_background = background_color.copy() - 20
    upper_bound_background = background_color.copy() + 20

    # Correct values.
    for i in range(0,3):
        if lower_bound_background[i] < 0:
            lower_bound_background[i] = 0
        if lower_bound_background[i] > 255:
            lower_bound_background[i] = 255
        if upper_bound_background[i] < 0:
            upper_bound_background[i] = 0
        if upper_bound_background[i] > 255:
            upper_bound_background[i] = 255

    # Read the image
    image = cv2.imread(image_path)
    # Turn background color to white
    mask = cv2.inRange(image, lower_bound_background, upper_bound_background)
    image[mask != 0] = [255, 255, 255]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    # Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and odd.
    # Increase the size for more blurr and thus later to focus on general stuctures in the image.
    ksize = (23, 23)
    xstd = 0 # Standard deviation of Gaussian kernal in X direction.
    blurred = cv2.GaussianBlur(gray, ksize, xstd)
    # Edge detection using Canny
    # The function finds edges in the input image and marks them in the output map edges using the Canny algorithm. 
    # The smallest value between threshold1 and threshold2 is used for edge linking. 
    # The largest value is used to find initial segments of strong edges.
    t1 = 10  # Decrease to connect more edges.
    t2 = 20  # Decrease to find more detailed edges.
    edges = cv2.Canny(blurred, t1, t2)
       
    # Calculate the total area of the image
    total_image_area = gray.shape[0] * gray.shape[1]
    
    # Calculate the total area covered by edges (non-black areas)
    edge_area = np.sum(edges > 0)
    
    # Calculate the black area (domain area)
    black_area = total_image_area - edge_area

    # Calculate the white area in the original image (background area).
    # Default (background) color is assumed to be white.
    white_area = np.sum(gray > 250)  # Assuming white pixels are those with a value greater than 250
    
    # Adjust the black area by subtracting the white area and normalize by total image area.
    norm_adjusted_black_area = (black_area - white_area) / total_image_area
    print("Total white area in the original image:", white_area)
    
    # Plot the original, grayscale, blurred, and edge-detected images
    plt.figure(figsize=(12, 3))
    
    # Adding the main title for the entire figure
    plt.suptitle(f'Domain thickness: {np.round(norm_adjusted_black_area, 5)}', fontsize=16)

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    
    plt.subplot(1, 4, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Image')
    
    plt.subplot(1, 4, 3)
    plt.imshow(blurred, cmap='gray')
    plt.title('Blurred Image')
    
    plt.subplot(1, 4, 4)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edges')
    
    plt.savefig(f'{output_path}/domain_thickness_score.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_path}/domain_thickness_score.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    #return thicknesses, adjusted_black_area
    return norm_adjusted_black_area


def min_distance(coords: Tuple[float, float], compare_coords: List[Tuple[float, float]]) -> float:
    """
    Calculate the minimum Euclidean distance between a given coordinate and a list of coordinates.

    Args:
        coords (Tuple[float, float]): The reference coordinate as a tuple (x, y).
        compare_coords (List[Tuple[float, float]]): A list of coordinates to compare against.

    Returns:
        float: The minimum Euclidean distance between `coords` and the coordinates in `compare_coords`.
    """
    return np.min([helperfuncs.euclidean_distance(x, coords) for x in compare_coords])

def generate_input(sdata, figure_path, CONST):
    ax = sdata.pl.render_images(CONST.IMAGE_TYPE).pl.show(
        title='',
        frameon=False,
        return_ax=True,
        pad_extent=0,
        dpi=300,
        show=False
    )
    ax.axis('off')
    ax.invert_yaxis()
    plt.savefig(f'{figure_path}/input_domain_thickness_analysis.png', bbox_inches='tight')
    plt.savefig(f'{figure_path}/input_domain_thickness_analysis.pdf', bbox_inches='tight')
    plt.close()