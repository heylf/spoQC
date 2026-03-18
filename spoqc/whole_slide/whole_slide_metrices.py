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

    print(background_color)

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


# TODO check for thickness issues (use 3D ome.tif)
# 1) x = z-coordinate and y = qv of transcript
# 2) x = z-coordinate and y = intensity of transcript
# 3) plots z axis varbility for each x,y (image density plot)

# TODO check for staining variability across the slide. (uneven illumination)
# intensity density

# TODO that one is garbage. Remove it.
# !! TODO check this this might be wrong !!
# image coorinate system does not direct translate to transcript or ccell coordinates
def staining_qc(sdata: Any, figure_path: str, image_type: str, resolution: str, threads: int) -> None:
    """
    Perform quality control on image data by calculating the distances between nucleus centroids 
    and staining intensity, and generating visualizations.
    Does the centroid of the polygon overlaps with the staining intensity?

    Args:
        sdata (Any): Input spatial data containing morphology and nucleus boundary information.
                     Expected to have 'morphology_focus' with 'scale0' and 'nucleus_boundaries' attributes.
        figure_path (str): Path to save the generated visualization figures.
        threads (int): Number of threads to use for parallel distance calculations.

    Returns:
        None: Saves the visualization figure as an HTML file in the specified path.
    """

    DPI = 300

    stain_x = sdata[image_type][resolution].image.y.values
    stain_y = sdata[image_type][resolution].image.x.values

    # Create a list of tuples. x and y should technically be the same since it is a pixel / intensity point.
    dapi_coordinates = list(zip(stain_x, stain_y))

    nucleus_centroid_x = list(sdata['nucleus_boundaries'].centroid.x)
    nucleus_centroid_y = list(sdata['nucleus_boundaries'].centroid.y)
    nucleus_centroid_coords = list(zip(nucleus_centroid_x, nucleus_centroid_y))

    with ProcessPoolExecutor(max_workers=threads) as executor:
        nucleus_centroid_distances = list(executor.map(functools.partial(min_distance, compare_coords=dapi_coordinates), 
                                        nucleus_centroid_coords)) 

    # Normalize the sum of the distances by the total amount of points.
    norm_sum_nucleus_centroid_distance=max(nucleus_centroid_distances)/len(nucleus_centroid_distances)
    norm_sum_nucleus_centroid_distance=np.round(norm_sum_nucleus_centroid_distance, decimals=5)

    df = pd.DataFrame({'nucleus_centroid_distances': nucleus_centroid_distances,
                       'log10_nucleus_centroid_distances': [np.log10(x) for x in nucleus_centroid_distances]
                       })

    figures = []

    fig = px.violin(df, x='nucleus_centroid_distances', width=800, height=800)

    fig.update_layout(
        title=f"Normalized sum of the distances between staining " + \
        f"intensities and nucleus centroids: {norm_sum_nucleus_centroid_distance}",
    )

    helperfuncs.apply_general_plotly_layout(fig, True)

    figures.append(fig)
    fig.write_image(f'{figure_path}/violinplot_nucleus_centroid_distances.png', scale=int(DPI/100))

    with open(f'{figure_path}/centroid_intensity_distances.html', 'w') as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

