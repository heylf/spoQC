import pandas as pd
import numpy as np
import plotly.express as px
import concurrent.futures
import geopandas as gpd

from scipy.spatial import cKDTree
from typing import Tuple, List

from .. import helperfuncs

def is_convex(polygon: List[Tuple[float, float]]) -> Tuple[bool, float]:
    """
    Determines if a polygon is convex and measures its convexity.
    
    Parameters:
        polygon (List[Tuple[float, float]]): List of (x, y) vertices defining the polygon.
        
    Returns:
        Tuple[bool, float]: 
            - A boolean indicating whether the polygon is convex.
            - A float representing the convexity metric (range [0, 1]; 1 is fully convex).
    """
    n = len(polygon)
    if n < 3:
        raise ValueError("A polygon must have at least three vertices.")
    
    cross_products = []
    for i in range(n):
        # Get three consecutive points
        p1 = np.array(polygon[i])
        p2 = np.array(polygon[(i + 1) % n])
        p3 = np.array(polygon[(i + 2) % n])
        
        # Compute vectors
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Compute the cross product of vectors
        cross_product = np.cross(v1, v2)
        cross_products.append(cross_product)
    
    # Check if all cross products have the same sign
    all_positive = all(cp > 0 for cp in cross_products)
    all_negative = all(cp < 0 for cp in cross_products)
    
    is_polygon_convex = all_positive or all_negative

    # Measure convexity as the ratio of consistent angles
    neg = 0
    pos = 0
    if all_positive or all_negative:
        pos = sum(cp != 0 for cp in cross_products)
    else:
        # Sum up the boolean vectore (i.e., sum up all Ture)
        neg = sum(cp < 0 for cp in cross_products)
        pos = sum(cp > 0 for cp in cross_products)

    # Just take the maximum amount of consistent angles
    convexity_metric = np.max([neg,pos]) / len(cross_products)
    
    return is_polygon_convex, convexity_metric


# Function to assign nulcei to cell
def find_overlapping_nuclei(cells: gpd.GeoDataFrame, nucleus: gpd.GeoDataFrame):
    print("[NOTE] Find overlapping nuceli for cells")
    timer = helperfuncs.Timer()
    timer.start()
    overlaps = []
    nucleus_centroids = nucleus.geometry.centroid
    for cell in cells.geometry:
        overlapping_indices = nucleus[nucleus_centroids.geometry.intersects(cell)].index.tolist()
        overlaps.append(overlapping_indices)
    timer.stop()
    return overlaps


def convexityqc(sdata, figure_path):

    timer = helperfuncs.Timer()

    # Convexity calculation for cell polygon
    print("[NOTE] Calculate convexity for cells")
    timer.start()
    cell_convexity_metric_list = []
    for poly in sdata['cell_boundaries']['geometry']:
        cell_is_polygon_convex, cell_convexity_metric = is_convex(list(poly.exterior.coords))
        cell_convexity_metric_list.append(cell_convexity_metric)
    timer.stop()

    sdata['table'].obs['convexity_cell'] = [True if x > 0.5 else False for x in cell_convexity_metric_list]
    sdata['table'].obs['convexity_metric_cell'] = cell_convexity_metric_list

    # Find nuceli cell overlaps
    nulcei_of_the_cell = find_overlapping_nuclei(sdata['cell_boundaries'], sdata['nucleus_boundaries'])
    sdata['table'].obs['nuclei_idxs'] = nulcei_of_the_cell

    # Convexity calcualteion for nuclei associated with cell
    print("[NOTE] Calculate convexity for cells")
    timer.start()
    nulcei_convexity_metric_list = []
    min_convexity_metric_list = []
    nuclei_idxs = sdata['table'].obs['nuclei_idxs']
    for cell_nuclei in nuclei_idxs:
            if ( len(cell_nuclei) != 0 ):
                # Since we might have more then one nuclei in a cell we take the mean.
                convexities = []
                for nuceuls_idx in cell_nuclei:
                    nuceuls_poly = sdata['nucleus_boundaries']['geometry'].loc[nuceuls_idx]
                    is_polygon_convex, convexity_metric = is_convex(list(nuceuls_poly.exterior.coords))
                    convexities.append(convexity_metric)
                nulcei_convexity_metric_list.append(np.mean(convexities))
                min_convexity_metric_list.append(np.min(convexities))
            else:
                nulcei_convexity_metric_list.append(0)
                min_convexity_metric_list.append(0)
    timer.stop()

    sdata['table'].obs['convexity_mean_nuceli'] = nulcei_convexity_metric_list
    sdata['table'].obs['convexity_min_nuceli'] = min_convexity_metric_list
    sdata['table'].obs['convexity_nuclei'] = [True if x > 0.5 else False for x in min_convexity_metric_list]

    plotcats = [['convexity_cell', 'convexity_metric_cell'], 
                ['convexity_nuclei', 'convexity_mean_nuceli', 'convexity_metric_cell']]

    figures = []
    for i, cat in enumerate(plotcats):

        helperfuncs.plot_scatter(sdata['table'], figure_path, cat[0], None, 
                                cat[0], ['black', 'lightblue'], None)
        helperfuncs.plot_scatter_density(sdata['table'], figure_path, f'{cat[0]}_{cat[1]}', 
                                         cat[0], cat[1], ['black', 'lightblue'], None)
        
        if ( i == 1 ):
            helperfuncs.plot_scatter_density(sdata['table'], figure_path, f'{cat[0]}_{cat[2]}', 
                                             cat[0], cat[2], ['black', 'lightblue'], None)
        
        fig = px.histogram(sdata['table'].obs, x=cat[1], nbins=100, width=800, height=800)
        fig.update_layout(
            title=f"Total distribution {cat[1]} for cells of all samples"
        )
        helperfuncs.apply_general_plotly_layout(fig, True)
        figures.append(fig)
        fig.write_image(f"{figure_path}/histogram_{cat[1]}.png", scale=3)

    with open(f'{figure_path}/convexity.html', 'w') as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


def compute_border_score_for_point(point_idx, points, rotation_matrices, radius, tree):
    x1, y1 = points[point_idx]

    # Find nearby points
    indices = tree.query_ball_point([x1, y1], r=radius)
    if not indices:
        return 0, point_idx  # No neighbors found.

    relevant_points = points[indices]
    diffs = relevant_points - np.array([x1, y1])  # (N,2)

    scores = []

    for rotation_matrix in rotation_matrices:
        # Rotate points
        rotated = diffs @ rotation_matrix  # much faster

        # New coords for cell to look at
        x_coords = rotated[:, 0]

        # Get all positive distances. 
        # Add one to solve issue with inf.
        num_left = np.count_nonzero(x_coords > 0) + 1

        # Get all negative distances. 
        # Add one to solve issue with inf. Both sites have to be treated equally.
        num_right = np.count_nonzero(x_coords < 0) + 1

        # I am not interested in the direction just the magnitude.
        score = abs(np.log2(num_left / num_right))
        scores.append(score)

    return max(scores), point_idx


def get_border_scores_optimized(df, radius, step, threads):
    points = df[['x', 'y']].values
    tree = cKDTree(points)

    # Precompute rotation matrices only once
    angles = np.radians(np.arange(0, 360, step))
    rotation_matrices = np.stack([
        np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        for a in angles
    ])

    # Move args outside to avoid heavy pickling
    compute_args = (points, rotation_matrices, radius, tree)

    def wrapper(idx):
        return compute_border_score_for_point(idx, *compute_args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        results = list(executor.map(wrapper, range(len(points))))

    return np.array(results)


def define_border_cells(sdata: dict, figure_path: str, thresh: float,
                        radius: float, stepsize: float, threads: int) -> None:
    
    """
    Identifies and annotates border cells in spatial transcriptomics data.
    
    Args:
        sdata (dict): A dictionary containing spatial transcriptomics data. 
                      Assumes 'table' key includes an `obsm` attribute with spatial coordinates.
        figure_path (str): Path to save the visualization of border cells.
        thresh (float): Threshold for classifying cells as border cells based on scores.
        radius (float): Radius used for calculating border scores.
        stepsize (float): Step size used in the border score calculation.
        threads (int): Number of threads to use for parallel processing.

    Returns:
        None: Modifies the `sdata` object in-place by adding:
              - `border_cell`: A boolean column in `obs` indicating whether each cell is a border cell.
              - `border_scores`: A column in `obs` with the border scores for each cell.
              Additionally, saves a scatter plot visualization to the specified path.

    Notes:
        - The border scores are computed using `get_border_scores_optimized`.
        - A scatter plot of the border cells is generated using `helperfuncs.plot_scatter`.
    """

    df = pd.DataFrame({
        'x': sdata['table'].obsm['spatial'][:, 0],
        'y': sdata['table'].obsm['spatial'][:, 1],
    })

    # Get scores
    border_scores_indices = get_border_scores_optimized(df, radius, stepsize, threads)

    # Store scores
    indices = border_scores_indices[:, 1].astype(int)
    scores = border_scores_indices[:, 0]

    border_scores = np.full(sdata['table'].n_obs, -1.0)
    border_scores[indices] = scores

    border_cells = border_scores >= thresh

    sdata['table'].obs['border_cell'] = border_cells
    sdata['table'].obs['border_scores'] = border_scores

    # Plot for border cells
    helperfuncs.plot_scatter(sdata['table'], figure_path, 'border_cell', None,
                             'border_cell', ['lightblue', 'red'], 'Border Cells')


def multi_nuceli_qc(sdata, figure_path):

    # This is from convexity calculations
    nuclei_counts = [len(x) for x in sdata['table'].obs['nuclei_idxs']]

    sdata['table'].obs['multi_nuceli'] = [True if x > 1 else False for x in nuclei_counts]
    sdata['table'].obs['wmulti_nuceli'] = [1 if x > 1 else 0 for x in nuclei_counts]

    sdata['table'].obs['nucleus_free'] = [True if x == 0 else False for x in nuclei_counts]
    sdata['table'].obs['wnucleus_free'] = [1 if x == 0 else 0 for x in nuclei_counts]

    sdata['table'].obs['nuceli_count'] = nuclei_counts

    figures = []

    fig = px.bar(sdata['table'].obs, x='nuceli_count')
    fig.update_layout(
        title=f"Nuclei counts for all samples"
    )
    helperfuncs.apply_general_plotly_layout(fig, True)
    figures.append(fig)
    fig.write_image(f"{figure_path}/barplot_nuceli_counts.png", scale=3)

    with open(f'{figure_path}/nuceli_counts.html', 'w') as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    for cat in ['multi_nuceli', 'nucleus_free']:
        helperfuncs.plot_scatter(sdata['table'], figure_path, cat, None, 
                                 cat, ['lightblue', 'red'], f'Cells with {cat}')

        if ( len( [True for x in sdata['table'].obs[cat] if x == True ] ) > 10 ):
            helperfuncs.plot_density(sdata['table'], f'w{cat}', figure_path)
            helperfuncs.plot_scatter_density(sdata['table'], figure_path, cat, 
                                             cat, f'w{cat}', ['lightblue', 'red'], f'Cells with {cat}')
