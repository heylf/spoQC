import pandas as pd
import numpy as np
import concurrent.futures

from scipy.spatial import cKDTree

from ... import helperfuncs

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