from concurrent.futures import ThreadPoolExecutor

import numpy as np
from threadpoolctl import threadpool_limits

from ... import helperfuncs
from ...core import groupreduce, spatial


def get_border_scores(points, radius, step, threads):
    """
    Border score of every point: over rotations by multiples of `step` degrees, the largest
    |log2((1 + #neighbours right of the point) / (1 + #neighbours left of it))|, counting the
    points within `radius` (the point itself sits at 0 and counts on neither side).

    The per-point original rotated its (k, 2) neighbour offsets with `diffs @ rotation_matrix`,
    a BLAS call whose kernel (and so its rounding, e.g. FMA use) can depend on the shape.
    Points with the same neighbour count k are stacked into one (points, k, 2) matmul: numpy
    runs it as one (k, 2) @ (2, 2) product per point, with the original's shape, strides and
    row order (neighbours ascending), so each product is the original's on any BLAS.
    """
    n_points = len(points)
    # The original decided with cKDTree.query_ball_point (leafsize 16).
    point_pos, neighbour_pos = spatial.pairs_within(points, points, radius, threads, decide="tree", leafsize=16)
    diffs = points[neighbour_pos] - points[point_pos]  # (pairs, 2), grouped by point
    offsets = groupreduce.group_offsets(point_pos, n_points)
    sizes = np.diff(offsets)
    stacks = []
    for k in np.unique(sizes[sizes > 0]):
        same_size = np.flatnonzero(sizes == k)
        stacks.append((same_size, diffs[offsets[same_size][:, None] + np.arange(k)]))

    angles = np.radians(np.arange(0, 360, step))
    rotation_matrices = np.stack([
        np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        for a in angles
    ])

    def rotation_scores(rotation_matrix):
        # Add one to both sides to avoid inf; both sides are treated equally.
        num_left = np.ones(n_points, dtype=np.int64)
        num_right = np.ones(n_points, dtype=np.int64)
        for same_size, stacked in stacks:
            x_coords = (stacked @ rotation_matrix)[..., 0]
            num_left[same_size] += np.count_nonzero(x_coords > 0, axis=1)
            num_right[same_size] += np.count_nonzero(x_coords < 0, axis=1)
        # Only the magnitude matters, not the direction. The few distinct ratios go through
        # the scalar log2, as the per-point original did.
        ratios, ratio_idx = np.unique(num_left / num_right, return_inverse=True)
        return np.array([abs(np.log2(ratio)) for ratio in ratios])[ratio_idx]

    # One rotation per thread (numpy releases the GIL); BLAS kept to one thread so the
    # total stays within `threads`.
    with threadpool_limits(limits=1, user_api="blas"), ThreadPoolExecutor(threads) as executor:
        return np.max(list(executor.map(rotation_scores, rotation_matrices)), axis=0)


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
        - The border scores are computed using `get_border_scores`.
        - A scatter plot of the border cells is generated using `helperfuncs.plot_scatter`.
    """

    border_scores = get_border_scores(np.ascontiguousarray(sdata['table'].obsm['spatial'][:, :2]), radius, stepsize, threads)

    border_cells = border_scores >= thresh

    sdata['table'].obs['border_cell'] = border_cells
    sdata['table'].obs['border_scores'] = border_scores

    # Plot for border cells
    helperfuncs.plot_scatter(sdata['table'], figure_path, 'border_cell', None,
                             'border_cell', ['lightblue', 'red'], 'Border Cells')