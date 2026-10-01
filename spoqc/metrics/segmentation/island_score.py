import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ... import helperfuncs
from ...core import spatial


def find_connected_groups(points: np.ndarray, distance_threshold: float, threads: int) -> np.ndarray:
    """
    Labels connected groups of points: points are connected if they are within
    distance_threshold of each other, directly or through other points.

    Args:
        points (np.ndarray): (n, 2) coordinates.
        distance_threshold (float): The maximum distance within which points are considered connected.
        threads (int): neighbour-search threads.

    Returns:
        np.ndarray: the group of each point; groups are numbered in the order of their
        lowest-positioned point.
    """
    n_points = len(points)
    # The original decided with scipy's KDTree.query_ball_point (leafsize 10).
    pos_a, pos_b = spatial.pairs_within(points, points, distance_threshold, threads, decide="tree", leafsize=10)
    adjacency = coo_matrix((np.ones(len(pos_a), dtype=bool), (pos_a, pos_b)), shape=(n_points, n_points))
    _, labels = connected_components(adjacency, directed=False)
    _, first_point = np.unique(labels, return_index=True)
    return np.argsort(np.argsort(first_point))[labels]


def calc_island_score(sdata, figure_path, distance_threshold, min_group_count, threads):

    island_indices = find_connected_groups(sdata['table'].obsm['spatial'][:, :2], distance_threshold, threads)
    island_scores = np.bincount(island_indices)[island_indices]

    sdata['table'].obs['island_index'] = island_indices
    sdata['table'].obs['island_score'] = island_scores
    sdata['table'].obs['small_islands'] = [True if x < min_group_count else False for x in island_scores]

    helperfuncs.plot_scatter(sdata['table'], figure_path, 'island', None, 
                            'small_islands', None, None)