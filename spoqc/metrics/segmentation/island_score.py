import numpy as np

from scipy.spatial import KDTree
from typing import Tuple, List

from ... import helperfuncs

def find_connected_groups_iterative(points: List[Tuple[float, float]], distance_threshold: float) -> List[List[int]]:
    """
    Identifies connected groups of points based on a distance threshold using an iterative approach.

    Args:
        points (List[Tuple[float, float]]): A list of 2D points represented as tuples of (x, y) coordinates.
        distance_threshold (float): The maximum distance within which points are considered connected.

    Returns:
        List[List[int]]: A list of groups, where each group is represented by a list of indices 
                         of points belonging to that group.
    
    Notes:
        - Points are connected if the distance between them is less than or equal to `distance_threshold`.
        - The algorithm uses a k-d tree for efficient neighbor queries and a breadth-first search (BFS) 
          to determine connected groups.
    """
    # Convert points to a numpy array
    points_array = np.array(points)

    # Build the k-d tree
    tree = KDTree(points_array)

    # Find neighbors for each point within the distance threshold
    neighbors = tree.query_ball_point(points_array, distance_threshold)

    # Perform a connected components search iteratively
    visited = set()
    connected_groups = []

    # Go over each cell and check if you have neighbours in the distance threshold.
    # Check for those neighbours again if you have neighbours in the distance threshold and add them
    # to the group.
    for i in range(len(points)):
        if i not in visited:
            # Start a new group and a queue for BFS
            group = []
            queue = [i]

            while queue:
                node = queue.pop(0)  # Dequeue the first element
                if node not in visited:
                    visited.add(node)
                    group.append(node)
                    # Add unvisited neighbors to the queue
                    queue.extend([neighbor for neighbor in neighbors[node] if neighbor not in visited])

            connected_groups.append(group)

    return connected_groups


def calc_island_score(
        sdata,
        figure_path,
        *,
        distance_threshold = 15,
        min_group_count = 10,
    ):

    # coordinated of the cells
    adata_x = sdata['table'].obsm['spatial'][:,0]
    adata_y = sdata['table'].obsm['spatial'][:,1]

    # Create a list of tuples. x and y should technically be the same since it is a pixel / intensity point.
    adata_coordinates = list(zip(adata_x, adata_y))

    groups = find_connected_groups_iterative(adata_coordinates, distance_threshold)

    island_indices = np.array([-1] * sdata['table'].n_obs)
    island_scores = np.array([-1] * sdata['table'].n_obs)

    for i,group in enumerate(groups):
        island_indices[group] = i
        island_scores[group] = len(group)

    sdata['table'].obs['island_index'] = island_indices
    sdata['table'].obs['island_score'] = island_scores
    sdata['table'].obs['small_islands'] = [True if x < min_group_count else False for x in island_scores]

    helperfuncs.plot_scatter(sdata['table'], figure_path, 'island', None, 
                            'small_islands', None, None)