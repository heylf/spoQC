"""Spatial-index queries returning (left, right) position pair arrays.

The index only proposes candidates (by bounding box, or within a padded radius); the
exact predicate of the original per-object code decides, so results are identical to
a full scan. These are spoQC's only containment, radius and nearest-neighbour searches.
"""

from concurrent.futures import ThreadPoolExecutor
from itertools import chain

import numpy as np
import shapely
from scipy.spatial import cKDTree

from .groupreduce import group_offsets, ragged_lists, ragged_reduce


def polygons_containing(polygons: np.ndarray, points: np.ndarray, threads: int):
    """
    Finds every (polygon, point) pair where shapely.intersects(point, polygon),
    i.e. the point lies inside or on the boundary of the polygon.

    Parameters:
        polygons (np.ndarray): shapely polygons.
        points (np.ndarray): shapely points.
        threads (int): number of threads; each handles a contiguous chunk of polygons.

    Returns:
        Tuple[np.ndarray, np.ndarray]: polygon positions and point positions,
        sorted by polygon position, then point position.
    """
    tree = shapely.STRtree(points)

    def pairs(chunk):
        polygon_pos, point_pos = tree.query(polygons[chunk])
        polygon_pos = chunk[polygon_pos]
        hit = shapely.intersects(points[point_pos], polygons[polygon_pos])
        return polygon_pos[hit], point_pos[hit]

    with ThreadPoolExecutor(threads) as executor:
        chunk_pairs = list(executor.map(pairs, np.array_split(np.arange(len(polygons)), threads)))
    polygon_pos = np.concatenate([p[0] for p in chunk_pairs])
    point_pos = np.concatenate([p[1] for p in chunk_pairs])
    order = np.lexsort((point_pos, polygon_pos))
    return polygon_pos[order], point_pos[order]


def _search_pad(dtype) -> float:
    """
    Relative pad on the KD-tree search radius.

    The deciding formula runs on the dtype-rounded coordinates the tree also holds, so its
    only error is the rounding of a subtraction, two squares, a sum and a square root, which
    is relative and below 4 * eps(dtype). The pad covers that 16-fold for float32 and keeps
    a 1e-9 floor so float64 callers stay far above the tree's own float64 round-off.
    """
    return max(64 * np.finfo(dtype).eps, 1e-9)


def _cast(query_xy, ref_xy, dtype):
    query_xy, ref_xy = np.asarray(query_xy), np.asarray(ref_xy)
    dtype = np.result_type(query_xy, ref_xy) if dtype is None else np.dtype(dtype)
    return query_xy.astype(dtype, copy=False), ref_xy.astype(dtype, copy=False), dtype


def _distances(query, ref, query_pos, ref_pos):
    """The callers' original expression, on gathered pairs, in the arrays' dtype."""
    return np.sqrt((ref[ref_pos, 0] - query[query_pos, 0]) ** 2 + (ref[ref_pos, 1] - query[query_pos, 1]) ** 2)


def _candidates(tree, query64, radius, threads):
    """(query, ref) pairs the tree finds within radius (a scalar or one radius per query), grouped by query."""
    n_candidates = tree.query_ball_point(query64, radius, workers=threads, return_length=True)
    hit = np.flatnonzero(n_candidates)
    neighbours = tree.query_ball_point(query64[hit], radius if np.ndim(radius) == 0 else radius[hit], workers=threads)
    ref_pos = np.fromiter(chain.from_iterable(neighbours), dtype=np.intp, count=n_candidates[hit].sum())
    return np.repeat(hit, n_candidates[hit]), ref_pos


def pairs_within(query_xy, ref_xy, r, threads: int, dtype=None, exclude_self: bool = False,
                 decide: str = "formula", leafsize: int = 16):
    """
    Finds every (query, ref) pair with
        np.sqrt((ref_x - query_x) ** 2 + (ref_y - query_y) ** 2) <= r
    evaluated in `dtype` on coordinates cast to `dtype` (decide="formula"), or every pair
    scipy's cKDTree.query_ball_point returns (decide="tree").

    That is the pandas expression `np.sqrt((df['x'] - x1) ** 2 + (df['y'] - y1) ** 2) <= r`
    with df[['x', 'y']] of dtype `dtype` and a float scalar (x1, y1): pandas casts the
    scalar to the Series dtype for the arithmetic, and compares with `r` exactly as numpy
    does, so pass `r` as the object the original compared with (e.g. a Python int).

    decide="formula": a KD-tree over ref (queried with `threads` workers) proposes candidates
    within a padded radius; the expression decides. The result is therefore identical to
    evaluating the expression on every pair.

    decide="tree": for callers whose original decided with query_ball_point itself (a float64
    sum of squares against r * r at the leaves, plus whole subtrees accepted by their bounds,
    so the answer depends on the tree's leafsize). The tree is built exactly as the original
    built it, cKDTree(ref as float64, leafsize=leafsize) with scipy's other defaults, and
    queried with the float64 query points and r, no pad and no formula: the pairs are the
    original's. `dtype` is not used.

    Non-finite coordinates raise ValueError (scipy).

    Parameters:
        query_xy (np.ndarray): (n_query, 2) coordinates.
        ref_xy (np.ndarray): (n_ref, 2) coordinates.
        r: radius (scalar).
        threads (int): KD-tree query workers.
        dtype: arithmetic dtype; defaults to np.result_type(query_xy, ref_xy).
        exclude_self (bool): drop (i, i) pairs; for query_xy and ref_xy being the same points.
        decide (str): "formula" or "tree", see above.
        leafsize (int): KD-tree leaf size for decide="tree" (scipy KDTree default 10, cKDTree 16).

    Returns:
        Tuple[np.ndarray, np.ndarray]: query positions and ref positions, sorted by query
        position, then ref position. groupreduce.group_offsets(query_pos, n_query) groups them.
    """
    if decide == "tree":
        tree = cKDTree(np.asarray(ref_xy, dtype=np.float64), leafsize=leafsize)
        query_pos, ref_pos = _candidates(tree, np.asarray(query_xy, dtype=np.float64), r, threads)
        keep = np.ones(len(query_pos), dtype=bool)
    elif decide == "formula":
        query, ref, dtype = _cast(query_xy, ref_xy, dtype)
        tree = cKDTree(ref.astype(np.float64))
        query_pos, ref_pos = _candidates(tree, query.astype(np.float64), float(r) * (1 + _search_pad(dtype)), threads)
        keep = _distances(query, ref, query_pos, ref_pos) <= r
    else:
        raise ValueError(f"decide must be 'formula' or 'tree', not {decide!r}")
    if exclude_self:
        keep &= query_pos != ref_pos
    query_pos, ref_pos = query_pos[keep], ref_pos[keep]
    order = np.lexsort((ref_pos, query_pos))
    return query_pos[order], ref_pos[order]


def neighbour_lists(xy, r, threads: int) -> list:
    """For each point, a list of the positions of the other points within r (pairs_within), ascending."""
    point_pos, neighbour_pos = pairs_within(xy, xy, r, threads, exclude_self=True)
    return ragged_lists(neighbour_pos.tolist(), group_offsets(point_pos, len(xy)))


def nearest(query_xy, ref_xy, threads: int, distance_upper_bound: float = np.inf, dtype=None):
    """
    Nearest ref point of every query point.

    ref_pos is cKDTree(ref).query(query, k=1, distance_upper_bound=...) on the float64 values of
    the dtype-cast coordinates: scipy's decision, ties included; len(ref_xy) where no ref point
    lies within distance_upper_bound.

    distance is the minimum over ALL ref points of the pairs_within expression (in `dtype`),
    exact by construction: with d0 the tree's nearest distance, the expression's minimiser j*
    satisfies expr(j*) <= expr(tree nearest), so its exact distance is below d0 * (1 + pad);
    the tree's candidates within d0 * (1 + pad) ** 2 therefore contain j*, and the minimum
    over a superset that contains j* is the global minimum. inf where the tree found none.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ref positions (np.intp) and distances (dtype).
    """
    query, ref, dtype = _cast(query_xy, ref_xy, dtype)
    query64 = query.astype(np.float64)
    tree = cKDTree(ref.astype(np.float64))
    tree_distance, ref_pos = tree.query(query64, k=1, distance_upper_bound=distance_upper_bound, workers=threads)

    distance = np.full(len(query), np.inf, dtype=dtype)
    found = np.flatnonzero(np.isfinite(tree_distance))
    # Every found query has at least its tree nearest among the candidates.
    query_pos, cand_pos = _candidates(tree, query64[found], tree_distance[found] * (1 + _search_pad(dtype)) ** 2, threads)
    d = _distances(query[found], ref, query_pos, cand_pos)
    offsets = group_offsets(query_pos, len(found))
    distance[found] = ragged_reduce(d, offsets, np.min, np.full(len(found), np.inf, dtype=dtype))
    return ref_pos, distance
