"""spoqc.core.spatial vs a brute-force scan with the same shapely predicate."""

import numpy as np
import pytest
import shapely
from shapely.geometry import Point, Polygon, box

from spoqc.core import spatial


def brute_force_pairs(polygons, points):
    pairs = [(i, j) for i, poly in enumerate(polygons) for j in np.flatnonzero(shapely.intersects(points, poly))]
    return (np.array([p[0] for p in pairs], dtype=np.int64), np.array([p[1] for p in pairs], dtype=np.int64))


@pytest.fixture
def polygons_and_points():
    rng = np.random.default_rng(0)
    polygons = [box(10 * i, 0, 10 * i + 10, 10) for i in range(5)]                 # shared edges
    polygons.append(Polygon([(0, 20), (10, 20), (10, 30), (0, 30)],
                            [[(3, 23), (7, 23), (7, 27), (3, 27)]]))             # hole
    polygons.append(Polygon([(60, 20), (70, 30), (70, 20), (60, 30)]))          # invalid bow-tie
    polygons += [shapely.buffer(Point(xy), 4, quad_segs=3) for xy in rng.uniform(0, 100, (60, 2))]  # overlapping
    points = [Point(10, 5), Point(20, 10), Point(5, 25), Point(1, 21), Point(62, 25), Point(65, 29)]
    points += [Point(xy) for xy in np.round(rng.uniform(-5, 105, (400, 2)), 1)]
    return np.array(polygons), np.array(points)


@pytest.mark.parametrize('threads', [1, 2, 5, 16])
def test_polygons_containing_matches_brute_force(polygons_and_points, threads):
    polygons, points = polygons_and_points
    expected = brute_force_pairs(polygons, points)
    actual = spatial.polygons_containing(polygons, points, threads)
    for exp, act in zip(expected, actual):
        np.testing.assert_array_equal(act, exp, strict=True)


def test_polygons_containing_counts_boundary_points_for_every_touching_polygon(polygons_and_points):
    polygons, points = polygons_and_points
    polygon_pos, point_pos = spatial.polygons_containing(polygons, points, 2)
    assert polygon_pos[point_pos == 0].tolist()[:2] == [0, 1]  # Point(10, 5) on the shared edge of boxes 0 and 1


def test_polygons_containing_without_points_returns_empty_pairs(polygons_and_points):
    polygons, _ = polygons_and_points
    polygon_pos, point_pos = spatial.polygons_containing(polygons, np.array([], dtype=object), 2)
    assert len(polygon_pos) == 0 and len(point_pos) == 0
