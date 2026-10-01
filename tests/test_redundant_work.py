"""void.py built per-triangle point-index lists that nothing ever read.

Stage 1 bar: bit-identical numbers, minimum change. This removes allocation
only -- the counts it returns are untouched -- so there is nothing to compare
numerically beyond "the counts are still right".
"""
import inspect

import numpy as np


class TestTriangleIndexListsRemoved:
    def test_triangle_counter_returns_only_counts(self):
        """indices_list was materialised on 4 call sites and never read."""
        from spoqc.metrics.segmentation import void

        src = inspect.getsource(void.count_stuff_in_triangles_via_delaunay)
        assert "indices_list" not in src
        assert "return counts" in src

    def test_no_call_site_still_unpacks_two_values(self):
        from spoqc.metrics.segmentation import void

        src = inspect.getsource(void)
        assert "counts, indices = count_stuff_in_triangles_via_delaunay" not in src

    def test_counter_still_counts_correctly(self):
        """Behaviour of the surviving return value is unchanged."""
        from scipy.spatial import Delaunay

        from spoqc.metrics.segmentation import void

        pts = np.array([[0.0, 0], [10, 0], [0, 10], [10, 10]])
        d = Delaunay(pts)
        stuff = np.array([[1.0, 1], [2, 2], [9, 9], [-5, -5]])
        counts = void.count_stuff_in_triangles_via_delaunay(d, stuff)
        assert len(counts) == len(d.simplices)
        # the three in-hull points are attributed; the outside one is not
        assert counts.sum() == 3

    def test_counts_match_the_previous_implementation(self):
        """Differential: the removed block never touched `counts`."""
        from scipy.spatial import Delaunay

        from spoqc.metrics.segmentation import void

        def _original(delaunay, stuff):
            num_triangles = len(delaunay.simplices)
            simplex_ids = delaunay.find_simplex(stuff)
            counts = np.bincount(
                simplex_ids[simplex_ids >= 0], minlength=num_triangles
            )
            point_idx = np.nonzero(simplex_ids >= 0)[0]
            order = np.argsort(simplex_ids[point_idx], kind="stable")
            point_idx = point_idx[order]
            sorted_simplex_ids = simplex_ids[point_idx]
            boundaries = np.searchsorted(
                sorted_simplex_ids, np.arange(num_triangles + 1)
            )
            indices_list = [
                (point_idx[boundaries[i]:boundaries[i + 1]],)
                for i in range(num_triangles)
            ]
            return counts, indices_list

        rng = np.random.default_rng(0)
        for seed_points in (12, 40):
            pts = rng.uniform(0, 100, (seed_points, 2))
            d = Delaunay(pts)
            stuff = rng.uniform(-10, 110, (500, 2))
            expected, _ = _original(d, stuff)
            np.testing.assert_array_equal(
                void.count_stuff_in_triangles_via_delaunay(d, stuff), expected
            )
