"""Four places where a scalar or a count was computed by walking the whole
array through the Python interpreter, or by materializing a copy of it.

Each test pins behaviour (identical result) and, where the win is memory rather
than time, pins the absence of the copy.
"""
import numpy as np
import pytest


class TestRelevanceMax:
    """relevance.py:33 -- max(arr.flatten()) boxed every pixel as a Python int.
    relevance now lives in pixel_metrics.relevance."""

    def test_source_no_longer_uses_builtin_max(self):
        import inspect

        from spoqc.metrics.image import pixel_metrics

        source = inspect.getsource(pixel_metrics.relevance)
        assert "max(xy_intensities.flatten())" not in source
        assert "xy_intensities.max()" in source

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
    def test_result_identical(self, dtype):
        rng = np.random.default_rng(0)
        a = (rng.random((128, 128)) * 1000).astype(dtype)
        assert a.max() == max(a.flatten())


class TestPixelCountFromShape:
    """pixel_scoring_dask.py:103 -- len(img.values[0].flatten()) read the whole
    image and copied it again, to compute a number available from .shape."""

    def test_source_no_longer_flattens(self):
        import inspect

        from spoqc.image_analysis import pixel_scoring_dask

        assert ".image.values[0].flatten()" not in inspect.getsource(pixel_scoring_dask)

    @pytest.mark.parametrize("shape", [(7, 13), (1, 1), (2048, 4096)])
    def test_count_identical(self, shape):
        a = np.zeros(shape, dtype=np.uint16)
        assert int(np.prod(a.shape[-2:])) == a.size == len(a.flatten())

    def test_works_on_a_channel_dim(self):
        """The real array is (c, y, x); the count must be y*x, not c*y*x."""
        a = np.zeros((1, 64, 32), dtype=np.uint16)
        assert int(np.prod(a.shape[-2:])) == 64 * 32


class TestBackgroundMaskIsWastedWork:
    """hqcr.py:313 -- `flat_index >= 0` is always true for a rasterize(fill=0)
    index map, so the mask copied both full-size arrays while removing nothing.

    It is NOT a correctness fix. `ndimage.mean` only computes the labels named in
    `index`, and rasterized polygon ids start at 1, so background pixels carried
    as label 0 were never read. The results are identical; the work was wasted.
    """

    def test_source_uses_strict_greater_than(self):
        import inspect

        from spoqc.subworkflows import hqcr

        src = inspect.getsource(hqcr)
        assert "flat_index >= 0" not in src
        assert "flat_index > 0" in src

    def test_old_mask_removed_nothing(self):
        index_map = np.array([[0, 0, 1], [0, 2, 2]])
        assert (index_map.ravel() >= 0).all(), (
            "the >= 0 mask is vacuously true, so it only copied the arrays"
        )

    def test_new_mask_keeps_only_in_cell_pixels(self):
        index_map = np.array([[0, 0, 1], [0, 2, 2]])
        assert (index_map.ravel() > 0).sum() == 3

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_results_are_identical_either_way(self, seed):
        """The acceptance bar: both masks must produce the same per-label means."""
        from scipy import ndimage

        rng = np.random.default_rng(seed)
        index_map = rng.integers(0, 6, (40, 40))
        values = rng.random((40, 40)) * 100
        fi, fl = index_map.ravel(), values.ravel()
        labels = np.arange(1, 6)

        old = ndimage.mean(input=fl[fi >= 0], labels=fi[fi >= 0], index=labels)
        new = ndimage.mean(input=fl[fi > 0], labels=fi[fi > 0], index=labels)
        np.testing.assert_array_equal(
            old, new, err_msg="dropping background must not change any label mean"
        )
