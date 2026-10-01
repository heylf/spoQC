"""Differential test: structural image analysis (hqpr/hqtr pixel metrics) against the verbatim original.

tests/reference_pixel_metrics.py holds the previous metric wrappers and
start_image_struc_analyis. Both versions run on the same synthetic images; every parquet written
to the metrices folder (each is a pixel-clustering feature) must match byte for byte, with the
same file names, and every plot_pixels call must receive the same array, dtype and arguments.
"""

import math
import os
from types import SimpleNamespace

import cv2
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import xarray as xr
from skimage.feature import local_binary_pattern

import reference_pixel_metrics as reference
from spoqc import helperfuncs
from spoqc.image_analysis import _slidingwindow, structure_analysis
from spoqc.metrics.image import pixel_metrics, utility
from spoqc.metrics.transcript_density import transcript_density_image

IMAGE_TYPE, RESOLUTION = "morphology_focus", "s0"
SHAPES = [(1, 9), (9, 1), (2, 3), (5, 5), (7, 4), (4, 9), (37, 23), (1030, 517)]
HQPR_KINDS = ["tissue", "full_range", "constant", "edges"]
HQTR_KINDS = ["density", "zeros"]
THREADS = 2  # the new code's thread budget (raster.load_intensity_image)


def make_image(kind, shape, rng):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    if (
        kind == "tissue"
    ):  # realistic range with blobs, noise and a zero background patch
        img = (
            90
            + 800 * (np.sin(yy / 5.0) * np.cos(xx / 3.0)) ** 2
            + rng.gamma(2.0, 30.0, shape)
        )
        img[: shape[0] // 3, : shape[1] // 3] = 0
        return img.astype(np.uint16)
    if kind == "full_range":  # hits 0 and 65535 (the uint16 +1 wraps)
        img = rng.integers(0, 65536, shape, dtype=np.uint16)
        img.flat[0], img.flat[-1] = 0, 65535
        return img
    if kind == "constant":
        return np.full(shape, 7, dtype=np.uint16)
    if kind == "edges":
        return np.where((yy // 3 + xx // 2) % 2 == 0, 3000, 12).astype(np.uint16)
    if kind == "density":  # hqtr: int64 transcript densities, including values > 255
        img = rng.poisson(3.0, shape).astype(np.int64)
        img[shape[0] // 2 :, shape[1] // 2 :] *= 120
        return img
    if kind == "zeros":
        return np.zeros(shape, dtype=np.int64)
    raise ValueError(kind)


class FakeSdata(dict):
    """The parts of a SpatialData object start_image_struc_analyis reads."""

    def __init__(self, image):
        image3d = image[None]
        super().__init__(
            {
                IMAGE_TYPE: {
                    RESOLUTION: SimpleNamespace(
                        image=xr.DataArray(
                            da.from_array(image3d, chunks=(1, 256, 256)), dims=("c", "y", "x")
                        )
                    )
                }
            }
        )
        # A few transcripts for the (patched-out) transcript point plot: core.transcripts reads them
        # through sdata.points and caches them keyed weakly by this object.
        points = pd.DataFrame({"x": [0.5, 1.5], "y": [0.5, 2.5]})
        self.points = {"transcripts": dd.from_pandas(points, npartitions=1)}

    __hash__ = object.__hash__  # identity, as for a SpatialData object


def record_calls(monkeypatch, density=None):
    """Record plot_pixels / save_figure calls instead of drawing; serve `density` as the hqtr image."""
    calls = []

    def plot_pixels(
        figure_path, image, imagedim, suffix, title, cmap, axis_off, baroff, **kwargs
    ):
        image = np.asarray(image)
        kwargs = {"points": None, "legend_dict": None, "flip": False, **kwargs}  # plot_pixels defaults
        calls.append(
            (
                "plot_pixels",
                suffix,
                title,
                cmap,
                axis_off,
                baroff,
                kwargs,
                image.dtype,
                image.copy(),
            )
        )

    monkeypatch.setattr(helperfuncs, "plot_pixels", plot_pixels)
    monkeypatch.setattr(
        helperfuncs,
        "plot_scatter_by_category",
        lambda *a, **k: calls.append(("scatter",)),
    )
    for module in (reference, utility):
        monkeypatch.setattr(
            module,
            "save_figure",
            lambda fig, *paths, **k: calls.append(("save_figure", paths)),
        )
    if density is not None:
        monkeypatch.setattr(
            transcript_density_image,
            "generate_transcript_density_image",
            lambda *a, **k: density.flatten(),
        )
    return calls


def run(func, base, image, modality, monkeypatch, *extra, **kwargs):
    """Run one start_image_struc_analyis; return (plot calls, {file: parquet table}, figure files)."""
    staining = "0" if modality == "hqpr" else None
    metrices = (
        f"{base}/tmp/metrices/{modality}/{staining}"
        if staining
        else f"{base}/tmp/metrices/{modality}"
    )
    figures = (
        f"{base}/fig/{modality}/{modality}_metrices/{staining}"
        if staining
        else f"{base}/fig/{modality}/{modality}_metrices"
    )
    os.makedirs(metrices)
    os.makedirs(figures)
    calls = record_calls(monkeypatch, density=image if modality == "hqtr" else None)
    dim_x, dim_y = image.shape
    func(
        FakeSdata(image), f"{base}/fig", f"{base}/tmp", modality, IMAGE_TYPE, RESOLUTION,
        None, dim_x, dim_y, True, *extra, **kwargs, **({"staining": staining} if staining else {}),
    )  # fmt: skip
    tables = {f: pq.read_table(f"{metrices}/{f}") for f in sorted(os.listdir(metrices))}
    calls = [
        tuple(
            str(c).replace(base, "") if isinstance(c, (str, tuple)) else c for c in call
        )
        for call in calls
    ]
    return calls, tables, sorted(os.listdir(figures))


def assert_same_calls(new, old):
    assert len(new) == len(old), (len(new), len(old))
    for n, o in zip(new, old):
        if n[0] != "plot_pixels":
            assert n == o
            continue
        assert n[:8] == o[:8], (n[:8], o[:8])
        assert n[8].shape == o[8].shape and n[8].tobytes() == o[8].tobytes(), n[1]


def assert_same_tables(new, old):
    assert list(new) == list(old)
    for name in new:
        assert new[name].schema.equals(old[name].schema, check_metadata=True), name
        assert new[name].equals(old[name]), name
        a, b = new[name].column(0).to_numpy(), old[name].column(0).to_numpy()
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), name


def compare(image, modality, tmp_path, monkeypatch):
    # cv2.GaussianBlur on uint16 (relevance) varies call to call on small images at >= 32 cv2 threads
    # (cv2's default here is every host CPU), in the original too.
    # Pinned to 1 so the original's own race cannot flake the comparison (numba threads are separate).
    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        return _compare(image, modality, tmp_path, monkeypatch)
    finally:
        cv2.setNumThreads(previous)


def _compare(image, modality, tmp_path, monkeypatch):
    old = run(
        reference.start_image_struc_analyis,
        f"{tmp_path}/old",
        image,
        modality,
        monkeypatch,
    )
    monkeypatch.undo()
    new = run(structure_analysis.start_image_struc_analyis, f"{tmp_path}/new", image, modality, monkeypatch, THREADS)
    assert_same_calls(new[0], old[0])
    assert_same_tables(new[1], old[1])
    assert new[2] == old[2]
    return new


@pytest.mark.parametrize("numba_threads", [1, 4], indirect=True)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kind", HQPR_KINDS)
def test_hqpr_matches_original(kind, shape, numba_threads, tmp_path, monkeypatch):
    image = make_image(kind, shape, np.random.default_rng(sum(shape)))
    _, tables, _ = compare(image, "hqpr", tmp_path, monkeypatch)
    expected = [
        "edge_strength",
        "energy",
        "entropy",
        "homogenity",
        "intensity",
        "lbp",
        "relevance",
        "uniformity",
    ]
    assert list(tables) == [f"{m}_output_hqpr_0.parquet" for m in expected]


@pytest.mark.parametrize("numba_threads", [1, 4], indirect=True)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kind", HQTR_KINDS)
def test_hqtr_matches_original(kind, shape, numba_threads, tmp_path, monkeypatch):
    image = make_image(kind, shape, np.random.default_rng(sum(shape)))
    _, tables, _ = compare(image, "hqtr", tmp_path, monkeypatch)
    expected = ["edge_strength", "energy", "entropy", "homogenity", "lbp", "relevance", "transcript_density",
                "uniformity"]  # fmt: skip
    assert list(tables) == [f"{m}_output_hqtr.parquet" for m in expected]


def test_texture_metrics_constant_window_is_perfectly_homogeneous():
    entropy, uniformity, homogeneity = _slidingwindow.texture_metrics(
        np.full((6, 6), 3, np.uint8), 5
    )
    assert (
        (homogeneity == 1).all()
        and (entropy == 0).all()
        and entropy.dtype == np.float32
    )
    assert (uniformity == np.float32(np.log(25))).all()  # kl = -(1 * log(1 / (1/25)))


def lbp_tie_images(P, R, n=4000, width=16, period=16):
    """Column-constant uint16 images where a circle point lands on an exact interpolation tie.

    For each point k whose row offset rp[k] = f + num/D (reduced fraction), rows f and f+1 below
    a row of value cval = num/g hold 0 and D, so the interpolated sample equals the centre
    exactly and texture - centre >= 0 hinges on the last bit of dr. These are the cases where
    chunked (row-offset) LBP differed from one whole-image call.
    """
    i = np.arange(P, dtype=np.float64)
    rp = np.round(-R * np.sin(2 * np.pi * i / P), 5)
    for k in range(P):
        f = math.floor(rp[k])
        num = round((rp[k] - f) * 100000)
        if num == 0:
            continue
        g = math.gcd(num, 100000)
        denominator, cval = 100000 // g, num // g
        if denominator > 65535:
            continue
        img = np.full((n, width), 1, np.uint16)
        for r0 in range(8, n - 8, period):
            img[r0] = cval
            img[r0 + f] = 0
            img[r0 + f + 1] = denominator
        yield k, img


LBP_PARAMS = [(100, 3), (8, 1), (8, 1.5), (16, 2.5), (24, 3.7), (12, 0.4)]


@pytest.mark.parametrize("numba_threads", [1, 4], indirect=True)
@pytest.mark.parametrize("n_points, radius", [p for p in LBP_PARAMS if p != (8, 1)])  # (8, 1): no tie fits uint16
def test_lbp_matches_skimage_on_interpolation_ties(n_points, radius, numba_threads):
    cases = 0
    for k, img in lbp_tie_images(n_points, radius):
        expected = local_binary_pattern(img, n_points, radius, method="uniform")
        got = pixel_metrics.lbp(img, n_points, radius)[0]
        assert got.dtype == expected.dtype and got.tobytes() == expected.tobytes(), (k, int((got != expected).sum()))
        cases += 1
    assert cases > 0


@pytest.mark.parametrize("numba_threads", [1, 4], indirect=True)
@pytest.mark.parametrize("n_points, radius", LBP_PARAMS)
@pytest.mark.parametrize("kind", ["levels0-3", "gamma", "flat_perturbed", "int64_density", "tiny"])
def test_lbp_matches_skimage_on_random_images(kind, n_points, radius, numba_threads):
    rng = np.random.default_rng(n_points)
    img = {
        "levels0-3": lambda: rng.integers(0, 4, (3000, 48)).astype(np.uint16),
        "gamma": lambda: rng.gamma(2, 30, (700, 301)).astype(np.uint16),
        "flat_perturbed": lambda: np.where((np.arange(3000)[:, None] % 13 == 0) & (np.arange(48) % 5 == 0), 1001, 1000).astype(np.uint16),
        "int64_density": lambda: make_image("density", (301, 257), rng),
        "tiny": lambda: rng.integers(0, 5, (3, 2)).astype(np.uint16),
    }[kind]()
    expected = local_binary_pattern(img, n_points, radius, method="uniform")
    got = pixel_metrics.lbp(img, n_points, radius)[0]
    assert got.dtype == expected.dtype and got.tobytes() == expected.tobytes(), int((got != expected).sum())


def float64_texture(img):
    """The new texture kernel stored at float64, i.e. before the float32 rounding that masks
    1-ulp differences in the per-window sums."""
    out = [np.empty(img.shape, np.float64) for _ in range(3)]
    _slidingwindow._texture_windows(np.pad(img, 2, mode="reflect"), 2, *out)
    return out


def float64_reference_texture(img):
    return [
        reference.sliding_window_padded(kernel, img, 2, dtype=np.float64, mode="reflect")
        for kernel in (reference.entropy, reference.kl_divergence_uniform, reference.homogeneity)
    ]


@pytest.mark.parametrize("numba_threads", [4], indirect=True)
@pytest.mark.parametrize(
    "image",
    [
        np.random.default_rng(1).integers(0, 256, (1030, 517)).astype(np.uint8),  # many distinct values
        make_image("density", (517, 1030), np.random.default_rng(2)),
    ],
    ids=["uint8", "int64_density"],
)
def test_texture_window_sums_match_original_at_float64(image, numba_threads):
    for name, new, old in zip(["entropy", "kl", "homogeneity"], float64_texture(image), float64_reference_texture(image)):
        assert new.tobytes() == old.tobytes(), name
