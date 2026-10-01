"""Differential test: define_bounding_boxes and spoqc.core.raster vs a verbatim copy of the
original (tests/reference_bounding_boxes.py).

Every figure call is captured instead of written (plot_pixels, plt.imshow, plt.plot, save_figure),
so the arrays handed to the figure code are compared too, together with the returned boxes and
the written .txt. Mutants of the new code must fail the same comparison.
"""

import contextlib
import inspect
import os
import types
from unittest import mock

import dask.array as da
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xarray as xr
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk

import reference_bounding_boxes as ref
from legacy.parquet_writer import ddf_to_parquet  # origin/dev's writer, the reference for core.parquet
from spoqc import helperfuncs
from spoqc.core import figures, raster
from spoqc.hqr import combine_masks_zoom
from spoqc.image_analysis import bounding_boxes
from spoqc.metrics.transcript_density import transcript_density_image

IMAGE_TYPE, RESOLUTION = "morphology_focus", "scale0"


# ---------------------------------------------------------------------------
# Synthetic masks
# ---------------------------------------------------------------------------
def masks():
    """Named 0/1 int8 masks: empty, full, single pixels, border, touching and random cases."""
    rng = np.random.default_rng(7)
    out = {"empty": np.zeros((40, 50), np.int8), "full": np.ones((40, 50), np.int8)}
    for name, (y, x) in {
        "centre": (20, 25),
        "corner00": (0, 0),
        "cornerNN": (39, 49),
        "edge": (0, 17),
    }.items():
        m = np.zeros((40, 50), np.int8)
        m[y, x] = 1
        out[f"pixel_{name}"] = m
    border = np.zeros((40, 50), np.int8)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = 1
    out["border"] = border
    diagonal = np.zeros((40, 50), np.int8)
    idx = np.arange(40)
    diagonal[idx, idx] = 1  # one 8-connected component, 40 pieces under 4-connectivity
    diagonal[5, 30] = diagonal[6, 31] = diagonal[7, 30] = 1
    out["diagonal"] = diagonal
    spiral = np.zeros(
        (41, 41), np.int8
    )  # one component that crosses every row seam several times
    for k in range(0, 20, 4):
        spiral[k, k : 41 - k] = spiral[40 - k, k : 41 - k] = spiral[k : 41 - k, k] = 1
        spiral[k + 2 : 41 - k, 40 - k] = 1
    out["spiral"] = spiral
    for density in (0.002, 0.05, 0.3, 0.6):
        out[f"random_{density}"] = (rng.random((63, 77)) < density).astype(np.int8)
    out["row"] = (rng.random((1, 90)) < 0.2).astype(np.int8)
    out["column"] = (rng.random((90, 1)) < 0.2).astype(np.int8)
    out["tiny"] = np.array([[0, 1], [1, 0]], np.int8)
    return out


MASKS = masks()


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("numba_threads", [1, 4], indirect=True)
@pytest.mark.parametrize("radius", [0, 1, 2, 3, 10])
@pytest.mark.parametrize("name", sorted(MASKS))
def test_dilate_disk_is_skimage_dilation(name, radius, numba_threads):
    mask = MASKS[name]
    expected = dilation(mask, disk(radius))
    actual = raster.dilate_disk(mask, radius)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def test_dilate_disk_accepts_a_flipped_view():
    mask = MASKS["random_0.05"]
    assert (
        raster.dilate_disk(np.flipud(mask), 3).tobytes()
        == dilation(np.flipud(mask), disk(3)).tobytes()
    )


def test_dilate_disk_rejects_non_binary_images():
    with pytest.raises(ValueError, match="0/1"):
        raster.dilate_disk(np.array([[0, 2]], np.int8), 1)


def skimage_boxes(image):
    return np.array(
        [region.bbox for region in regionprops(label(image))], dtype=np.int64
    ).reshape(-1, 4)


@pytest.mark.parametrize("threads", [1, 2, 3, 8, 200])
@pytest.mark.parametrize("name", sorted(MASKS))
def test_component_boxes_is_regionprops_bbox_in_label_order(name, threads):
    mask = MASKS[name]
    for image in (mask, dilation(mask, disk(1))):
        actual = raster.component_boxes(image, threads)
        assert actual.dtype == np.int64
        np.testing.assert_array_equal(actual, skimage_boxes(image))


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.float32])
def test_read_pixel_columns_matches_dask_on_a_many_part_directory(tmp_path, dtype):
    values = np.random.default_rng(3).integers(0, 100, 5003).astype(dtype)
    frame = pd.DataFrame({"other": np.arange(len(values)), "v": values})
    ddf = dd.from_pandas(
        frame, npartitions=13
    )  # part.10 sorts before part.2 lexicographically
    ddf_to_parquet(ddf, "p", str(tmp_path), [], "mask_smoothed_raw")
    path = f"{tmp_path}/p_output_mask_smoothed_raw"
    assert len(os.listdir(path)) >= 13
    expected = (
        dd.read_parquet(path, columns=["v"], engine="pyarrow")["v"].compute().to_numpy()
    )
    actual = raster.read_pixel_columns(path, ["v"], len(values), 3)["v"]
    assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()


def test_read_pixel_columns_reads_every_row_group_of_a_file(tmp_path):
    values = np.arange(10_007, dtype=np.int64) * 3
    pq.write_table(
        pa.Table.from_arrays([pa.array(values)], names=["d"]),
        f"{tmp_path}/f.parquet",
        row_group_size=1000,
    )
    assert pq.ParquetFile(f"{tmp_path}/f.parquet").metadata.num_row_groups == 11
    assert (
        raster.read_pixel_columns(f"{tmp_path}/f.parquet", ["d"], len(values), 3)["d"].tobytes()
        == values.tobytes()
    )


# ---------------------------------------------------------------------------
# define_bounding_boxes: reference vs new, every figure input captured
# ---------------------------------------------------------------------------
def fake_sdata(image):
    """The sdata[image_type][resolution].image access define_bounding_boxes uses, dask-backed."""
    data = xr.DataArray(da.from_array(image, chunks=(1, 64, 64)), dims=("c", "y", "x"))
    return {IMAGE_TYPE: {RESOLUTION: types.SimpleNamespace(image=data)}}


def snapshot(value):
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, pd.DataFrame):
        return ("frame", tuple(value.columns), snapshot(value.to_numpy()))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(snapshot(v) for v in value))
    if isinstance(value, dict):
        return ("dict", tuple((k, snapshot(v)) for k, v in sorted(value.items())))
    return (type(value).__name__, repr(value))


@contextlib.contextmanager
def captured(module, density=None):
    """Record every figure call of define_bounding_boxes in `module` instead of drawing it.

    density: for synthetic hqtr runs, the density image the reference regenerates (its
    generate_transcript_density_image is replaced by one that returns it and records the
    transcript density figure it would write). The new code reads the saved density instead.
    """
    events = []

    def record(kind):
        def call(*args, **kwargs):
            events.append((kind, snapshot(list(args)), snapshot(kwargs)))

        return call

    def fake_generate(sdata, figure_path, imagedim, image_type, resolution):
        record("transcript_density_figure")(
            sdata is not None, figure_path, image_type, density
        )
        return density.flatten()

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(helperfuncs, "plot_pixels", record("plot_pixels"))
        )
        stack.enter_context(mock.patch.object(plt, "imshow", record("imshow")))

        def figures_imshow(ax, *args, dpi=None, **kwargs):
            # the new code's imshow, recorded as the plt.imshow call it replaces
            assert ax is plt.gca() and dpi == 300, (ax, dpi)
            record("imshow")(*args, **kwargs)

        stack.enter_context(mock.patch.object(figures, "imshow", figures_imshow))
        stack.enter_context(mock.patch.object(plt, "plot", record("plot")))
        stack.enter_context(
            mock.patch.object(
                module,
                "save_figure",
                lambda fig, *paths, **kw: events.append(("save", paths, snapshot(kw))),
            )
        )
        if density is not None and module is ref:
            stack.enter_context(
                mock.patch.object(
                    ref.metrics.transcript_density.transcript_density_image,
                    "generate_transcript_density_image",
                    fake_generate,
                )
            )
        if module is not ref:
            stack.enter_context(
                mock.patch.object(
                    transcript_density_image,
                    "generate_transcript_density_image",
                    mock.Mock(side_effect=AssertionError("density regenerated")),
                )
            )
        yield events


def run_define(
    module,
    sdata,
    figure_path,
    tmp,
    modality,
    imagedim,
    dim_x,
    dim_y,
    threads,
    density=None,
    **kwargs,
):
    """Run one define_bounding_boxes and return (boxes, txt bytes, figure events)."""
    staining = kwargs.get("staining")
    folder = f"{figure_path}/{modality}/{modality}_bounding_box/" + (
        f"{staining}/" if staining else ""
    )
    os.makedirs(f"{folder}/subfigures", exist_ok=True)
    extra = () if module is ref else (threads,)
    with captured(module, density) as events:
        boxes = module.define_bounding_boxes(
            sdata,
            figure_path,
            tmp,
            modality,
            IMAGE_TYPE,
            RESOLUTION,
            dim_x,
            dim_y,
            imagedim,
            "raw",
            *extra,
            **kwargs,
        )
    txt = (
        f"{folder}/{modality}s_{staining}.txt"
        if modality == "hqpr"
        else f"{folder}/{modality}s.txt"
    )
    with open(txt, "rb") as f:
        written = f.read()
    os.remove(txt)
    return boxes, written, events


def write_mask(tmp, prefix, mask, npartitions):
    """The mask parquet directory pixel_scoring_refinement writes."""
    frame = pd.DataFrame(
        {
            f"{prefix}_beliefs": mask.ravel().astype(np.float32),
            f"{prefix}_beliefs_smoothed": mask.ravel().astype(np.float32),
            f"{prefix}_mask_smoothed": mask.ravel(),
        }
    )
    ddf_to_parquet(
        dd.from_pandas(frame, npartitions=npartitions),
        prefix,
        tmp,
        [],
        "mask_smoothed_raw",
    )


def write_density(tmp, density):
    """The density parquet structure_analysis writes for hqtr."""
    os.makedirs(f"{tmp}/metrices/hqtr", exist_ok=True)
    helperfuncs.nparr_to_parquet(
        density.flatten(), "transcript_density", f"{tmp}/metrices/hqtr", "hqtr"
    )


def is_density_figure(event):
    """The reference's 'transcript_density' figure, a duplicate of the hqtr_metrices one that the
    new bounding-box step no longer writes (one figure per plot)."""
    if event[0] == "transcript_density_figure":
        return True
    return event[0] == "plot_pixels" and event[1][1][3] == ("str", "'transcript_density'")


def assert_same(expected, actual):
    boxes_e, txt_e, events_e = expected
    boxes_a, txt_a, events_a = actual
    events_e = [e for e in events_e if not is_density_figure(e)]
    assert not any(is_density_figure(e) for e in events_a), "bounding boxes wrote the density figure"
    assert snapshot(boxes_a) == snapshot(boxes_e), (boxes_e, boxes_a)
    assert all(type(v) is float for box in boxes_a for v in box)
    assert txt_a == txt_e
    assert [e[0] for e in events_a] == [e[0] for e in events_e]
    for i, (e, a) in enumerate(zip(events_e, events_a)):
        assert a == e, f"figure call {i} ({e[0]}) differs"


def scene(shape=(160, 190), seed=0):
    """A mask with components of every kind plus an image and a density image of the same shape."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, np.int8)
    mask[10:40, 5:60] = 1  # large
    mask[50:52, 70:72] = 1  # small, dropped by the size filter
    mask[0:30, 150:190] = 1  # touches the top/right border
    mask[100:159, 0:3] = 1  # touches the left/bottom border
    mask[80:120, 80:84] = 1  # two blobs joined only after dilation
    mask[80:120, 86:90] = 1
    rr = np.arange(60)
    mask[95 + rr // 2, 100 + rr] = 1  # thin diagonal line
    mask |= (rng.random(shape) < 0.002).astype(np.int8)
    image = rng.integers(0, 4000, (2,) + shape).astype(np.uint16)
    density = rng.integers(0, 30, shape).astype(np.int64)
    return mask, image, density


@pytest.mark.parametrize("threads", [1, 3, 8])
@pytest.mark.parametrize(
    "modality, kwargs",
    [
        ("hqpr", dict(staining="1", minum_num_pixel=150)),
        ("hqpr", dict(staining="0", minum_num_pixel=150, dilation_radius=3)),
        ("hqtr", dict(dilation_radius=1, minum_num_pixel=150)),
        ("hqtr", dict(dilation_radius=1, minum_num_pixel=10**9)),  # nothing kept
    ],
)
def test_define_bounding_boxes_is_bit_identical(tmp_path, modality, kwargs, threads):
    mask, image, density = scene()
    prefix = f"hqpr_{kwargs['staining']}" if modality == "hqpr" else "hqtr"
    write_mask(str(tmp_path), prefix, mask, npartitions=12)
    write_density(str(tmp_path), density)
    imagedim = helperfuncs.ImageDimStruct(
        np.float64(1000.0), np.float64(2000.0), np.float64(1190.0), np.float64(2160.0)
    )
    args = (
        fake_sdata(image),
        str(tmp_path / "fig"),
        str(tmp_path),
        modality,
        imagedim,
        *mask.shape,
        threads,
    )
    expected = run_define(
        ref, *args, density=density if modality == "hqtr" else None, **kwargs
    )
    actual = run_define(
        bounding_boxes, *args, density=density if modality == "hqtr" else None, **kwargs
    )
    if kwargs["minum_num_pixel"] == 150:
        assert len(expected[0]) >= 3
    assert_same(expected, actual)


def test_define_bounding_boxes_on_an_empty_mask(tmp_path):
    _, image, _ = scene()
    write_mask(
        str(tmp_path), "hqpr_0", np.zeros(image.shape[1:], np.int8), npartitions=3
    )
    imagedim = helperfuncs.ImageDimStruct(
        np.float64(0.0), np.float64(0.0), np.float64(190.0), np.float64(160.0)
    )
    args = (
        fake_sdata(image),
        str(tmp_path / "fig"),
        str(tmp_path),
        "hqpr",
        imagedim,
        *image.shape[1:],
        2,
    )
    expected = run_define(ref, *args, staining="0")
    assert expected[0] == []
    assert_same(expected, run_define(bounding_boxes, *args, staining="0"))


def test_load_intensity_image_hqpr_is_the_flipped_channel():
    _, image, _ = scene()
    for staining, channel in ((None, 0), ("0", 0), ("1", 1)):
        actual = raster.load_intensity_image(
            fake_sdata(image),
            None,
            "hqpr",
            IMAGE_TYPE,
            RESOLUTION,
            160,
            190,
            2,
            staining=staining,
        )
        expected = np.flipud(image[channel])
        assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()


# ---------------------------------------------------------------------------
# Mutants: each must break the comparison above
# ---------------------------------------------------------------------------
def mutant(module, old, new):
    source = inspect.getsource(module)
    assert source.count(old) == 1, old
    clone = types.ModuleType(f"{module.__name__}_mutant")
    clone.__package__ = module.__package__
    clone.__file__ = module.__file__
    exec(compile(source.replace(old, new), module.__file__, "exec"), clone.__dict__)
    return clone


RASTER_MUTANTS = {
    "4-connectivity": (
        "_EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)",
        "_EIGHT_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)",
    ),
    "no diagonal seam links": ("for shift in (-1, 0, 1):", "for shift in (0,):"),
    "no seam links": ("for shift in (-1, 0, 1):", "for shift in ():"),
    # csgraph numbers components by their lowest node, so an unsorted `merged` is already in
    # first-piece order (an equivalent mutant); a different order must be caught.
    "components in box order": (
        "return merged[np.argsort(first_piece)]",
        "return merged[np.lexsort((merged[:, 1], merged[:, 0]))]",
    ),
    "components reversed": (
        "return merged[np.argsort(first_piece)]",
        "return merged[np.argsort(first_piece)[::-1]]",
    ),
    "radius + 1": (
        "footprint = disk(radius).astype(bool)",
        "footprint = disk(radius + 1).astype(bool)",
    ),
    "square footprint": (
        "half_widths = ((footprint.sum(axis=1) - 1) // 2).astype(np.int64)",
        "half_widths = np.full(len(footprint), (len(footprint) - 1) // 2, dtype=np.int64)",
    ),
    "strict gap": (
        "hit[x] |= gaps[source, x] <= width",
        "hit[x] |= gaps[source, x] < width",
    ),
    "lexicographic parts": ("key=natural_sort_key)", "key=None)"),
    "image not flipped": ("    return np.flipud(image)", "    return image"),
    "wrong channel": ("channel = int(staining) if staining else 0", "channel = 0"),
}

BOUNDING_BOX_MUTANTS = {
    ">= size filter": (
        "kept_boxes = boxes[num_pixels > minum_num_pixel]",
        "kept_boxes = boxes[num_pixels >= minum_num_pixel]",
    ),
    "numpy floats in txt": (
        "bounding_boxes = (kept_boxes + offset).tolist()",
        "bounding_boxes = [list(b) for b in (kept_boxes + offset)]",
    ),
    "x/y offset swapped": (
        "offset = np.array([imagedim.bb_ymin, imagedim.bb_xmin, imagedim.bb_ymin, imagedim.bb_xmin]",
        "offset = np.array([imagedim.bb_xmin, imagedim.bb_ymin, imagedim.bb_xmin, imagedim.bb_ymin]",
    ),
    "last subfigure index": (
        "    idx = len(kept_boxes)",
        "    idx = len(kept_boxes) - 1",
    ),
}


def mutant_differs(tmp_path, new_module, modality, kwargs):
    mask, image, density = scene()
    # Make one kept component area exactly equal the threshold for the '>=' mutant.
    mask[125:145, 25:50] = 0
    mask[130:140, 30:45] = 1
    prefix = f"hqpr_{kwargs['staining']}" if modality == "hqpr" else "hqtr"
    write_mask(str(tmp_path), prefix, mask, npartitions=12)
    write_density(str(tmp_path), density)
    imagedim = helperfuncs.ImageDimStruct(
        np.float64(1000.0), np.float64(2000.0), np.float64(1190.0), np.float64(2160.0)
    )
    args = (
        fake_sdata(image),
        str(tmp_path / "fig"),
        str(tmp_path),
        modality,
        imagedim,
        *mask.shape,
        3,
    )
    dens = density if modality == "hqtr" else None
    expected = run_define(ref, *args, density=dens, **kwargs)
    try:
        assert_same(expected, run_define(new_module, *args, density=dens, **kwargs))
    except (AssertionError, ValueError):
        return True
    return False


@pytest.mark.parametrize("name", sorted(RASTER_MUTANTS))
def test_raster_mutant_is_caught(tmp_path, name):
    broken = mutant(raster, *RASTER_MUTANTS[name])
    kwargs = dict(
        staining="1", dilation_radius=1, minum_num_pixel=12 * 17
    )  # the 10x15 block dilates to 12x17
    with mock.patch.object(bounding_boxes, "raster", broken):
        caught = mutant_differs(tmp_path, bounding_boxes, "hqpr", kwargs)
        if (
            not caught
        ):  # seam and ordering mutants need a component split across many blocks
            caught = any(
                not np.array_equal(
                    broken.component_boxes(MASKS[m], 8), skimage_boxes(MASKS[m])
                )
                for m in ("spiral", "diagonal", "random_0.3")
            )
    assert caught, f"mutant {name!r} survived"


@pytest.mark.parametrize("name", sorted(BOUNDING_BOX_MUTANTS))
def test_bounding_box_mutant_is_caught(tmp_path, name):
    broken = mutant(bounding_boxes, *BOUNDING_BOX_MUTANTS[name])
    kwargs = dict(staining="1", dilation_radius=1, minum_num_pixel=12 * 17)
    assert mutant_differs(tmp_path, broken, "hqpr", kwargs), f"mutant {name!r} survived"


def test_read_pixel_columns_raises_on_an_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="no .parquet part files"):
        raster.read_pixel_columns(str(tmp_path), ["v"], 2, 3)


@pytest.mark.parametrize("type_", [pa.int8(), pa.uint8(), pa.int32(), pa.int64(), pa.float32(), pa.float64()])
@pytest.mark.parametrize("null_part", [0, 1])
def test_read_pixel_columns_matches_dask_on_nulls(tmp_path, type_, null_part):
    """Nulls in one part only: NaN for floats; integers become float64 everywhere, as dask returns them."""
    parts = [[1, 0, 1, 1], [0, 1, 1]]
    parts[null_part][1] = None
    for i, values in enumerate(parts):
        pq.write_table(
            pa.table({"v": pa.array(values, type=type_)}), f"{tmp_path}/part.{i}.parquet", row_group_size=2
        )
    expected = dd.read_parquet(str(tmp_path), columns=["v"], engine="pyarrow")["v"].compute().to_numpy()
    assert np.isnan(expected).sum() == 1
    actual = raster.read_pixel_columns(str(tmp_path), ["v"], len(expected), 3)["v"]
    assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()


def test_read_pixel_columns_raises_on_nulls_in_a_bool_column(tmp_path):
    """dask would return an object array of True/False/None; no pixel column is boolean."""
    pq.write_table(pa.table({"v": pa.array([True, None], type=pa.bool_())}), f"{tmp_path}/part.0.parquet")
    with pytest.raises(ValueError, match="1 nulls in the bool column"):
        raster.read_pixel_columns(str(tmp_path), ["v"], 2, 3)


def test_read_pixel_columns_raises_on_parts_with_different_types(tmp_path):
    pq.write_table(pa.table({"v": pa.array([1, 2], type=pa.int8())}), f"{tmp_path}/part.0.parquet")
    pq.write_table(pa.table({"v": pa.array([3, 4], type=pa.int64())}), f"{tmp_path}/part.1.parquet")
    with pytest.raises(ValueError, match="disagree"):
        raster.read_pixel_columns(str(tmp_path), ["v"], 4, 3)


def test_hqtr_bounding_boxes_no_longer_write_the_duplicate_density_figure(tmp_path):
    mask, image, density = scene()
    write_mask(str(tmp_path), "hqtr", mask, npartitions=4)
    write_density(str(tmp_path), density)
    imagedim = helperfuncs.ImageDimStruct(np.float64(0.0), np.float64(0.0), np.float64(190.0), np.float64(160.0))
    args = (fake_sdata(image), str(tmp_path / "fig"), str(tmp_path), "hqtr", imagedim, *mask.shape, 2)
    expected = run_define(ref, *args, density=density, dilation_radius=1)
    actual = run_define(bounding_boxes, *args, density=density, dilation_radius=1)
    assert sum(map(is_density_figure, expected[2])) == 1
    assert not any(map(is_density_figure, actual[2]))
    assert_same(expected, actual)


class FakeSdata(dict):
    """fake_sdata plus the sdata.labels access of the zoom step."""


def test_combine_masks_zoom_no_longer_writes_the_duplicate_density_figure(tmp_path):
    """The zoom step reads the saved density for its hqtr input figure and writes no full-image
    'transcript_density' figure (a duplicate of the hqtr_metrices one: one figure per
    plot). Its unsmoothed pass still raises the pre-existing NotImplementedError.

    Pre-existing (origin/dev too): combined_beliefs reads *_beliefs_smoothed columns that beliefs_df
    does not have, so the step dies with a KeyError before its modality loop. The test pins that,
    then gives beliefs_df those aliases to reach the loop the duplicate figure was written from."""
    mask, image, density = scene()
    tmp = str(tmp_path)
    pd.DataFrame(
        {"hqcr_beliefs_smoothed": mask.ravel().astype(np.float32), "hqcr_mask_smoothed": mask.ravel()}
    ).to_parquet(f"{tmp}/hqcr_output_mask_smoothed_raw.parquet")
    write_mask(tmp, "hqpr_0", mask, npartitions=2)
    write_mask(tmp, "hqtr", mask, npartitions=2)
    write_density(tmp, density)
    labels = types.SimpleNamespace(image=xr.DataArray((mask > 0).astype(np.int32), dims=("y", "x")))
    sdata = FakeSdata(fake_sdata(image))
    sdata.labels = {seg: {RESOLUTION: labels} for seg in ("cell_labels", "nucleus_labels")}
    imagedim = helperfuncs.ImageDimStruct(0.0, 0.0, 190.0, 160.0)
    suffixes = []

    def plot(figure_path, image, imagedim, suffix, *args, **kwargs):
        suffixes.append(suffix)

    args = (sdata, str(tmp_path / "fig"), tmp, IMAGE_TYPE, RESOLUTION, imagedim, *mask.shape, "0", 2)
    with (
        mock.patch.object(helperfuncs, "plot_pixels", plot),
        pytest.raises(KeyError, match="hqcr_beliefs_smoothed"),
    ):
        combine_masks_zoom.start_combining_masks(*args)
    suffixes.clear()

    def frame_with_smoothed_aliases(data):
        frame = pd.DataFrame(data)
        if "hqcr_beliefs" in frame:
            for column in list(frame.columns):
                frame[f"{column}_smoothed"] = frame[column]
        return frame

    with (
        mock.patch.object(helperfuncs, "plot_pixels", plot),
        mock.patch.object(
            combine_masks_zoom, "pd", types.SimpleNamespace(DataFrame=frame_with_smoothed_aliases, read_parquet=pd.read_parquet)
        ),
        mock.patch.object(
            transcript_density_image,
            "generate_transcript_density_image",
            mock.Mock(side_effect=AssertionError("density regenerated")),
        ),
        mock.patch.object(raster, "load_intensity_image", wraps=raster.load_intensity_image) as load,
        pytest.raises(NotImplementedError, match="unsmoothed pass"),
    ):
        combine_masks_zoom.start_combining_masks(*args)
    assert "transcript_density" not in suffixes, suffixes
    assert "input_transcript_densities_zoom_smoothed" in suffixes, suffixes
    assert "input_pixel_intensities_zoom_smoothed" in suffixes, suffixes
    assert [c.args[2] for c in load.call_args_list] == ["hqpr", "hqtr"]
