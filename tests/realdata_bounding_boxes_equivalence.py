"""Real-data differential check: original vs new define_bounding_boxes, hqpr and hqtr.

Not collected by pytest. Usage:
    PYTHONPATH=<repo> python tests/realdata_bounding_boxes_equivalence.py <sdata.zarr> <mask parquet> <threads> [crop]

Crops the sample the way cli.py's TESTING mode does (origin 10500, `crop` pixels square), cuts the
same window out of a real full-scale MRF mask (<mask parquet>: an `*_output_mask_smoothed_raw`
parquet, e.g. hqcr's), and writes the parquets the earlier steps leave behind: the mask directory
(pixel_scoring_refinement) and, for hqtr, the transcript density that structure_analysis saves
from generate_transcript_density_image. Both versions then run with every figure call captured,
and boxes, .txt bytes and every figure input are compared. Prints wall, CPU and traced peak memory.
"""

import os
import sys
import tempfile
import time
import tracemalloc

import contextlib
from unittest import mock

import dask
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import spatialdata as sd
from spatialdata.models import PointsModel

sys.path.insert(0, os.path.dirname(__file__))
import reference_bounding_boxes as ref  # noqa: E402
from test_bounding_boxes_equivalence import (
    assert_same,
    run_define,
    write_density,
    write_mask,
)  # noqa: E402

from spoqc import helperfuncs  # noqa: E402
from spoqc.core import figures  # noqa: E402
from spoqc.image_analysis import bounding_boxes  # noqa: E402

START = 10500
IMAGE_TYPE, RESOLUTION = "morphology_focus", "scale0"


def crop_mask(path, full_shape, rows, cols):
    """rows x cols window of a flattened full-image parquet column, reading only its row groups."""
    file = pq.ParquetFile(path)
    column = [n for n in file.schema_arrow.names if n.endswith("_mask_smoothed")][0]
    lo, hi = rows.start * full_shape[1], rows.stop * full_shape[1]
    parts, offset = [], 0
    for group in range(file.metadata.num_row_groups):
        n = file.metadata.row_group(group).num_rows
        if offset + n > lo and offset < hi:
            values = file.read_row_group(group, columns=[column]).column(0).to_numpy()
            parts.append(values[max(lo - offset, 0) : min(hi - offset, n)])
        offset += n
    return (
        np.concatenate(parts)
        .reshape(rows.stop - rows.start, full_shape[1])[:, cols]
        .copy()
    )


def measured(label, func, trace=False):
    if trace:
        tracemalloc.start()
    cpu, wall = os.times(), time.perf_counter()
    func()
    wall = time.perf_counter() - wall
    after = os.times()
    cpu = (after.user - cpu.user) + (after.system - cpu.system)
    memory = ""
    if trace:
        memory = f"  traced peak {tracemalloc.get_traced_memory()[1] / 1e9:6.2f} GB"
        tracemalloc.stop()
    print(f"[TIME] {label:12s} wall {wall:7.2f} s  cpu {cpu:7.2f} s  cores {cpu / wall:5.2f}{memory}", flush=True)


def without_figures(module, sdata, figure_path, tmp, modality, imagedim, dim_x, dim_y, threads, **kwargs):
    extra = () if module is ref else (threads,)
    with contextlib.ExitStack() as stack:
        for owner, name in ((helperfuncs, "plot_pixels"), (plt, "imshow"), (figures, "imshow"), (plt, "plot"), (module, "save_figure")):
            stack.enter_context(mock.patch.object(owner, name, lambda *a, **k: None))
        module.define_bounding_boxes(sdata, figure_path, tmp, modality, IMAGE_TYPE, RESOLUTION, dim_x, dim_y,
                                     imagedim, "raw", *extra, **kwargs)


def main(path, mask_path, threads, crop):
    assert bounding_boxes.__file__.startswith(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    dask.config.set(scheduler="threads", num_workers=threads)
    full = sd.read_zarr(path)
    full_shape = full[IMAGE_TYPE][RESOLUTION].image.shape[1:]
    full.points["transcripts"] = PointsModel.parse(
        helperfuncs.deduplicate_dask_index(full.points["transcripts"])
    )
    cropped, _, _ = helperfuncs.image_crop(
        full, START, START, START + crop, START + crop, "global"
    )
    sdata = sd.SpatialData(
        images={IMAGE_TYPE: cropped[IMAGE_TYPE]},
        points={"transcripts": cropped["transcripts"]},
        shapes={"nucleus_boundaries": cropped["nucleus_boundaries"]},
    )
    image = sdata[IMAGE_TYPE][RESOLUTION].image
    dim_x, dim_y = len(image.y.values), len(image.x.values)
    extent = sd.get_extent(sdata[IMAGE_TYPE], coordinate_system="global")
    imagedim = helperfuncs.ImageDimStruct(
        extent["x"][0], extent["y"][0], extent["x"][1], extent["y"][1]
    )
    # The flipped image's row r is full-image row H-1-r, so the crop's flipped rows are these.
    rows = slice(full_shape[0] - START - dim_x, full_shape[0] - START)
    mask = crop_mask(mask_path, full_shape, rows, slice(START, START + dim_y))
    print(
        f"[NOTE] crop {dim_x}x{dim_y}, mask foreground {mask.mean():.3f}, threads {threads}",
        flush=True,
    )

    with tempfile.TemporaryDirectory() as tmp:
        for prefix in ("hqpr_0", "hqtr"):
            write_mask(tmp, prefix, mask, npartitions=-(-mask.size // 1_000_000))
        write_density(
            tmp,
            ref.generate_transcript_density_image(
                sdata, None, imagedim, IMAGE_TYPE, RESOLUTION
            ).reshape(dim_x, dim_y),
        )
        for modality, kwargs in (
            ("hqpr", dict(staining="0")),
            ("hqtr", dict(dilation_radius=1)),
        ):
            args = (sdata, f"{tmp}/fig", tmp, modality, imagedim, dim_x, dim_y, threads)
            print(f"== {modality} {kwargs}", flush=True)
            expected = run_define(ref, *args, **kwargs)
            assert_same(expected, run_define(bounding_boxes, *args, **kwargs))
            print(f"[OK] {modality}: {len(expected[0])} boxes, {len(expected[2])} figure calls identical", flush=True)
            # Timings with the figure calls stubbed out (figure writing is core/figures' cost).
            for run in (1, 2, 3):
                for name, module in (("original", ref), ("new", bounding_boxes)):
                    measured(f"{name} #{run}", lambda: without_figures(module, *args, **kwargs))
            for name, module in (("original", ref), ("new", bounding_boxes)):
                measured(f"{name} mem", lambda: without_figures(module, *args, **kwargs), trace=True)


if __name__ == "__main__":
    main(
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        int(sys.argv[4]) if len(sys.argv) > 4 else 4000,
    )
