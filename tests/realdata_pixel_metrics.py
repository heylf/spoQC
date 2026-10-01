"""Real-data check (not collected by pytest): structural image analysis, original vs new, on a crop.

Usage:
    PYTHONPATH=<repo>:<repo>/tests python tests/realdata_pixel_metrics.py <crop.npy> <threads>

<crop.npy> is a 2D uint16 morphology image crop (e.g. 4000 x 4000 of a Xenium morphology_focus s0).
Runs start_image_struc_analyis for hqpr on the crop and for hqtr on an int64 density-like image
derived from it (crop // 64), asserts that every written parquet and plotted array is byte-identical,
then times each metric kernel old vs new and prints wall time and cores busy (CPU time / wall).
"""

import os
import sys
import tempfile
import time

import cv2
import numba
import numpy as np
from pytest import MonkeyPatch

import reference_pixel_metrics as reference
from spoqc.image_analysis._slidingwindow import texture_metrics
from spoqc.metrics.image import pixel_metrics
from test_pixel_metrics_differential import compare, float64_reference_texture, float64_texture, record_calls


def timed(label, func):
    cpu, wall = os.times(), time.perf_counter()
    func()
    wall = time.perf_counter() - wall
    after = os.times()
    cpu = (after.user - cpu.user) + (after.system - cpu.system)
    print(
        f"[TIME] {label:34s} wall {wall:7.3f} s  cpu {cpu:7.3f} s  cores busy {cpu / wall:5.2f}",
        flush=True,
    )
    return wall


def main(crop_path, threads):
    assert pixel_metrics.__file__.startswith(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    numba.set_num_threads(threads)
    cv2.setNumThreads(threads)
    crop = np.load(crop_path)
    density = (crop // 64).astype(np.int64)
    for modality, image in [("hqpr", crop), ("hqtr", density)]:
        with tempfile.TemporaryDirectory() as tmp, MonkeyPatch.context() as monkeypatch:
            compare(image, modality, tmp, monkeypatch)
        print(
            f"[EXACT] {modality}: all parquets and plotted arrays byte-identical",
            flush=True,
        )

    xy = np.flipud(crop)
    tex = np.floor(
        (xy.astype(np.float64) - xy.min()) / max(xy.max() - xy.min(), 1) * 255
    ).astype(np.uint8)
    background = 91.0
    for name, a, b in zip(["entropy", "kl", "homogeneity"], float64_texture(tex), float64_reference_texture(tex)):
        assert a.tobytes() == b.tobytes(), name
    print("[EXACT] texture window sums byte-identical at float64", flush=True)
    with MonkeyPatch.context() as monkeypatch:
        record_calls(monkeypatch)
        texture_metrics(tex[:20, :20], 5)
        pixel_metrics.lbp(xy[:20, :20], 100, 3)
        reference.pixel_entropy("", tex[:20, :20], 5, None)
        reference.pixel_uniformity("", tex[:20, :20], 5, None)
        reference.pixel_homogeneity("", tex[:20, :20], None, 5)
        old, new = {}, {}
        old["intensity"] = timed(
            "old intensity (snr)", lambda: np.log2((xy.flatten() + 1) / background)
        )
        new["intensity"] = timed(
            "new intensity (snr)",
            lambda: pixel_metrics.signal_noise_ratio(xy, background),
        )
        old["lbp"] = timed("old lbp", lambda: reference.pixel_lbp("", xy, 100, 3, None))
        new["lbp"] = timed("new lbp", lambda: pixel_metrics.lbp(xy, 100, 3))
        old["edge_strength"] = timed(
            "old edge_strength", lambda: reference.pixel_edge_strength("", xy, None)
        )
        new["edge_strength"] = timed(
            "new edge_strength", lambda: pixel_metrics.edge_strength(xy)
        )
        old["energy"] = timed(
            "old energy", lambda: reference.pixel_energy("", xy, 5, None)
        )
        new["energy"] = timed(
            "new energy", lambda: pixel_metrics.energy(xy, 5)
        )
        old["relevance"] = timed(
            "old relevance", lambda: reference.pixel_relevance("", xy, background, None)
        )
        new["relevance"] = timed(
            "new relevance", lambda: pixel_metrics.relevance(xy, background)
        )
        old["texture"] = timed(
            "old entropy+uniformity+homogeneity",
            lambda: (
                reference.pixel_entropy("", tex, 5, None),
                reference.pixel_uniformity("", tex, 5, None),
                reference.pixel_homogeneity("", tex, None, 5),
            ),
        )
        new["texture"] = timed(
            "new texture (one pass)", lambda: texture_metrics(tex, 5)
        )
    print(
        f"[TIME] total old {sum(old.values()):.3f} s, new {sum(new.values()):.3f} s",
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
