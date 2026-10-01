"""Real-data check (not collected by pytest): core.spatial callers vs their verbatim originals.

Usage:
    python tests/realdata_spatial_neighbours.py <spatialdata.zarr> <case> [threads]

Cases: radius20, radius30, radius100 (points_within_radius callers; radius30 also runs
the hqcr bad-quality probabilities), island, border, moran_outside, doublet_cells.
Each case runs the verbatim original and the new code on the same real inputs, asserts
exact equality (values, dtypes, order) and prints wall time and cores used for both.
"""

import os
import resource
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import zarr

from spoqc.core import spatial
from spoqc.metrics.segmentation import border_score, doublet_score, island_score
from spoqc.metrics.transcript_density import local_moran_I
from spoqc.priors.hqcr import transcript_and_gene_counts
from test_core_spatial_neighbours import (
    original_border_scores,
    original_cells_near_doublets,
    original_fill_outside,
    original_get_bad_quality_probability,
    original_island_scores,
    original_points_within_radius,
)


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def timed(label, fn, *args):
    cpu0, wall0 = cpu_seconds(), time.perf_counter()
    result = fn(*args)
    wall = time.perf_counter() - wall0
    print(
        f"{label}: {wall:.2f} s wall, {(cpu_seconds() - cpu0) / wall:.2f} cores",
        flush=True,
    )
    return result


def assert_same(expected, actual):
    if isinstance(expected, list):
        assert actual == expected
        assert all(type(v) is int for lst in actual for v in lst)
    else:
        np.testing.assert_array_equal(actual, expected, strict=True)


def cells_xy(zarr_path):
    return zarr.open(f"{zarr_path}/tables/table/obsm/spatial", mode="r")[:]


def margin_report(xy, radius, threads):
    """Distance of the closest candidate pair to the radius: how far this data is from any tie at r."""
    q, r = spatial.pairs_within(xy, xy, radius * 1.01, threads, exclude_self=True)
    d = np.sqrt((xy[r, 0] - xy[q, 0]) ** 2 + (xy[r, 1] - xy[q, 1]) ** 2)
    gap = np.abs(d - radius)
    print(
        f"pairs within 1.01 r: {len(q):,}; exactly at r: {int((gap == 0).sum())}; "
        f"min |d - r| = {gap.min():.3e} (float64 ULP at r: {np.spacing(float(radius)):.1e})"
    )


def case_radius(zarr_path, radius, threads):
    xy = cells_xy(zarr_path)
    df_coords = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]})
    print(f"cells: {len(xy):,} dtype {xy.dtype}, radius {radius}")
    expected = timed(
        "original points_within_radius",
        original_points_within_radius,
        df_coords,
        radius,
        False,
    )
    actual = timed(
        f"new neighbour_lists threads={threads}",
        spatial.neighbour_lists,
        xy,
        radius,
        threads,
    )
    assert_same(expected, actual)
    print(
        f"EXACT MATCH neighbour lists r={radius}: {sum(map(len, actual)):,} neighbours"
    )
    margin_report(xy, radius, threads)
    if radius == 30:
        rng = np.random.default_rng(0)
        cell_df = pd.DataFrame({"qc_cluster": rng.integers(0, 3, len(xy))})
        bad = np.int64(0)
        cpu0, wall0 = cpu_seconds(), time.perf_counter()
        exp = np.array(
            [
                original_get_bad_quality_probability(x, cell_df, expected, bad, "qc_cluster")
                for x in range(len(xy))
            ]
        )
        wall = time.perf_counter() - wall0
        print(
            f"original get_bad_quality_probability loop (after the radius scan): {wall:.2f} s wall, "
            f"{(cpu_seconds() - cpu0) / wall:.2f} cores"
        )
        act = timed(
            f"new get_bad_quality_probabilities threads={threads} (incl. search)",
            transcript_and_gene_counts.get_bad_quality_probabilities,
            df_coords,
            cell_df["qc_cluster"].to_numpy(),
            bad,
            threads,
        )
        assert_same(exp, act)
        print("EXACT MATCH hqcr bad-quality probabilities")


def case_island(zarr_path, threads):
    xy = cells_xy(zarr_path)
    exp_idx, exp_score = timed("original KDTree + BFS", original_island_scores, xy, 15)
    act_idx = timed(
        f"new find_connected_groups threads={threads}",
        island_score.find_connected_groups,
        xy,
        15,
        threads,
    )
    assert_same(exp_idx, act_idx)
    assert_same(exp_score, np.bincount(act_idx)[act_idx])
    print(f"EXACT MATCH island: {exp_idx.max() + 1:,} groups")
    margin_report(xy, 15, threads)


def case_border(zarr_path, threads):
    xy = cells_xy(zarr_path)
    expected = timed(
        "original per-point border scores (serial)", original_border_scores, xy, 50, 10
    )
    actual = timed(
        f"new get_border_scores threads={threads}",
        border_score.get_border_scores,
        xy,
        50,
        10,
        threads,
    )
    assert_same(expected, actual)
    print(f"EXACT MATCH border: {len(np.unique(actual))} distinct scores")
    margin_report(xy, 50, threads)


def case_moran_outside(zarr_path, threads):
    table = ds.dataset(f"{zarr_path}/points/transcripts/points.parquet").to_table(
        columns=["x", "y", "feature_name", "cell_id"]
    )
    coords = np.column_stack(
        (
            table["x"].to_numpy().astype(np.float64),
            table["y"].to_numpy().astype(np.float64),
        )
    )
    feat = table["feature_name"].to_pandas().to_numpy()
    outside = table["cell_id"].to_numpy() == -1
    del table
    rng = np.random.default_rng(0)
    local_I = np.where(outside, -1.0, rng.random(len(coords))).astype(np.float32)
    print(
        f"transcripts: {len(coords):,}, outside cells: {int(outside.sum()):,}, genes {len(np.unique(feat))}"
    )
    cache = os.environ.get("SPOQC_REALDATA_CACHE")  # reuse a previous run's original output
    if cache and os.path.exists(f"{cache}/moran_outside_original.npy"):
        expected = np.load(f"{cache}/moran_outside_original.npy")
        print("original per-feature cKDTree: loaded from cache")
    else:
        expected = timed(
            "original per-feature cKDTree",
            original_fill_outside,
            coords,
            feat,
            local_I.copy(),
            outside,
        )
        if cache:
            np.save(f"{cache}/moran_outside_original.npy", expected)
    actual = timed(
        f"new fill_outside_from_nearest_inside threads={threads}",
        local_moran_I.fill_outside_from_nearest_inside,
        coords,
        feat,
        local_I.copy(),
        outside,
        threads,
    )
    assert_same(expected, actual)
    print(
        f"EXACT MATCH Moran outside transcripts: {int((actual[outside] != 0).sum()):,} filled from a neighbour"
    )


def case_doublet_cells(zarr_path, threads, n_doublets=1500, jitter=4):
    xy = cells_xy(zarr_path)
    rng = np.random.default_rng(0)
    # ovrlpy-style doublets: int64 grid coordinates + builtin-min origin, near real cells
    min_x, min_y = min(pd.Series(xy[:, 0])), min(pd.Series(xy[:, 1]))
    picks = rng.choice(len(xy), n_doublets, replace=False)
    doublet_df = pd.DataFrame(
        {
            "x": np.rint(xy[picks, 0] - min_x).astype(np.int64)
            + rng.integers(-jitter, jitter + 1, n_doublets),
            "y": np.rint(xy[picks, 1] - min_y).astype(np.int64)
            + rng.integers(-jitter, jitter + 1, n_doublets),
            "integrity": rng.random(n_doublets).astype(np.float32),
            "signal": (rng.random(n_doublets) * 10).astype(np.float32),
        }
    )
    corrected = doublet_df.copy()
    corrected["x"] = doublet_df["x"] + min_x
    corrected["y"] = doublet_df["y"] + min_y
    expected = timed(
        "original doublet x cell loop", original_cells_near_doublets, xy, corrected, 10
    )
    actual = timed(
        f"new flag_cells_near_doublets threads={threads}",
        doublet_score.flag_cells_near_doublets,
        xy,
        corrected[["x", "y"]].to_numpy(),
        10,
        threads,
    )
    for exp, act in zip(expected, actual):
        assert_same(exp, act)
    print(f"EXACT MATCH doublet cells: {int(expected[0].sum()):,} cells flagged")


if __name__ == "__main__":
    path, case = sys.argv[1], sys.argv[2]
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    if case.startswith("radius"):
        case_radius(path, int(case[len("radius") :]), threads)
    else:
        globals()[f"case_{case}"](path, threads)
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GB")
