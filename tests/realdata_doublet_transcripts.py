"""Real-data check (not collected by pytest): KD-tree transcript/doublet proximity vs the original loop.

Usage:
    python tests/realdata_doublet_transcripts.py <points.parquet dir> [n_doublets] [n_reference_doublets]

Reads the transcript x/y exactly as spatialdata stores them (float32), builds ovrlpy-style
doublets (int64 grid coordinates + builtin-min origin, sampled from real transcript
locations with jitter), runs the verbatim original loop on `n_reference_doublets` of them
against ALL transcripts, and asserts exact equality with the new code on the same doublets.
Then times the new code on all `n_doublets` at 1 and 4 threads and reports cores used.
"""

import resource
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from spoqc.core.spatial import _search_pad
from spoqc.metrics.segmentation.doublet_score import flag_transcripts_near_doublets
from test_doublet_transcripts import DISTANCE_THRESH, original_loop

JITTER = 4  # grid units; most jittered doublets still sit on dense tissue


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def timed(fn, *args):
    cpu0, wall0 = cpu_seconds(), time.perf_counter()
    result = fn(*args)
    wall = time.perf_counter() - wall0
    return result, wall, (cpu_seconds() - cpu0) / wall


def main(points_dir, n_doublets=1500, n_reference=1500):
    table = ds.dataset(points_dir).to_table(columns=["x", "y", "__null_dask_index__"])
    transcripts = pd.DataFrame(
        {"x": table["x"].to_numpy(), "y": table["y"].to_numpy()},
        index=table["__null_dask_index__"].to_numpy(),
    )
    del table
    print(f"transcripts: {len(transcripts):,}  dtype x={transcripts['x'].dtype}")

    rng = np.random.default_rng(0)
    min_x, min_y = min(transcripts["x"]), min(transcripts["y"])
    picks = rng.choice(len(transcripts), n_doublets, replace=False)
    doublet_df = pd.DataFrame(
        {
            "x": np.rint(transcripts["x"].to_numpy()[picks] - min_x).astype(np.int64)
            + rng.integers(-JITTER, JITTER + 1, n_doublets),
            "y": np.rint(transcripts["y"].to_numpy()[picks] - min_y).astype(np.int64)
            + rng.integers(-JITTER, JITTER + 1, n_doublets),
            "integrity": rng.random(n_doublets).astype(np.float32),
            "signal": (rng.random(n_doublets) * 10).astype(np.float32),
        }
    )
    corrected = doublet_df.copy()
    corrected["x"] = doublet_df["x"] + min_x
    corrected["y"] = doublet_df["y"] + min_y
    reference = corrected.iloc[rng.choice(n_doublets, n_reference, replace=False)]

    (ref_d, ref_w), ref_wall, ref_cores = timed(
        original_loop, transcripts, reference, DISTANCE_THRESH
    )
    (new_d, new_w), new_wall, _ = timed(
        flag_transcripts_near_doublets, transcripts, reference, DISTANCE_THRESH, 4
    )
    assert ref_d.dtype == new_d.dtype and ref_w.dtype == new_w.dtype
    assert np.array_equal(ref_d, new_d) and np.array_equal(ref_w, new_w)
    print(
        f"EXACT MATCH on {n_reference} doublets x {len(transcripts):,} transcripts: "
        f"{int(ref_d.sum()):,} transcripts flagged; dtypes {new_d.dtype}/{new_w.dtype}"
    )
    per_doublet = ref_wall / n_reference
    print(
        f"original: {ref_wall:.1f} s for {n_reference} doublets = {per_doublet:.3f} s/doublet "
        f"(cores {ref_cores:.2f}); extrapolated to {n_doublets}: {per_doublet * n_doublets:.0f} s"
    )

    # float32 formula vs the exact distance between the float32-cast points the KD-tree holds:
    # the relative error must stay under the relative search pad.
    tx, ty = transcripts["x"], transcripts["y"]
    worst = 0.0
    for _, doublet in reference.iloc[:50].iterrows():
        x1, y1 = doublet["x"], doublet["y"]
        near = np.flatnonzero(
            (np.abs(tx.to_numpy() - x1) <= DISTANCE_THRESH + 1)
            & (np.abs(ty.to_numpy() - y1) <= DISTANCE_THRESH + 1)
        )
        f32 = np.sqrt((tx.iloc[near] - x1) ** 2 + (ty.iloc[near] - y1) ** 2).to_numpy()
        f64 = np.hypot(
            tx.to_numpy()[near].astype(np.float64) - float(np.float32(x1)),
            ty.to_numpy()[near].astype(np.float64) - float(np.float32(y1)),
        )
        nonzero = f64 > 0
        worst = max(worst, float((np.abs(f32 - f64)[nonzero] / f64[nonzero]).max(initial=0.0)))
    print(
        f"max relative |formula - exact| over near pairs (50 doublets): {worst:.2e} "
        f"(relative search pad {_search_pad(np.float32):.2e})"
    )
    assert worst < _search_pad(np.float32)

    for threads in (1, 4):
        _, wall, cores = timed(
            flag_transcripts_near_doublets,
            transcripts,
            corrected,
            DISTANCE_THRESH,
            threads,
        )
        print(
            f"new, {n_doublets} doublets, threads={threads}: {wall:.2f} s, cores used {cores:.2f}, "
            f"speedup vs extrapolated original {per_doublet * n_doublets / wall:.0f}x"
        )


if __name__ == "__main__":
    main(sys.argv[1], *map(int, sys.argv[2:]))
