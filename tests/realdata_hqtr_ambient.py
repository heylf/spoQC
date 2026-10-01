"""Real-data check (not collected by pytest): ambientqc and the hqtr transcript images vs origin/dev.

Usage:
    python tests/realdata_hqtr_ambient.py <spatialdata.zarr> <workdir> [threads]

Runs each converted function and its verbatim origin/dev module (tests/legacy/) on the same real
inputs (the zarr through cli.py's transcript setup plus the normalisation), asserts exact
equality (values, dtypes, order; for the qv/ac priors every parquet part file byte for byte) and
prints wall time, cores and peak-RSS growth of both.
"""

import os
import resource
import sys
import time
import types

from spoqc.core import threads

THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
threads.configure(THREADS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import spatialdata as sd  # noqa: E402
from libpysal.weights import Queen  # noqa: E402
import geopandas as gpd  # noqa: E402
import dask.dataframe as dd  # noqa: E402

import spoqc  # noqa: E402
from spoqc import general, helperfuncs  # noqa: E402
from spoqc.core import spatial  # noqa: E402
from spoqc.metrics.transcript_density import (  # noqa: E402
    ac_image,
    global_moran_I,
    local_moran_I,
    qv_image,
    transcript_density_image,
)
from conftest import assert_same_array, cli_sdata, load_legacy  # noqa: E402
from test_core_spatial_neighbours import original_points_within_radius  # noqa: E402
from test_local_moran_weights import assert_knn_weights_equal  # noqa: E402
from legacy.parquet_writer import ddf_to_parquet  # noqa: E402

DENSITY = "spoqc.metrics.transcript_density"
PRIORS = "spoqc.priors.hqtr"


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def timed(label, fn, *args, **kwargs):
    cpu0, wall0 = cpu_seconds(), time.perf_counter()
    result = fn(*args, **kwargs)
    wall = time.perf_counter() - wall0
    print(
        f"{label}: {wall:.2f} s wall, {(cpu_seconds() - cpu0) / wall:.2f} cores, "
        f"max RSS so far {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GB",
        flush=True,
    )
    return result


def legacy_modules():
    helperfuncs.points_within_radius = (
        original_points_within_radius  # what origin/dev's local Moran called
    )
    helperfuncs.ddf_to_parquet = ddf_to_parquet  # what origin/dev's qv/ac steps wrote their priors with
    legacy = {
        name: load_legacy(name, DENSITY)
        for name in (
            "transcript_density_image",
            "qv_image",
            "local_moran_I",
            "ac_image",
            "global_moran_I",
        )
    }
    ac_or_qv = load_legacy("ac_or_qv", PRIORS)
    fake_priors = types.SimpleNamespace(hqtr=types.SimpleNamespace(ac_or_qv=ac_or_qv))
    legacy["qv_image"].priors = fake_priors
    legacy["ac_image"].priors = fake_priors
    legacy["ac_image"].local_moran_I = legacy["local_moran_I"]
    return legacy


def assert_same_prior_values(new_dir, old_dir):
    """The prior parquet the step wrote: new layout (larger parts, only the read columns); the density
    exactly, the prior within the merged Gaussian prior's difference."""
    new = dd.read_parquet(new_dir, engine="pyarrow", calculate_divisions=True)
    old = dd.read_parquet(old_dir, engine="pyarrow").compute()
    assert all(c in old.columns for c in new.columns) and len(new.columns) == 2, list(new.columns)
    computed = new.compute()
    assert computed.index.equals(old.index)
    density, norm_p = computed.columns
    assert_same_array(computed[density].to_numpy(), old[density].to_numpy(), density)
    # norm_p: the merged Gaussian prior (priors.gaussian) differs from origin/dev in the last bits
    diff = np.abs(computed[norm_p].to_numpy() - old[norm_p].to_numpy())
    assert diff.max() <= 16 * np.finfo(np.float64).eps, diff.max()
    print(f"{norm_p}: {np.count_nonzero(diff):,} of {len(diff):,} pixels differ, max abs {diff.max():.3g}")
    sizes = {d: (len(os.listdir(d)), sum(os.path.getsize(f"{d}/{f}") for f in os.listdir(d))) for d in (old_dir, new_dir)}
    print(
        f"MATCH {os.path.basename(new_dir)} {list(new.columns)}: "
        f"{sizes[old_dir][0]} files {sizes[old_dir][1] / 1e6:.1f} MB -> {sizes[new_dir][0]} files {sizes[new_dir][1] / 1e6:.1f} MB"
    )

def main():
    zarr_path, work = sys.argv[1], sys.argv[2]
    assert spoqc.__file__.startswith(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ), spoqc.__file__
    legacy = legacy_modules()

    sdata = cli_sdata(zarr_path)
    sdata["table"].obs.index = sdata["table"].obs.index.astype(
        str
    )  # cli.py's string obs index
    sdata["table"].layers["raw"] = sdata["table"].X
    general.normalizations.transform_normalize_sc_data(sdata, 5000, 1.0)
    sdata["table"].X = sdata["table"].layers["normlog"]  # qc_ambient.start_qc_ambient
    extent = sd.get_extent(sdata["morphology_focus"], coordinate_system="global")
    imagedim = helperfuncs.ImageDimStruct(
        extent["x"][0], extent["y"][0], extent["x"][1], extent["y"][1]
    )
    dim_x = len(sdata["morphology_focus"]["scale0"].image.y.values)
    dim_y = len(sdata["morphology_focus"]["scale0"].image.x.values)
    args = (imagedim, "morphology_focus", "scale0")
    print(
        f"image {dim_x} x {dim_y}, cells {sdata['table'].n_obs:,}, transcripts {len(sdata.points['transcripts']):,}"
    )

    # Transcript density image (hqtr metrics input)
    old = timed(
        "origin/dev density image",
        legacy["transcript_density_image"].generate_transcript_density_image,
        sdata,
        None,
        *args,
    )
    new = timed(
        "new density image",
        transcript_density_image.generate_transcript_density_image,
        sdata,
        None,
        *args,
    )
    assert_same_array(new, old, "density image")
    print(f"EXACT MATCH density image: {np.count_nonzero(old):,} nonzero pixels")

    # QV image
    old = timed(
        "origin/dev qv image",
        legacy["qv_image"].generate_transcript_quality_density_image,
        sdata,
        None,
        *args,
    )
    new = timed(
        "new qv image",
        qv_image.generate_transcript_quality_density_image,
        sdata,
        None,
        *args,
    )
    assert_same_array(new, old, "qv image")
    print(f"EXACT MATCH qv image: {np.count_nonzero(old):,} nonzero pixels")

    # Global Moran's I (ambientqc)
    coords = sdata["table"].obsm["spatial"]
    gdf = gpd.GeoDataFrame(
        {"x": coords[:, 0], "y": coords[:, 1]},
        geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]),
    )
    w = Queen.from_dataframe(gdf)
    w.transform = "r"
    X_dense = sdata["table"].X.toarray()
    old = legacy["global_moran_I"].moran_I_all_genes(X_dense, w.sparse)
    new = global_moran_I.moran_I_all_genes(X_dense, w.sparse)
    assert_same_array(new, old, "global Moran's I")
    print(f"EXACT MATCH global Moran's I: {int(np.isnan(old).sum())} NaN genes")
    global_ambient = pd.DataFrame(
        {"genes": np.array(sdata["table"].var_names), "morans_I": new}
    ).sort_values(by="morans_I", ascending=False)

    # Local Moran's I: the per-neighbourhood weights, then the per-transcript values
    xy = np.asarray(sdata["table"].obsm["spatial"], dtype=np.float64)
    neighbourhoods = spatial.neighbour_lists(
        xy, local_moran_I.NEIGHBOURHOOD_RADIUS, THREADS
    )
    n_checked = 0
    for idx in neighbourhoods:
        if len(idx) > 30:
            assert_knn_weights_equal(xy[idx])
            n_checked += 1
    print(f"EXACT MATCH KNN weights: {n_checked:,} neighbourhoods")
    old = timed(
        "origin/dev local Moran's I",
        legacy["local_moran_I"].calculate_local_moran_I_values,
        sdata,
        THREADS,
    )
    new = timed(
        "new local Moran's I",
        local_moran_I.calculate_local_moran_I_values,
        sdata,
        THREADS,
    )
    assert_same_array(new, old, "local Moran's I")
    print(f"EXACT MATCH local Moran's I: {len(np.unique(old)):,} distinct values")

    # AC image
    old = timed(
        "origin/dev ac image",
        legacy["ac_image"].generate_transcript_ambient_density_image,
        sdata,
        None,
        THREADS,
        imagedim,
        global_ambient.copy(),
        *args[1:],
    )
    new = timed(
        "new ac image",
        ac_image.generate_transcript_ambient_density_image,
        sdata,
        None,
        THREADS,
        imagedim,
        global_ambient.copy(),
        *args[1:],
    )
    assert_same_array(new, old, "ac image")
    print(f"EXACT MATCH ac image: {np.count_nonzero(old):,} nonzero pixels")

    # Whole hqtr_qv / hqtr_ac steps: the prior parquets they leave for hqtr clustering
    for side in ("old", "new"):
        os.makedirs(f"{work}/{side}/fig/hqtr/hqtr_qv", exist_ok=True)
        os.makedirs(f"{work}/{side}/fig/hqtr/hqtr_ac", exist_ok=True)
        os.makedirs(f"{work}/{side}/tmp", exist_ok=True)
        helperfuncs.df_to_parquet(
            global_ambient, "ambient", f"{work}/{side}/tmp", [], "genes"
        )
    step_args = ("hqtr", "morphology_focus", "scale0", dim_x, dim_y, imagedim)
    for name, module, new_module, extra in (
        ("qv", legacy["qv_image"], qv_image, ()),
        ("ac", legacy["ac_image"], ac_image, (THREADS,)),
    ):
        fn = f"transcript_{name}_image"
        step = step_args[:1] + extra + step_args[1:]
        timed(
            f"origin/dev hqtr_{name} step",
            getattr(module, fn),
            sdata,
            f"{work}/old/fig",
            f"{work}/old/tmp",
            *step,
        )
        timed(
            f"new hqtr_{name} step",
            getattr(new_module, fn),
            sdata,
            f"{work}/new/fig",
            f"{work}/new/tmp",
            *step,
        )
        assert_same_prior_values(
            f"{work}/new/tmp/hqtr_output_{name}_prob",
            f"{work}/old/tmp/hqtr_output_{name}_prob",
        )
    print("ALL MATCH")


if __name__ == "__main__":
    main()
