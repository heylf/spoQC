"""Real-data check (not collected by pytest): hqcr clustering vs the verbatim original.

Usage:
    python tests/realdata_hqcr_clustering.py <spatialdata.zarr> <tmp_folder> [rows] [threads]

Builds hqcr's clustering input the way `spoqc -s all` does (table X = normlog, the 13 QC
columns written through the [:, :13] view, obs from the run's *_hqcr.parquet files), keeps the
first `rows` cells (default 40,000: the original's compress_index() allocates
(n_nodes, 2, n_obs) float32 on sparse input, 24.7 GB for all 167,780 cells), and runs the
original and the new clustering on it. Asserts exact equality of the kNN graph, .uns and labels.
"""

import sys
import time
import types

from spoqc.core import threads

THREADS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
threads.configure(THREADS)

import anndata as ad
import numpy as np

from spoqc.general import normalizations
from spoqc.subworkflows import hqcr
from test_hqcr_clustering_differential import (
    assert_identical,
    original_clustering_for_hqcr,
    outputs,
)

zarr_path, tmp_folder = sys.argv[1], sys.argv[2]
rows = int(sys.argv[3]) if len(sys.argv) > 3 else 40_000


def clustering_input():
    adata = ad.read_zarr(f"{zarr_path}/tables/table")
    adata.obs.index = [int(i) for i in range(adata.n_obs)]  # cli.py
    adata.obs.index = adata.obs.index.astype(str)
    adata.obs.index.name = "index"
    adata.layers["raw"] = adata.X
    sdata = {"table": adata}
    normalizations.transform_normalize_sc_data(sdata, 5000, 1.0)
    normalizations.fill_nans_for_0_transcript_cells(sdata)
    adata.X = adata.layers["normlog"]  # qc_ambient
    qc, _, _ = hqcr.load_data_for_hqcr(sdata, tmp_folder, "canorm_transcript_counts")
    return qc[:rows].copy()


results = {}
for name, fn in (
    ("original", original_clustering_for_hqcr),
    ("new", hqcr.clustering_for_hqcr),
):
    adata = clustering_input()
    np.random.seed(123)
    t0 = time.perf_counter()
    fn(adata, "unused", types.SimpleNamespace(THREADS=THREADS), 123)
    print(
        f"{name}: {time.perf_counter() - t0:.1f} s on {adata.n_obs} cells, {adata.obs['leiden'].nunique()} clusters"
    )
    results[name] = outputs(adata)
assert_identical(results["new"], results["original"])
print(
    "identical: kNN distances and connectivities, .uns['neighbors'], Leiden labels, global RNG state"
)
