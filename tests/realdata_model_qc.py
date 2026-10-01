"""Real-data check (not collected by pytest): Model QC's Moran's I loop vs origin/dev's.

Usage:
    python tests/realdata_model_qc.py <table.h5ad> [n_pcs] [threads]

<table.h5ad> is spoQC's table after cli.run's normalisation (layer 'normlogscale').
Runs sc.tl.pca as run_qc_model does, then for the first n_pcs PCs origin/dev's loop
(fresh Queen weights per PC, esda Moran with 999 permutations) and the new one
(weights once, core.moran.moran), from the same seed. Asserts every Moran attribute
and the final global RNG state are identical, and prints both walls.
"""

import sys

from spoqc.core import threads

threads.configure(int(sys.argv[3]) if len(sys.argv) > 3 else 4)

import time  # noqa: E402

import anndata as ad  # noqa: E402
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scanpy as sc  # noqa: E402
from esda.moran import Moran  # noqa: E402
from libpysal.weights import Queen  # noqa: E402

from spoqc.core import moran as core_moran  # noqa: E402
from test_moran_equivalence import assert_same_moran, assert_same_state  # noqa: E402

adata = ad.read_h5ad(sys.argv[1])
n_pcs = int(sys.argv[2]) if len(sys.argv) > 2 else 60
adata.X = adata.layers["normlogscale"]
sc.tl.pca(adata, n_comps=100)
df = pd.DataFrame({"x": adata.obsm["spatial"][:, 0], "y": adata.obsm["spatial"][:, 1]})
for i in range(n_pcs):
    df[f"PC{i}"] = adata.obsm["X_pca"][:, i]

np.random.seed(123)
t = time.time()
ref = []
for i in range(n_pcs):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    ref.append(Moran(df["PC" + str(i)], Queen.from_dataframe(gdf), permutations=999))
t_ref = time.time() - t
ref_state = np.random.get_state()

np.random.seed(123)
t = time.time()
w = core_moran.queen_weights(df[["x", "y"]].to_numpy())
new = [core_moran.moran(df["PC" + str(i)], w, permutations=999) for i in range(n_pcs)]
t_new = time.time() - t

assert_same_state(ref_state, np.random.get_state())
for r, n in zip(ref, new):
    assert_same_moran(r, n)
print(
    f"identical over {n_pcs} PCs x 999 permutations on {len(df)} cells; "
    f"origin/dev loop {t_ref:.1f} s, new {t_new:.1f} s"
)
