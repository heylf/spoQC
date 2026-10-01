"""Real-data check (not collected by pytest): what origin/dev's hqcr view write changed downstream.

Usage:
    python tests/realdata_hqcr_view_write.py <spatialdata.zarr> <tmp_folder> [threads]

origin/dev's load_data_for_hqcr set X on a view of sdata['table'], which wrote the 13 QC
columns into sdata['table'].X, the same object as layers['normlog'] (set by qc_ambient).
This builds the table as `spoqc -s all` has it when hqcr starts, runs the verbatim original
and the fixed load_data_for_hqcr on two copies, then:
1. asserts the clustering input is identical (so hqcr's own outputs are unchanged);
2. reports how the table's X / normlog / raw / normlogscale differ after hqcr;
3. runs the computations of the later steps that read them (cell cycle QC reads X; model QC
   reads normlogscale) and reports the differences.
"""

import sys

from spoqc.core import threads

THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
threads.configure(THREADS)

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from spoqc import helperfuncs
from spoqc.general import normalizations
from spoqc.subworkflows import hqcr, qc_cellcycle

zarr_path, tmp_folder = sys.argv[1], sys.argv[2]


def original_load_data_for_hqcr(sdata, spoqc_tmp_folder, counts):
    """Verbatim from spoQC origin/dev db00d98 (subworkflows/hqcr.py), load_cell_df qualified."""
    print("[NOTE] Gather cell QC metrices")
    helperfuncs.read_sdata_parquet_tmp_files(sdata, spoqc_tmp_folder, "hqcr")
    qc_domains_adata = sdata["table"]

    cell_df = hqcr.load_cell_df(counts, sdata)

    # Check for NaNs
    print("[DEBUG] Number of NaNs in each column:")
    print(cell_df.isna().sum())

    # This I have to do to avoid an error because of the number of features I have selected.
    qc_metrices = list(cell_df.columns)
    qc_domains_adata = qc_domains_adata[:, 0 : len(qc_metrices)]
    qc_domains_adata.X = cell_df

    return qc_domains_adata, cell_df, qc_metrices


def table_at_hqcr():
    adata = ad.read_zarr(f"{zarr_path}/tables/table")
    adata.obs.index = [int(i) for i in range(adata.n_obs)]  # cli.py
    adata.obs.index = adata.obs.index.astype(str)
    adata.obs.index.name = "index"
    adata.layers["raw"] = adata.X
    sdata = {"table": adata}
    normalizations.transform_normalize_sc_data(sdata, 5000, 1.0)
    normalizations.fill_nans_for_0_transcript_cells(sdata)
    adata.X = adata.layers["normlog"]  # qc_ambient
    return sdata


def csr_diff(name, a, b):
    a, b = sp.csr_matrix(a), sp.csr_matrix(b)
    d = abs(a - b)
    changed = d.nnz and np.count_nonzero(d.data)
    rows, cols = d.nonzero()
    ref = np.asarray(abs(b)[rows, cols]).ravel()
    rel = (
        np.max(d.data[d.data != 0] / np.where(ref == 0, np.inf, ref))
        if changed
        else 0.0
    )
    print(
        f"  {name}: {changed} entries differ (nnz {a.nnz} vs {b.nnz}), in {len(set(cols))} genes, "
        f"{len(set(rows))} cells; max |d| {d.max() if changed else 0:.6g}, max rel {rel:.6g}"
    )


def vec_diff(name, a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "fc":
        d = np.abs(a - b)
        rel = d / np.where(b == 0, np.inf, np.abs(b))
        print(
            f"  {name}: {np.count_nonzero(d)} of {a.size} differ; max |d| {d.max():.6g}, max rel {rel.max():.6g}"
        )
    else:
        print(f"  {name}: {np.count_nonzero(a != b)} of {a.size} differ")


arms = {}
for name, load in (
    ("origin", original_load_data_for_hqcr),
    ("fixed", hqcr.load_data_for_hqcr),
):
    sdata = table_at_hqcr()
    qc, _, _ = load(sdata, tmp_folder, "canorm_transcript_counts")
    arms[name] = (sdata["table"], qc)

(t_old, qc_old), (t_new, qc_new) = arms["origin"], arms["fixed"]
for attr in ("data", "indices", "indptr"):
    a, b = getattr(qc_old.X, attr), getattr(qc_new.X, attr)
    assert a.dtype == b.dtype and np.array_equal(a, b), attr
print("clustering input identical (float32 CSR data, indices, indptr)")
print(f"first 13 genes: {list(t_old.var_names[:13])}")
print(f"origin: X is layers['normlog']: {t_old.X is t_old.layers['normlog']}")

pristine = table_at_hqcr()["table"]
print("table after hqcr, origin vs fixed:")
csr_diff("X", t_old.X, t_new.X)
csr_diff("layers['normlog']", t_old.layers["normlog"], t_new.layers["normlog"])
csr_diff("layers['raw']", t_old.layers["raw"], t_new.layers["raw"])
vec_diff(
    "layers['normlogscale']", t_old.layers["normlogscale"], t_new.layers["normlogscale"]
)
print("fixed arm vs the table before hqcr:")
csr_diff("X", t_new.X, pristine.X)
csr_diff("layers['normlog']", t_new.layers["normlog"], pristine.layers["normlog"])

# cell cycle QC (runs in -s all after hqcr, on sdata['table'].X): the computations of cellcycle_qc
genes = qc_cellcycle._cell_cycle_genes
print(
    "cell cycle QC (sc.tl.score_genes_cell_cycle on X, then PCA of the cell cycle genes):"
)
s_genes = sorted(set(genes["S"]) & set(t_old.var_names))
g2m_genes = sorted(set(genes["G2M"]) & set(t_old.var_names))
if not s_genes or not g2m_genes:
    # run_qc_cellcycle's own check: the step writes nothing for this panel
    print(f"  skipped, as run_qc_cellcycle does: {len(s_genes)} S and {len(g2m_genes)} G2M genes in the panel")
else:
    out = {}
    for name, (table, _) in arms.items():
        s_genes = list(set(genes["S"]) & set(table.var_names))
        g2m_genes = list(set(genes["G2M"]) & set(table.var_names))
        cc_genes = list(set(s_genes + g2m_genes))
        sc.tl.score_genes_cell_cycle(table, s_genes=s_genes, g2m_genes=g2m_genes)
        cc = table[:, cc_genes]
        sc.tl.pca(cc, use_highly_variable=False)
        out[name] = (
            table.obs[["S_score", "G2M_score", "phase"]].copy(),
            cc.obsm["X_pca"][:, :2].copy(),
            cc_genes,
        )
    print(
        f"  cell cycle genes among the overwritten 13: {sorted(set(out['origin'][2]) & set(t_old.var_names[:13]))}"
    )
    for col in ("S_score", "G2M_score", "phase"):
        vec_diff(col, out["origin"][0][col].to_numpy(), out["fixed"][0][col].to_numpy())
    print(
        "  phase counts origin:",
        out["origin"][0]["phase"].value_counts().to_dict(),
        "fixed:",
        out["fixed"][0]["phase"].value_counts().to_dict(),
    )
    vec_diff("PC1-2 of the cell cycle genes", out["origin"][1], out["fixed"][1])
