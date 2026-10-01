"""load_data_for_hqcr no longer writes the QC table into sdata['table'].X (a correctness fix).

origin/dev set X on a view of sdata['table'], which wrote the 13 QC columns into the table's
X, the same object as layers['normlog']. The fixed version must leave the table untouched and
still give the clustering exactly the input it had (float32 CSR, same data/indices/indptr).
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from spoqc import helperfuncs
from spoqc.subworkflows import hqcr

N_CELLS, N_GENES = 400, 30
QC_COLUMNS = [
    "canorm_transcript_counts",
    "control_probe_counts",
    "canorm_n_genes_by_counts",
    "convexity_metric_cell",
    "convexity_min_nuceli",
    "nuceli_count",
    "border_scores",
    "thinness_score",
    "island_score",
    "wdoublet",
    "cell_overlap_area",
    "convexhull_outside_trnascripts",
    "num_low_qc_transcript",
]


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


def table():
    """A table as hqcr sees it: X is layers['normlog'] (float32 CSR), QC metrics in obs."""
    rng = np.random.default_rng(0)
    counts = sp.random(
        N_CELLS, N_GENES, density=0.3, format="csr", random_state=1, dtype=np.float32
    )
    normlog = counts.copy()
    normlog.data = np.log1p(normlog.data * 10).astype(np.float32)
    obs = pd.DataFrame(index=[str(i) for i in range(N_CELLS)])
    for col in QC_COLUMNS:
        values = rng.gamma(2.0, 3.0, N_CELLS)
        values[rng.random(N_CELLS) < 0.3] = (
            0.0  # zeros where the gene had a value, and vice versa
        )
        if col in ("nuceli_count", "wdoublet"):
            obs[col] = values.astype(np.int64)
        else:
            values[rng.random(N_CELLS) < 0.02] = np.nan  # a NaN is a stored value too
            obs[col] = values
    adata = ad.AnnData(X=normlog, obs=obs, layers={"raw": counts, "normlog": normlog})
    adata.X = adata.layers["normlog"]  # qc_ambient: the same object
    return {"table": adata}


@pytest.fixture
def tmp_folder(tmp_path):
    return str(
        tmp_path
    )  # no *_hqcr.parquet: the metrics are already in obs, as in -s all


def test_clustering_input_is_unchanged(tmp_folder):
    old_sdata, new_sdata = table(), table()
    old, old_df, old_metrics = original_load_data_for_hqcr(
        old_sdata, tmp_folder, "canorm_transcript_counts"
    )
    new, new_df, new_metrics = hqcr.load_data_for_hqcr(
        new_sdata, tmp_folder, "canorm_transcript_counts"
    )
    assert new_metrics == old_metrics
    pd.testing.assert_frame_equal(new_df, old_df)
    # the original's X is anndata's view class, a csr_matrix subclass
    assert isinstance(new.X, sp.csr_matrix) and isinstance(old.X, sp.csr_matrix)
    assert new.shape == old.shape
    for attr in ("data", "indices", "indptr"):
        a, b = getattr(new.X, attr), getattr(old.X, attr)
        assert a.dtype == b.dtype, attr
        np.testing.assert_array_equal(a, b, err_msg=attr)
    pd.testing.assert_frame_equal(
        new.obs, pd.DataFrame(old.obs)
    )  # old.obs is a DataFrameView


def test_table_is_left_untouched(tmp_folder):
    sdata, before = table(), table()
    hqcr.load_data_for_hqcr(sdata, tmp_folder, "canorm_transcript_counts")
    t, b = sdata["table"], before["table"]
    assert t.X is t.layers["normlog"]
    for layer in ("normlog", "raw"):
        assert (t.layers[layer] != b.layers[layer]).nnz == 0, layer
        np.testing.assert_array_equal(t.layers[layer].indptr, b.layers[layer].indptr)


def test_original_overwrote_the_table(tmp_folder):
    """Mutation check: the untouched-table test can see the original's write."""
    sdata, before = table(), table()
    original_load_data_for_hqcr(sdata, tmp_folder, "canorm_transcript_counts")
    changed = sdata["table"].layers["normlog"] != before["table"].layers["normlog"]
    assert changed.nnz > 0
    assert set(changed.nonzero()[1]) <= set(range(len(QC_COLUMNS)))
