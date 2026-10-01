"""Differential tests: hqcr's clustering (core.knn.neighbors, no UMAP) vs the verbatim original.

`original_clustering_for_hqcr` is copied verbatim (then line-wrapped by ruff format) from spoQC origin/dev db00d98
(subworkflows/hqcr.py). The original also ran sc.tl.umap, whose X_umap nothing in hqcr reads;
every other output (the kNN distances and connectivities, .uns['neighbors'], the Leiden labels,
numpy's global RNG state) must be bit-identical, dtypes included.
Below 8192 rows (euclidean) scanpy uses sklearn brute force, not pynndescent, so the pynndescent
cases have more rows than that.
"""

import types
from unittest import mock

import anndata as ad
import numba
import numpy as np
import pytest
import scanpy as sc
import scipy.sparse as sp
from pynndescent import pynndescent_

from spoqc import helperfuncs
from spoqc.core import knn
from spoqc.subworkflows import hqcr

THREADS = 4
SEED = 123
N_FEATURES = 13  # hqcr's QC metric columns


# ---------------------------------------------------------------- verbatim original (db00d98)


def original_clustering_for_hqcr(
    qc_domains_adata, figure_path, CONST, seed, test_res_n_clusters=10, test_res=False
):
    # leiden clustering
    print("[NOTE] Cell QC clustering")
    sc.pp.neighbors(qc_domains_adata, n_neighbors=20, random_state=seed)
    sc.tl.umap(qc_domains_adata, random_state=seed)

    if test_res:
        helperfuncs.test_resolutions_leiden(
            qc_domains_adata, figure_path, CONST.THREADS, k=test_res_n_clusters
        )

    sc.tl.leiden(qc_domains_adata, resolution=1.2)


# ---------------------------------------------------------------- inputs


def qc_table(n_cells, sparse, rng_seed=0):
    """Clustered QC-metric-like table: blobs of skewed counts, scores in [0, 1], flags, many zeros."""
    rng = np.random.default_rng(rng_seed)
    centre = rng.gamma(2.0, 2.0, size=(8, N_FEATURES))
    blob = rng.integers(0, 8, n_cells)
    x = centre[blob] + rng.normal(0.0, 0.6, (n_cells, N_FEATURES))
    x[:, 5] = rng.integers(0, 3, n_cells)  # nuclei count
    x[:, 9] = rng.random(n_cells) < 0.1  # doublet flag
    x[rng.random(x.shape) < 0.25] = 0.0
    x = np.clip(x, 0.0, None).astype(np.float32)
    adata = ad.AnnData(X=sp.csr_matrix(x) if sparse else x)
    adata.obs_names = [str(i) for i in range(n_cells)]
    return adata


def outputs(adata):
    out = {
        "leiden": adata.obs["leiden"],
        "uns_params": adata.uns["neighbors"]["params"],
        "uns_keys": sorted(adata.uns["neighbors"]),
        "global_rng": np.random.get_state()[1].copy(),
    }
    for key in ("distances", "connectivities"):
        m = adata.obsp[key]
        out[key] = (type(m), m.shape, m.data, m.indices, m.indptr)
    return out


def assert_identical(new, old):
    assert (
        new["leiden"].dtype == old["leiden"].dtype
    )  # categorical, same categories and order
    assert new["leiden"].equals(old["leiden"])
    assert new["uns_params"] == old["uns_params"]
    assert new["uns_keys"] == old["uns_keys"]
    np.testing.assert_array_equal(new["global_rng"], old["global_rng"])
    for key in ("distances", "connectivities"):
        (t_new, s_new, *arrays_new), (t_old, s_old, *arrays_old) = new[key], old[key]
        assert (t_new, s_new) == (t_old, s_old), key
        for a_new, a_old in zip(arrays_new, arrays_old):
            assert a_new.dtype == a_old.dtype, key
            np.testing.assert_array_equal(a_new, a_old, err_msg=key)


def run(fn, adata):
    np.random.seed(SEED)
    fn(adata, "unused", types.SimpleNamespace(THREADS=THREADS), SEED)
    return outputs(adata)


# ---------------------------------------------------------------- tests


@pytest.mark.parametrize("sparse", [True, False], ids=["csr", "dense"])
@pytest.mark.parametrize("n_cells", [5_000, 12_000])
def test_clustering_matches_original(n_cells, sparse):
    old = run(original_clustering_for_hqcr, qc_table(n_cells, sparse))
    new_adata = qc_table(n_cells, sparse)
    new = run(hqcr.clustering_for_hqcr, new_adata)
    assert_identical(new, old)
    assert "X_umap" not in new_adata.obsm


def test_small_table_keeps_scanpy_brute_force():
    """Below 8192 cells scanpy's own sklearn path runs; the primitive must not replace it."""
    old = run(original_clustering_for_hqcr, qc_table(1_000, True))
    new = run(hqcr.clustering_for_hqcr, qc_table(1_000, True))
    assert_identical(new, old)


@pytest.mark.parametrize("threads", [1, 3, 8])
def test_graph_does_not_depend_on_forest_threads(threads):
    ref, other = qc_table(10_000, True), qc_table(10_000, True)
    knn.neighbors(ref, 20, SEED, 1)
    knn.neighbors(other, 20, SEED, threads)
    for key in ("distances", "connectivities"):
        for attr in ("data", "indices", "indptr"):
            np.testing.assert_array_equal(
                getattr(other.obsp[key], attr), getattr(ref.obsp[key], attr)
            )


def test_forest_runs_on_the_thread_budget_and_index_is_not_compressed():
    """Guards the two changes: the forest gets `threads`, compress_index() never runs."""
    calls = []
    real = pynndescent_.make_forest

    def spy(*args, **kwds):
        calls.append(args[6])
        return real(*args, **kwds)

    adata = qc_table(10_000, True)
    with (
        mock.patch("spoqc.core.knn.make_forest", spy),
        mock.patch.object(
            pynndescent_.NNDescent,
            "compress_index",
            side_effect=AssertionError("compressed"),
        ),
    ):
        knn.neighbors(adata, 20, SEED, 5)
    assert calls == [5]
    assert pynndescent_.make_forest is knn._MAKE_FOREST  # restored after the fit


def test_forest_swap_is_undone_when_the_fit_fails():
    with mock.patch.object(pynndescent_.NNDescent, "__init__", side_effect=RuntimeError("fit failed")):
        with pytest.raises(RuntimeError, match="fit failed"):
            knn.neighbors(qc_table(10_000, True), 20, SEED, 2)
    assert pynndescent_.make_forest is knn._MAKE_FOREST


def test_forest_swap_refuses_a_replaced_make_forest():
    with mock.patch.object(pynndescent_, "make_forest", lambda *a, **k: None):
        with pytest.raises(RuntimeError, match="replaced by other code"):
            knn.neighbors(qc_table(10_000, True), 20, SEED, 2)


def test_nn_descent_thread_count_would_change_the_graph():
    """Mutation check: the comparison above can see a real difference. Raising scanpy's n_jobs
    (NN-descent threads) is the change the primitive must not make."""
    nn_threads = min(4, numba.config.NUMBA_NUM_THREADS)  # set_num_threads refuses more
    if nn_threads < 2:
        pytest.skip("needs at least 2 numba threads")
    ref, other = qc_table(12_000, True), qc_table(12_000, True)
    knn.neighbors(ref, 20, SEED, 1)
    with mock.patch.object(sc.settings, "_n_jobs", nn_threads):
        knn.neighbors(other, 20, SEED, 1)
    assert not np.array_equal(
        other.obsp["distances"].indices, ref.obsp["distances"].indices
    )
