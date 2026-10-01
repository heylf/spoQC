import numpy as np
import pandas as pd
import concurrent.futures
import scipy.sparse as sp
import concurrent.futures

from libpysal.weights import KNN
from scipy.spatial import cKDTree

from ... import helperfuncs

# Vectorized Moran-I for all genes in a neighborhood
def moran_I_all_genes(X_dense: np.ndarray, w) -> np.ndarray:
    """
    X_dense: (n, num_genes) float array for selected cells
    w: libpysal weights object
    returns: (num_genes,) Moran's I per gene
    """
    n = X_dense.shape[0]
    if n < 3:
        return np.full((X_dense.shape[1],), -1.0, dtype=np.float32)

    # ensure sparse CSR for W
    # NOTE: w.sparse is typically CSR; w.transform='r' row-standardizes
    w.transform = "r"
    weights = w.sparse  # scipy sparse

    # S0 for row-standardized weights is just sum(W)
    row_standardized_weights = weights.sum()
    if row_standardized_weights == 0:
        return np.full((X_dense.shape[1],), -1.0, dtype=np.float32)

    # center (do NOT standardize by std unless you want "z-scores"; Moran uses mean-centering)
    z = X_dense - X_dense.mean(axis=0, keepdims=True)

    # sparse matmul releases the GIL and is fast
    z_weights = weights @ z

    num = np.einsum("ij,ij->j", z, z_weights)         # sum over rows
    den = np.einsum("ij,ij->j", z, z)

    # protect against constant genes in the neighborhood
    out = np.full((X_dense.shape[1],), -1.0, dtype=np.float32)
    ok = den > 0
    out[ok] = (n / row_standardized_weights) * (num[ok] / den[ok])
    return out

# Choose a fast weights builder for points
def build_weights(coords_subset: np.ndarray, k):
    # fixed K neighbors
    # Take Minimum of k (30) cells if there are that many cells.
    w = KNN.from_array(coords_subset, k=k)  # tune k
    return w

# Core computation per i (no sdata['table'] slicing, no GeoPandas)
# This calculate all Moran'Is for all genes for one cell.
def compute_one_i(i: int, num_genes, distance_matrix, center_cell_ids, coords_all, rna_X, k=30):
    idx = distance_matrix[i]
    m = len(idx)
    center_cell_id = int(center_cell_ids[i])

    if m <= k:
        return center_cell_id, np.full((num_genes,), -1.0, dtype=np.float32)

    coords = coords_all[idx, :]

    # Extract expression for just those cells.
    # If rna_X is sparse: this makes a dense (m, num_genes) only for the neighborhood (cheap-ish).
    X_sub = rna_X[idx, :].toarray() if sp.issparse(rna_X) else np.asarray(rna_X[idx, :])

    w = build_weights(coords, k)
    I_all = moran_I_all_genes(X_sub, w)
    return center_cell_id, I_all


# Chunked worker: write into preallocated output
def worker_chunk(i_start: int, i_end: int, num_genes, distance_matrix, center_cell_ids, coords_all, rna_X):
    ids = np.empty((i_end - i_start,), dtype=np.int64)
    I_block = np.empty((i_end - i_start, num_genes), dtype=np.float32)
    for t, i in enumerate(range(i_start, i_end)):
        cid, I_all = compute_one_i(i, num_genes, distance_matrix, center_cell_ids, coords_all, rna_X)
        ids[t] = cid
        I_block[t, :] = I_all
    return i_start, ids, I_block


def calculate_local_moran_I_values(sdata, threads):

    # ----------------------------
    # Precompute once (avoid .todense())
    # ----------------------------
    genes_list = np.array(sdata['table'].var_names)
    num_genes = len(genes_list)

    # keep sparse if possible
    rna_X = sdata['table'].X  # typically CSR/CSC
    coords_all = np.asarray(sdata['table'].obsm['spatial'], dtype=np.float64)
    center_cell_ids = sdata['table'].obs.index.to_numpy()

    distance_matrix = helperfuncs.points_within_radius(
        # if it accepts array, give coords_all; otherwise keep your df_coords
        # df_coords,
        pd.DataFrame({"x": coords_all[:, 0], "y": coords_all[:, 1]}),
        100,
        False
    )

    # ----------------------------
    # Parallel execution
    # ----------------------------
    n = len(distance_matrix)
    chunk_size = 128  # bigger is usually better after vectorization
    chunks = [(start, min(start + chunk_size, n)) for start in range(0, n, chunk_size)]

    # Preallocate final result: (n_cells, n_genes)
    # If you need mapping by center_cell_id, keep ids separately (returned).
    all_ids = np.empty((n,), dtype=np.int64)
    all_I = np.empty((n, num_genes), dtype=np.float32)

    timer = helperfuncs.Timer()
    timer.start()

    # After vectorization, threads often work well because numpy/scipy sparse releases GIL.
    # If weight-building dominates and is pure Python, try ProcessPoolExecutor instead.
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as ex:
        futures = [ex.submit(worker_chunk, a, b, num_genes, distance_matrix, 
                             center_cell_ids, coords_all, rna_X) for (a, b) in chunks]
        for k, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
            i_start, ids_block, I_block = fut.result()
            i_end = i_start + len(ids_block)
            all_ids[i_start:i_end] = ids_block
            all_I[i_start:i_end, :] = I_block
            if k % 5 == 0 or k == len(futures):
                print(f"done chunks {k}/{len(futures)}")

    timer.stop()

    # Now you have:
    #   all_ids: (n,) center cell ids
    #   all_I:   (n, num_genes) Moran's I per center cell and gene
    
    # Make sure dtypes match your all_ids / var_names
    transcripts_df = sdata.points['transcripts'].compute()
    transcripts_cell_id = transcripts_df["cell_id"].to_numpy()
    transcripts_feature = transcripts_df["feature_name"].to_numpy()

    # Build fast maps -> indices
    cell_to_row = {cid: i for i, cid in enumerate(all_ids)}
    gene_to_col = {g: j for j, g in enumerate(sdata['table'].var_names)}

    # Vectorize mapping via pandas (fast C code) rather than Python loops
    # (This avoids a Python loop over transcripts.)
    cell_rows = pd.Index(transcripts_cell_id).map(cell_to_row).to_numpy()
    gene_cols = pd.Index(transcripts_feature).map(gene_to_col).to_numpy()

    # Initialize output
    loca_morans_I_array = np.full(len(transcripts_feature), -1.0, dtype=np.float32)

    # Valid rows are those that found both a cell and a gene
    valid = (cell_rows != -1) & (gene_cols != -1) & (~pd.isna(cell_rows)) & (~pd.isna(gene_cols))

    # Convert to int for indexing
    cell_rows = cell_rows.astype(np.int64, copy=False)
    gene_cols = gene_cols.astype(np.int64, copy=False)

    # One shot gather
    loca_morans_I_array[valid] = all_I[cell_rows[valid], gene_cols[valid]]
    transcripts_df['local_moran_I'] = loca_morans_I_array

    # --------------------------------------------------------
    # Now I have to take care of the transcripts outside cells
    # --------------------------------------------------------
    # For those transcripts I take the nearest transcripts with the same feature name.
    # If none can be found the local Moran's I will be set to 0.0.

    # Masks
    outside_mask = (loca_morans_I_array == -1)
    inside_mask  = ~outside_mask

    # Pull arrays once (avoid repeated pandas overhead)
    x = transcripts_df["x"].to_numpy(dtype=np.float64, copy=False)
    y = transcripts_df["y"].to_numpy(dtype=np.float64, copy=False)
    coords = np.column_stack((x, y))

    feat = transcripts_feature  # already a numpy array per your code
    local_I = transcripts_df["local_moran_I"].to_numpy(dtype=np.float32, copy=False)

    # Work on outside only, grouped by feature
    features_outside = np.unique(feat[outside_mask])

    dist_thresh = 100.0  # max distance

    for f in features_outside:
        # indices for this feature
        out_idx = np.flatnonzero(outside_mask & (feat == f))
        if out_idx.size == 0:
            continue

        in_idx = np.flatnonzero(inside_mask & (feat == f))
        if in_idx.size == 0:
            # no inside transcripts of this feature -> keep default behavior
            # your old code sets 0.0 when it can't find a neighbor within 100
            local_I[out_idx] = 0.0
            continue

        # KDTree on inside points of this feature
        tree = cKDTree(coords[in_idx])

        # Query nearest inside point for each outside point, with cutoff radius
        dists, nn = tree.query(coords[out_idx], k=1, distance_upper_bound=dist_thresh)

        # nn is an index into in_idx (or == len(in_idx) when no neighbor within R)
        has_neighbor = np.isfinite(dists) & (nn < in_idx.size)

        # default when no neighbor within R (matches your old win_moran_I init)
        local_I[out_idx] = 0.0
        local_I[out_idx[has_neighbor]] = local_I[in_idx[nn[has_neighbor]]]

    # Write back once
    transcripts_df["local_moran_I"] = local_I
    print('... done calculating local morans I')
    return np.array(transcripts_df["local_moran_I"])