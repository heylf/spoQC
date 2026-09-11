import spatialdata as sd
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd
import concurrent.futures
import scipy.sparse as sp
import concurrent.futures

from libpysal.weights import KNN
from scipy.spatial import cKDTree
from scipy.ndimage import convolve

from ... import helperfuncs
from ... import core

# Vectorized Moran-I for all genes in a neighborhood
def _moran_I_all_genes(X_dense: np.ndarray, w) -> np.ndarray:
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
def _build_weights(coords_subset: np.ndarray, k):
    # fixed K neighbors
    # Take Minimum of k (30) cells if there are that many cells.
    w = KNN.from_array(coords_subset, k=k)  # tune k
    return w

# Core computation per i (no sdata['table'] slicing, no GeoPandas)
# This calculate all Moran'Is for all genes for one cell.
def _compute_one_i(i: int, num_genes, distance_matrix, center_cell_ids, coords_all, rna_X, k=30):
    idx = distance_matrix[i]
    m = len(idx)
    center_cell_id = int(center_cell_ids[i])

    if m <= k:
        return center_cell_id, np.full((num_genes,), -1.0, dtype=np.float32)

    coords = coords_all[idx, :]

    # Extract expression for just those cells.
    # If rna_X is sparse: this makes a dense (m, num_genes) only for the neighborhood (cheap-ish).
    X_sub = rna_X[idx, :].toarray() if sp.issparse(rna_X) else np.asarray(rna_X[idx, :])

    w = _build_weights(coords, k)
    I_all = _moran_I_all_genes(X_sub, w)
    return center_cell_id, I_all


# Chunked worker: write into preallocated output
def _worker_chunk(i_start: int, i_end: int, num_genes, distance_matrix, center_cell_ids, coords_all, rna_X):
    ids = np.empty((i_end - i_start,), dtype=np.int64)
    I_block = np.empty((i_end - i_start, num_genes), dtype=np.float32)
    for t, i in enumerate(range(i_start, i_end)):
        cid, I_all = _compute_one_i(i, num_genes, distance_matrix, center_cell_ids, coords_all, rna_X)
        ids[t] = cid
        I_block[t, :] = I_all
    return i_start, ids, I_block


# TODO should become an own metric
def _calculate_local_moran_I_values(sdata, threads):

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
        futures = [ex.submit(_worker_chunk, a, b, num_genes, distance_matrix, 
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



# We are calculating a kernel density at the end so you will not have your usual [-1,1] autocorraltion values.
def _generate_transcript_ambient_density_image(
        sdata,
        figure_path,
        threads,
        global_ambient,
        imagedim,
        dim_x,
        dim_y,
        *, 
        kernel_radius=3,
        flip=False
):

    timer = helperfuncs.Timer()

    transcript_coords_df = sd.get_centroids(sdata['transcripts'], coordinate_system='global').compute()
    transcript_coords_df = transcript_coords_df.astype(int)
    xy_transcript_coords_df = transcript_coords_df.loc[:,['x','y']]
    xy_transcript_coords_df['morans_I'] = np.zeros(len(xy_transcript_coords_df))

    # Attach ambient score to transcript df.
    features = np.array(sdata['transcripts'].compute()['feature_name'])
    global_ambient.index = [i for i in range(0, len(global_ambient))]
    global_ambient.loc[np.isnan(global_ambient['morans_I']),'morans_I'] = 0.0 # Sometimes you have nan for moran's I.

    # I will not check for absolute values because negative autocorrelation might be biological meaningful.
    for i in range(0, len(global_ambient)):
        gene = global_ambient.loc[i, 'genes']
        morans_I = global_ambient.loc[i, 'morans_I']
        xy_transcript_coords_df.loc[features == gene, 'morans_I'] = morans_I

    # Now we will add the local morans I
    xy_transcript_coords_df['local_moran_I'] = _calculate_local_moran_I_values(sdata, threads)

    # First calculate the ambient potential (global Moran's I) ----------------------------------
    print("[NOTE] Calcualte pixel max")
    timer.start()
    gm = (
        xy_transcript_coords_df
        .groupby(["x", "y"])["morans_I"]
        .max()
        .rename("morans_I")
    )

    x_idx = range(int(imagedim.bb_xmin), int(imagedim.bb_xmax))
    y_idx = range(int(imagedim.bb_ymin), int(imagedim.bb_ymax))
    grid = [(x, y) for y in y_idx for x in x_idx]
    grid_mi = pd.MultiIndex.from_tuples(grid, names=["x", "y"])

    transcript_density_list = (
        gm.reindex(grid_mi)     # align to the full grid
        .fillna(0.0)
        .to_numpy()
        .astype("float64")
    )
    timer.stop()

    xy_transcript_density = np.array(transcript_density_list).reshape(dim_x, dim_y)

    # Create circular kernel (disk mask)
    y, x = np.ogrid[-kernel_radius:kernel_radius+1, -kernel_radius:kernel_radius+1]
    mask = (x**2 + y**2) <= kernel_radius**2
    kernel = mask.astype(xy_transcript_density.dtype)
    xy_kernel_transcript_density = convolve(xy_transcript_density, kernel, mode='constant', cval=0)
    xy_kernel_transcript_density = np.flipud(xy_kernel_transcript_density)
    # xy_kernel_transcript_density = xy_kernel_transcript_density.astype(np.uint16) # conversion needed for cv2

    if ( figure_path != None ):
        helperfuncs.plot_pixels(
            figure_path,
            np.log10(xy_kernel_transcript_density + 1),
            imagedim,
            'transcript_global_autocorrelation_density',
            'Transcript Global Autocorrelation Density (Potential)', 
            'gray',
            True,
            True
        )

    # Second calculate the ambient value (local Moran's I) ----------------------------------
    print("[NOTE] Calcualte pixel max")
    timer.start()
    gm = (
        xy_transcript_coords_df
        .groupby(["x", "y"])["local_moran_I"]
        .max()
        .rename("local_moran_I")
    )

    local_transcript_density_list = (
        gm.reindex(grid_mi)     # align to the full grid
        .fillna(0.0)
        .to_numpy()
        .astype("float64")
    )
    timer.stop()

    local_xy_transcript_density = np.array(local_transcript_density_list).reshape(dim_x, dim_y)

    # Create circular kernel (disk mask)
    kernel = mask.astype(local_xy_transcript_density.dtype)
    local_xy_kernel_transcript_density = convolve(local_xy_transcript_density, kernel, mode='constant', cval=0)
    local_xy_kernel_transcript_density = np.flipud(local_xy_kernel_transcript_density)

    if ( figure_path != None ):
        helperfuncs.plot_pixels(
            figure_path,
            np.log10(local_xy_kernel_transcript_density + 1),
            imagedim,
            'transcript_local_autocorrelation_density',
            'Transcript Local Autocorrelation Density (Value)', 
            'gray',
            True,
            True
        )

    # Now we combine both ambient potential with ambient value ---------------------------------
    # combined = -1     ---> -1 * 1 or 1 * -1 = disagreement, direction between global and local
    # combined = 1      ---> -1 * -1 or 1 * 1 = agreement, direction between global and local
    # combined = 0      ---> 0 * -1 or 0 * 1 or -1 * 0 or 1 * 0 = vanishing, RNA is either global or local ambient
    xy_kernel_ac_density = np.abs(local_xy_kernel_transcript_density * xy_kernel_transcript_density)

    if ( figure_path != None ):
        helperfuncs.plot_pixels(
            figure_path,
            np.log10(xy_kernel_ac_density + 1),
            imagedim,
            'transcript_autocorrelation_density',
            'Transcript Autocorrelation Density (Combined)', 
            'gray',
            True,
            True
        )

    return xy_kernel_ac_density.flatten()


def _transcript_ac_image(
        sdata,
        figure_path,
        spoqc_tmp_folder,
        modality,
        threads,
        imagedim,
        dim_x,
        dim_y,
        chunk_size,
    ):
    timer = helperfuncs.Timer()

    print("[NOTE] Generate ac image")
    timer.start()
    global_ambient = pd.read_parquet(f"{spoqc_tmp_folder}/ambient_output_genes.parquet", engine="pyarrow")
    timer.stop()

    # I have now for every pixel the density of the max autocorrelation.
    # That means I know now which pixels have high global gene correlation patterns.
    timer.start()
    np_arr = _generate_transcript_ambient_density_image(sdata, figure_path, threads, global_ambient, 
                                                       imagedim, dim_x, dim_y)
    image_ddf = dd.from_dask_array(da.from_array(np_arr, chunks=chunk_size), columns=["ac_density"])
    timer.stop()

    print("[NOTE] Generate ac histogram")
    timer.start()
    helperfuncs.plot_histogram_for_array(
        image_ddf[image_ddf['ac_density'] > 0]['ac_density'].compute().to_numpy(),
        100,
        figure_path,
        "Transcript autocorrelation density historgram",
        "transcript_ac"
    )
    timer.stop()

    helperfuncs.ddf_to_parquet(image_ddf, 'ac_density', spoqc_tmp_folder, [], modality)


def init_metric(enterprise):

    # These have to be defined.
    name = "ac_density"
    submetrics = ["ac_density"] # use the name above or fill in further metrics calculated by this metric
    modality = "hqtr"
    needs_metrics = ["global_moran_I"]
    step_when_it_is_calculated = ['all', 'unittest', 'hqtr', 'hqtr_ac']
    loaded_for_analysis = True
    loaded_for_visualization = True

    # These are given my your metric calc function.
    args = [enterprise.cargo.sdata, f'{enterprise.args.output_dir}/hqtr/hqtr_ac/', enterprise.args.tmp_dir, 'hqtr',
            enterprise.args.nthreads, enterprise.cargo.imagedim, enterprise.cargo.dim_x, enterprise.cargo.dim_y,
            enterprise.args.chunk_size]
    kwargs = None

    metric = core.metric.Metric(
        _transcript_ac_image, 
        name,
        submetrics,
        modality,
        needs_metrics = needs_metrics,
        step_when_it_is_calculated = step_when_it_is_calculated,
        loaded_for_analysis = loaded_for_analysis,
        loaded_for_visualization = loaded_for_visualization,
        args = args,
        kwargs = kwargs,
    )    
    
    return metric