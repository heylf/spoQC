"""Verbatim origin/dev db00d98 pixel-scoring code, for differential tests.

Sections, each copied unchanged except that cross-module calls are pointed at
the copies in this file (marked `# ref:`):
  - helperfuncs.read_data_as_ddf
  - metrics/image/utility.estimate_background_intensity_dask
  - metrics/image/pixel_score.py (dask_summify, calc_pixel_score)
  - image_analysis/pixel_scoring_dask.py (dask_clustering_mini_batches, start_pixel_qc)
  - priors/combine_priors.py (combine_priors_hqpr, combine_priors_hqtr)
  - image_analysis/pixel_scoring_refinement.py's mask_raw read-back
"""
import os
import sys

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
from dask_ml.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans

from conftest import load_legacy
from legacy.parquet_writer import ddf_to_parquet  # origin/dev's writer, the reference for core.parquet
from spoqc import helperfuncs, priors

# origin/dev priors/hqpr/pixel_score.py (since merged into priors.gaussian)
LEGACY_PIXEL_SCORE_PRIOR = load_legacy("pixel_score_prior", "spoqc.priors.hqpr")

def read_data_as_ddf(tmp_files, chunk_size):
    # Preallocate a Dask Array with correct shape and chunks
    col_series = [
        dd.read_parquet(file).iloc[:, 0].reset_index(drop=True)
        for file in tmp_files
    ]

    # Compute all per-file partition lengths together (in parallel) instead of
    # letting to_dask_array(lengths=True) block on each file one at a time.
    lengths_per_col = dask.compute(*[s.map_partitions(len) for s in col_series])

    array_columns = []
    for col_ddf, lengths in zip(col_series, lengths_per_col):
        col_arr = col_ddf.to_dask_array(lengths=tuple(lengths)).rechunk((chunk_size,))
        array_columns.append(col_arr[:, None])  # make 2D for stacking

    # Stack columns into 2D Dask Array
    dask_array = da.hstack(array_columns).astype(np.float32)  # Much cheaper than dd.concat
    dask_array = dask_array.rechunk((chunk_size, -1))
    # Optional but recommended to avoid re-reading Parquet each epoch:
    # da.to_zarr(dask_array, "dask_array.zarr", overwrite=True); dask_array = da.from_zarr("dask_array.zarr")

    return dask_array


def estimate_background_intensity_dask(sdata, image_type, resolution, staining, nbins=100, range_=None):
    """
    nbins: number of histogram bins
    range_: optional (min, max); if None, computed lazily with dask
    """
    intensities = sdata[image_type][resolution].image.data[int(staining)]
    intensities.ravel()

    if not hasattr(intensities, "chunks"):
        raise TypeError("Pass a dask.array for the Dask implementation.")

    # Compute min/max lazily if not supplied (cheap: just scalars)
    if range_ is None:
        vmin = da.nanmin(intensities)
        vmax = da.nanmax(intensities)
        vmin, vmax = da.compute(vmin, vmax)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            raise ValueError("Non-finite min/max encountered.")
        if vmin == vmax:
            vmax = vmin + 1.0
        range_ = (float(vmin), float(vmax))

    # Dask builds the histogram in a reduction; result is tiny (nbins) -> safe to .compute()
    hist, bin_edges = da.histogram(intensities, bins=nbins, range=range_)
    hist, bin_edges = da.compute(hist, bin_edges)

    max_bin_idx = int(np.argmax(hist))
    # center of the winning bin
    background = np.round((bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) * 0.5, 3)
    return background, hist, bin_edges

def dask_summify(spoqc_tmp_folder, suffix, metrices, chunk_size):
    tmp_files = [f'{spoqc_tmp_folder}/{metric}_output_{suffix}.parquet' for metric in metrices]
    ddf = read_data_as_ddf(  # ref:
        tmp_files, chunk_size)
    structure_scores = ddf.sum(axis=1)
    return structure_scores

# In[]
def calc_pixel_score(
        sdata,
        figure_path,
        spoqc_tmp_folder_metrices,
        modality,
        image_type,
        resolution,
        dim_x,
        dim_y,
        imagedim,
        tmp_suffix,
        plot_all_pixel_clusters,
        chunk_size,
        staining,
        image_ddf
    ):

    timer = helperfuncs.Timer()
    timer_all = helperfuncs.Timer()
    timer_all.start()

    # Calcualte structure and antistructure score.
    print("[NOTE] Dask summify")
    timer.start()
    s_score = dask_summify(spoqc_tmp_folder_metrices, tmp_suffix, 
                           ['edge_strength', 'energy', 'relevance', 'entropy'], chunk_size)
    as_score = dask_summify(spoqc_tmp_folder_metrices, tmp_suffix, ['homogenity', 'uniformity'], chunk_size)
    timer.stop()

    # Set same index on both score series
    print("[NOTE] Get intensities")
    timer.start()
    image_ddf = image_ddf.assign(s_score=s_score, as_score=as_score)

    background_intensity = 0.0
    if ( modality == 'hqpr' ):
        # Lazy Dask-native histogram instead of eagerly pulling the whole
        # full-resolution image into a numpy array just to estimate this.
        background_intensity, _, _ = estimate_background_intensity_dask(  # ref:
            sdata, image_type, resolution, staining
        )
        intensity_da = da.flip(
            sdata[image_type][resolution].image.data[int(staining)], axis=0
        ).flatten().rechunk((chunk_size,))
        image_ddf = image_ddf.assign(intensity=intensity_da)
    elif ( modality == 'hqtr' ):
        # Intensities already flipped
        td_file = f'{spoqc_tmp_folder_metrices}/transcript_density_output_hqtr.parquet'
        intensity = read_data_as_ddf(  # ref:
        [td_file], chunk_size)[:, 0]
        image_ddf = image_ddf.assign(intensity=intensity)
    else:
        sys.exit(f'[ERROR] Modality {modality} does not exist')

    # Materialize s_score/as_score/intensity once so the repeated .compute()
    # calls below don't each re-walk the whole graph from scratch.
    image_ddf = image_ddf.persist()
    timer.stop()

    # Background refined image (background_intensity was already estimated
    # directly from numpy above, no Dask round-trip needed).
    print('[NOTE] Background estimation')
    timer.start()
    t=1.5
    if ( background_intensity == 0 ):
        background_intensity = 1
    timer.stop()

    # Group by cluster and compute mean intensity, s_score, and as_score in a single pass
    # (one groupby/shuffle instead of three separate ones).
    print("[NOTE] Get mean intensity, s and as scores for each cluster")
    timer.start()
    cluster_means_df = image_ddf.groupby('cluster')[['intensity', 's_score', 'as_score']].mean().compute()
    cluster_mean_int_df = cluster_means_df['intensity'].rename('mean_cluster_intensity')
    timer.stop()

    # Since kmeans clusters might not find enough clusters I have to get all possible clsuter ids from the dataframe.
    clusters_ids = list(cluster_mean_int_df.index)
    clusters_ids_arr = np.array(clusters_ids)

    print("[NOTE] Compare clusters to background")
    timer.start()
    abs_cluster_signla_noise_log2fc = np.abs(
        np.log2((cluster_mean_int_df.to_numpy() + 1) / background_intensity)
    )
    background_clusters = set(clusters_ids_arr[abs_cluster_signla_noise_log2fc < t])
    timer.stop()

    print("[NOTE] Get ps scores")
    # Each pixel cluster get a pixel_score = s_score - as_score.
    timer.start()
    cluster_mean_s_score_ds = cluster_means_df['s_score'].rename('mean_s_score')
    cluster_mean_as_score_ds = cluster_means_df['as_score'].rename('mean_as_score')
    pixel_scores_ds = np.round(cluster_mean_s_score_ds - cluster_mean_as_score_ds, 2)
    timer.stop()

    # Generate plots to investigate individual pixel clusters.
    # cluster_array is only needed for plotting, so skip materializing it (and the loop
    # below) entirely when plot_all_pixel_clusters is False.
    if ( plot_all_pixel_clusters ):
        timer.start()
        print("[NOTE] Generate pixel cluster plots")
        cluster_array = image_ddf['cluster'].compute().to_numpy()
        for k in clusters_ids:
            cluster_selection = cluster_array == k

            s_score = np.round(cluster_mean_s_score_ds.loc[k],2)
            as_score = np.round(cluster_mean_as_score_ds.loc[k],2)
            pixel_score = pixel_scores_ds.loc[k]

            title = f'Pixel Cluster {k}'
            if ( k in background_clusters ):
                title = f'Background Pixel Cluster {k}'
            title = title + f' with pixel_score {pixel_score:.2f} s_core {s_score:.2f} and as_score {as_score:.2f}'

            # Check the sturucture of those pixel clusters.
            helperfuncs.plot_pixels(
                figure_path,
                np.array(cluster_selection).reshape(dim_x, dim_y),
                imagedim,
                'clusters',
                title,
                'gray',
                False,
                True
            )
        timer.stop()

    print("[NOTE] Pixel scoring calculation took:")
    timer_all.stop()

    return pixel_scores_ds, clusters_ids, image_ddf


def dask_clustering_mini_batches(spoqc_tmp_folder, suffix, n_clusters, seed, chunk_size, threads, sample_size=5_000_000):
    tmp_files = [f'{spoqc_tmp_folder}/{file}' for file in os.listdir(spoqc_tmp_folder)
             if file.endswith(f'{suffix}.parquet')]

    timer = helperfuncs.Timer()

    with dask.config.set(scheduler="threads", num_workers=threads):
        print("[NOTE] Read data")
        dask_array = read_data_as_ddf(  # ref:
            tmp_files, chunk_size)
        n_rows = dask_array.shape[0]

        print("[NOTE] Clustering (fit on subsample, predict on full array in parallel)")
        timer.start()

        # A streamed/incremental fit (dask_ml.wrappers.Incremental) is
        # inherently sequential -- each chunk's partial_fit depends on the
        # previous chunk's centroid state -- which turns into thousands of
        # sequential Python-level calls at full-image pixel counts. Fitting
        # once on a large i.i.d. subsample and then predicting the rest in
        # parallel (map_blocks, no shared state) avoids that entirely.
        frac = min(1.0, sample_size / n_rows)
        sample_mask = da.random.default_rng(seed).random(n_rows, chunks=(chunk_size,)) < frac
        sample_np = dask_array[sample_mask].compute()

        est = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            batch_size=min(chunk_size, len(sample_np)),
            reassignment_ratio=0.01,
            n_init=3,
        )
        est.fit(sample_np)

        labels = dask_array.map_blocks(
            lambda block: est.predict(block),
            dtype=np.int32,
            drop_axis=1,
        )
        timer.stop()
    return labels


# In[]
def start_pixel_qc(
        sdata,
        figure_path,
        spoqc_tmp_folder,
        modality,
        image_type,
        resolution,
        dim_x,
        dim_y,
        imagedim,
        seed,
        threads,
        *,
        plot_all_pixel_clusters=False,
        chunk_size=10_000,
        sample_size=5_000_000,
        staining=None,
        thresh_p=None,
        nstds_p=None,
    ):

    timer = helperfuncs.Timer()
    timer_all = helperfuncs.Timer()
    timer_all.start()

    # Just path variables
    tmp_suffix = modality
    spoqc_tmp_folder_metrices = ''
    if ( staining ):
        spoqc_tmp_folder_metrices = f'{spoqc_tmp_folder}/metrices/{modality}/{staining}'
        figure_path = f'{figure_path}/{modality}/{modality}_clustering/{staining}/'
        tmp_suffix = f'{modality}_{staining}'
    else:
        spoqc_tmp_folder_metrices = f'{spoqc_tmp_folder}/metrices/{modality}'
        figure_path = f'{figure_path}/{modality}/{modality}_clustering/'

    # Sanitycheck if files exists
    metrices = ['edge_strength', 'energy', 'relevance', 'entropy', 'homogenity', 'uniformity']
    for metric in metrices:
        metric_file = f"{spoqc_tmp_folder_metrices}/{metric}_output_{tmp_suffix}.parquet"
        if ( not os.path.exists(metric_file) ):
            sys.exit(f"[ERROR] File {metric_file} is missing")

    print('[NOTE] Agglomerate pixel metrices and cluster')
    num_values_image = len(sdata[image_type][resolution].image.values[0].flatten())
    empty_clusters = da.zeros(num_values_image, chunks=chunk_size)
    image_ddf = dd.from_dask_array(empty_clusters, columns=['cluster'])
    timer.start()
    n_clusters = 100
    image_ddf = image_ddf.assign(cluster = dask_clustering_mini_batches(
        spoqc_tmp_folder_metrices,
        tmp_suffix,
        n_clusters,
        seed,
        chunk_size,
        threads,
        sample_size
    ))
    print("[NOTE] Time for the whole clustering process:")
    image_ddf = image_ddf.persist()
    timer.stop()

    #####################
    ###### Metrics ######
    #####################
    # We calculate pixel scores for each pixel cluster.
    pixel_scores_ds, clusters_ids, image_ddf = calc_pixel_score(  # ref:
        sdata,
        figure_path,
        spoqc_tmp_folder_metrices,
        modality,
        image_type,
        resolution,
        dim_x,
        dim_y,
        imagedim,
        tmp_suffix,
        plot_all_pixel_clusters,
        chunk_size,
        staining,
        image_ddf   
    )

    ####################
    ###### Priors ######
    ####################
    # Based on the pixel_scores of the individual clusters figure out with GMM which clusters correspond to inforamtion.
    # Based on that you can assign a probability to each cluster that they belong to useful information.
    # Based on that you can assign to each pixel the prabolity of the pixel cluster they belong to.
    prob_densities = LEGACY_PIXEL_SCORE_PRIOR.calc_probs_pixel_score(
        np.array(pixel_scores_ds),
        figure_path,
        gmm_mod=3,
        nstds=nstds_p,
        t=thresh_p
    )    

    ########################
    ###### Downstream ######
    ########################

    # Map cluster densities to each pixel.
    cluster_prob_map = dict(zip(clusters_ids, prob_densities))
    image_ddf = image_ddf.assign(p_informative_pixel=image_ddf['cluster'].map(cluster_prob_map))

    # Min-Max normalization
    print("[NOTE] Min-max normalization")
    timer.start()
    scaler = MinMaxScaler()
    
    belief_name = f"{modality}_beliefs"
    mask_name = f"{modality}_mask"
    if modality == 'hqpr':
        belief_name = f"{modality}_{staining}_beliefs"
        mask_name = f"{modality}_{staining}_mask"

    # I only generate cluster probs and not all pixel probs.
    scaled_ddf = scaler.fit_transform(image_ddf[['p_informative_pixel']])
    image_ddf = image_ddf.assign(
        **{
            'norm_p_pixel_score': scaled_ddf.iloc[:, 0],
            'pixel_score_mask': (scaled_ddf.iloc[:, 0] > 0.5).astype(int),
        }
    )
    image_ddf = image_ddf.persist()
    timer.stop()

    helperfuncs.plot_pixels(
        figure_path,
        image_ddf['norm_p_pixel_score'].compute().to_numpy().reshape(dim_x, dim_y),
        imagedim,
        'norm_p_pixel_score',
        'Normalized pixel score probability', 
        'hot',
        False,
        False
    )
    
    print("[NOTE] Combining priors")
    timer.start()
    if modality == 'hqpr':
        image_ddf = combine_priors_hqpr(  # ref:
            spoqc_tmp_folder, image_ddf, belief_name, mask_name)
    if modality == 'hqtr':
        image_ddf = combine_priors_hqtr(  # ref:
            spoqc_tmp_folder, image_ddf, belief_name, mask_name)
    image_ddf = image_ddf.persist()
    timer.stop()

    helperfuncs.plot_pixels(
        figure_path,
        image_ddf[belief_name].compute().to_numpy().reshape(dim_x, dim_y),
        imagedim,
        'norm_p_beliefs',
        'Normalized combined probability (beliefs)', 
        'hot',
        False,
        False
    )

    print("[NOTE] Writing out data")
    timer.start()
    ddf_to_parquet(image_ddf, tmp_suffix, spoqc_tmp_folder, [], 'mask_raw')
    timer.stop()

    print("[NOTE] The pixel clustering and prior estimation took:")
    timer_all.stop()

# %%


def combine_priors_hqpr(spoqc_tmp_folder, image_ddf, belief_name, mask_name):
    image_ddf = image_ddf.rename(
        columns={
            "norm_p_pixel_score": belief_name,
            "pixel_score_mask": mask_name,
        }
    )
    return image_ddf


def combine_priors_hqtr(spoqc_tmp_folder, image_ddf, belief_name, mask_name):
    qv_ddf = dd.read_parquet(
        f"{spoqc_tmp_folder}/hqtr_output_qv_prob",
        columns=["norm_p_qv_density"],
        engine="pyarrow",
        calculate_divisions=True,
    )

    ac_ddf = dd.read_parquet(
        f"{spoqc_tmp_folder}/hqtr_output_ac_prob",
        columns=["norm_p_ac_density"],
        engine="pyarrow",
        calculate_divisions=True,
    )

    # Keep everything lazy / partitioned
    belief = (
        image_ddf["norm_p_pixel_score"]
        + qv_ddf["norm_p_qv_density"]
        + ac_ddf["norm_p_ac_density"]
    )

    image_ddf = image_ddf.assign(**{belief_name: belief})
    num_priors = 3.0
    scaled = image_ddf[belief_name] / num_priors

    return image_ddf.assign(
        **{
            belief_name: scaled,
            mask_name: (scaled > 0.5).astype("int8"),
        }
    )


def read_mask_raw_beliefs(spoqc_tmp_folder, prefix):
    # verbatim from pixel_scoring_refinement.py:55-56
    image_ddf = dd.read_parquet(f'{spoqc_tmp_folder}/{prefix}_output_mask_raw', columns=[f"{prefix}_beliefs"], engine="pyarrow")
    beliefs_raw = image_ddf[f"{prefix}_beliefs"].compute().to_numpy()
    return beliefs_raw
