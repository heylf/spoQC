import os
import dask
import dask.dataframe as dd
import dask.array as da
import numpy as np
import sys

from scipy.stats import norm
from dask_ml.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans

from .. import helperfuncs
from .. import metrics
from .. import priors

def dask_clustering_mini_batches(spoqc_tmp_folder, suffix, n_clusters, seed, chunk_size, threads, sample_size=5_000_000):
    tmp_files = [f'{spoqc_tmp_folder}/{file}' for file in os.listdir(spoqc_tmp_folder)
             if file.endswith(f'{suffix}.parquet')]

    timer = helperfuncs.Timer()

    with dask.config.set(scheduler="threads", num_workers=threads):
        print("[NOTE] Read data")
        dask_array = helperfuncs.read_data_as_ddf(tmp_files, chunk_size)
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
    pixel_scores_ds, clusters_ids, image_ddf = metrics.image.pixel_score.calc_pixel_score(
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
    prob_densities = priors.hqpr.pixel_score.calc_probs_pixel_score(
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
        image_ddf = priors.combine_priors.combine_priors_hqpr(spoqc_tmp_folder, image_ddf, belief_name, mask_name)
    if modality == 'hqtr':
        image_ddf = priors.combine_priors.combine_priors_hqtr(spoqc_tmp_folder, image_ddf, belief_name, mask_name)
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
    helperfuncs.ddf_to_parquet(image_ddf, tmp_suffix, spoqc_tmp_folder, [], 'mask_raw')
    timer.stop()

    print("[NOTE] The pixel clustering and prior estimation took:")
    timer_all.stop()

# %%
