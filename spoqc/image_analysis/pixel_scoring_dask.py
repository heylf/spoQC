import os
import dask
import dask.dataframe as dd
import dask.array as da
import numpy as np
import sys

from scipy.stats import norm
from dask_ml.preprocessing import MinMaxScaler
from dask_ml.wrappers import Incremental
from sklearn.cluster import MiniBatchKMeans

from .. import helperfuncs
from .. import metrics
from .. import priors

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


def dask_clustering_mini_batches(spoqc_tmp_folder, suffix, n_clusters, seed, chunk_size, threads):
    tmp_files = [f'{spoqc_tmp_folder}/{file}' for file in os.listdir(spoqc_tmp_folder)
             if file.endswith(f'{suffix}.parquet')]

    timer = helperfuncs.Timer()

    with dask.config.set(scheduler="threads", num_workers=threads):
        print("[NOTE] Read data")
        dask_array = read_data_as_ddf(tmp_files, chunk_size)

        print("[NOTE] Clustering")
        timer.start()
        est = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            batch_size=chunk_size,          # match your chunk size
            reassignment_ratio=0.01
        )
        inc = Incremental(est)
        inc.fit(dask_array)                 # streamed; low peak memory
        labels = inc.predict(dask_array)    # dask.array[int32], chunked
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
        staining=None,
        thresh_p=None,
        nstds_p=3
    ):

    timer = helperfuncs.Timer()
    timer_all = helperfuncs.Timer()
    timer_all.start()

    # Just path variables
    tmp_suffix = modality
    spoqc_tmp_folder_metrices = ''
    if ( staining ):
        spoqc_tmp_folder_metrices = f'{spoqc_tmp_folder}/metrices/{modality}/{staining}'
        figure_path = f'{figure_path}/{modality}/{staining}/{modality}_clustering/'
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
        threads
    ))
    print("[NOTE] Time for the whole clustering process:")
    timer.stop()
    image_ddf = image_ddf.persist()

    #####################
    ###### Metrics ######
    #####################
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

    # Map cluster probabilites to each pixel.
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

    # I only normlize cluster probs and not all pixel probs.
    scaled_ddf = scaler.fit_transform(image_ddf[['p_informative_pixel']])
    image_ddf = image_ddf.assign(
        **{
            'norm_p_pixel_score': scaled_ddf.iloc[:, 0],
            'pixel_score_mask': (scaled_ddf.iloc[:, 0] > 0.5).astype(int),
        }
    )
    timer.stop()
    image_ddf = image_ddf.persist()

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
    timer.stop()
    image_ddf = image_ddf.persist()

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
