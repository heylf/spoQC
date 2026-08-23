import dask.array as da
import numpy as np
import sys

from ... import helperfuncs
from ... import metrics


def dask_summify(spoqc_tmp_folder, suffix, metrices, chunk_size):
    tmp_files = [f'{spoqc_tmp_folder}/{metric}_output_{suffix}.parquet' for metric in metrices]
    ddf = helperfuncs.read_data_as_ddf(tmp_files, chunk_size)
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
        background_intensity, _, _ = metrics.image.utility.estimate_background_intensity_dask(
            sdata, image_type, resolution, staining
        )
        intensity_da = da.flip(
            sdata[image_type][resolution].image.data[int(staining)], axis=0
        ).flatten().rechunk((chunk_size,))
        image_ddf = image_ddf.assign(intensity=intensity_da)
    elif ( modality == 'hqtr' ):
        # Intensities already flipped
        td_file = f'{spoqc_tmp_folder_metrices}/transcript_density_output_hqtr.parquet'
        intensity = helperfuncs.read_data_as_ddf([td_file], chunk_size)[:, 0]
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
