import dask.dataframe as dd
import dask.array as da
import numpy as np
import sys

from ... import helperfuncs
from ... import metrics

def read_data_as_ddf(tmp_files, chunk_size):
    
    num_files = len(tmp_files)
    columns = [f'f{i}' for i in range(num_files)]

    # Preallocate a Dask Array with correct shape and chunks

    array_columns = []

    for i, file in enumerate(tmp_files):
        col_ddf = dd.read_parquet(file).iloc[:, 0].reset_index(drop=True)
        col_arr = col_ddf.to_dask_array(lengths=True).rechunk((chunk_size,))
        array_columns.append(col_arr[:, None])  # make 2D for stacking

    # Stack columns into 2D Dask Array
    # TODO remove .astype(np.float32) as soon as I have recaulcuated values as floats
    dask_array = da.hstack(array_columns).astype(np.float32)  # Much cheaper than dd.concat
    dask_array = dask_array.rechunk((chunk_size, -1))
    # Optional but recommended to avoid re-reading Parquet each epoch:
    # da.to_zarr(dask_array, "dask_array.zarr", overwrite=True); dask_array = da.from_zarr("dask_array.zarr")

    # Convert to Dask DataFrame
    # ddfs = dd.from_dask_array(dask_array, columns=columns)

    return dask_array


def dask_summify(spoqc_tmp_folder, suffix, metrices, chunk_size):
    tmp_files = [f'{spoqc_tmp_folder}/{metric}_output_{suffix}.parquet' for metric in metrices]
    ddf = read_data_as_ddf(tmp_files, chunk_size)
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

    # Calcualte structure and antistructure score.
    s_score = dask_summify(spoqc_tmp_folder_metrices, tmp_suffix, 
                           ['edge_strength', 'energy', 'relevance', 'entropy'], chunk_size)
    as_score = dask_summify(spoqc_tmp_folder_metrices, tmp_suffix, ['homogenity', 'uniformity'], chunk_size)

    # Set same index on both score series
    image_ddf = image_ddf.assign(s_score=s_score, as_score=as_score)

    if ( modality == 'hqpr' ):
        intensity = da.from_array(np.flipud(sdata[image_type][resolution].image.values[int(staining)]).flatten(), 
                                  chunks=chunk_size)
        image_ddf = image_ddf.assign(intensity=intensity)
    elif ( modality == 'hqtr' ):
        timer.start()
        # Intensities already flipped
        intensity = da.from_array(
            metrics.transcript_density.transcript_density_image.generate_transcript_density_image(
                sdata,
                figure_path,
                imagedim,
                image_type,
                resolution
            ), 
            chunks=chunk_size
        )
        timer.stop()
        image_ddf = image_ddf.assign(intensity=intensity)
    else:
        sys.exit(f'[ERROR] Modality {modality} does not exist')

    # Background refined image
    print('[NOTE] Refine pixel intensities')
    background_intensity = 0.0

    # TODO potential probelm with memory.
    if ( modality == 'hqpr' ):
        background_intensity, _, _ = metrics.image.utility.estimate_background_intensity(
                                                image_ddf['intensity'].compute().to_numpy()
                                     )

    t=1.5

    if ( background_intensity == 0 ):
        background_intensity = 1

    # Group by cluster and compute mean intensity
    cluster_mean_int_df = image_ddf.groupby('cluster')['intensity'].mean().rename('mean_cluster_intensity').compute()

    # Since kmeans clusters might not find enough clusters I have to get all possible clsuter ids from the dataframe.
    clusters_ids = list(cluster_mean_int_df.index)
    abs_cluster_signla_noise_log2fc = np.zeros(len(clusters_ids))

    background_clusters = []

    for i in range(0, len(clusters_ids)):
        mean_cluster_sigal = cluster_mean_int_df.iloc[i]
        abs_cluster_signla_noise_log2fc[i] = np.abs( np.log2( ( mean_cluster_sigal + 1) / background_intensity ) )
        if ( abs_cluster_signla_noise_log2fc[i] < t ):
            background_clusters.append(i)

    cluster_mean_s_score_ds = image_ddf.groupby('cluster')['s_score'].mean().rename('mean_s_score').compute()
    cluster_mean_as_score_ds = image_ddf.groupby('cluster')['as_score'].mean().rename('mean_as_score').compute()
    pixel_scores_ds = np.round(cluster_mean_s_score_ds - cluster_mean_as_score_ds, 2)

    cluster_array = image_ddf['cluster'].compute().to_numpy()

    # Calculating pixel scores and generate plots to investigate individual pixel clusters.
    # Each pixel cluster get a pixel_score = s_score - as_score.
    for k in clusters_ids:
        cluster_selection = cluster_array == k

        s_score = np.round(cluster_mean_s_score_ds.loc[k],2)
        as_score = np.round(cluster_mean_as_score_ds.loc[k],2)
        pixel_score = pixel_scores_ds.loc[k]

        title = f'Pixel Cluster {k}'
        if ( k in background_clusters ):
            title = f'Background Pixel Cluster {k}'
        title = title + f' with pixel_score {pixel_score:.2f} s_core {s_score:.2f} and as_score {as_score:.2f}'

        if ( plot_all_pixel_clusters ):

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

    helperfuncs.plot_histogram_for_array(np.array(pixel_scores_ds), 20, figure_path, "Pixel scores", "pixel_scores")

    return pixel_scores_ds, clusters_ids, image_ddf
