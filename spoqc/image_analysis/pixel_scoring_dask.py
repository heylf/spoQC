import os
import dask.dataframe as dd
import dask.array as da
import pandas as pd
import numpy as np
import sys

from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from dask_ml.preprocessing import MinMaxScaler
from dask_ml.wrappers import Incremental
from sklearn.cluster import MiniBatchKMeans

from .. import helperfuncs
from .. import image_analysis
from .. import metrics
from .. import hqr

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


def dask_clustering_mini_batches(spoqc_tmp_folder, suffix, n_clusters, seed, chunk_size, figure_path):
    tmp_files = [f'{spoqc_tmp_folder}/{file}' for file in os.listdir(spoqc_tmp_folder)
             if file.endswith(f'{suffix}.parquet')]

    timer = helperfuncs.Timer()

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


def dask_summify(spoqc_tmp_folder, suffix, metrices, chunk_size):
    tmp_files = [f'{spoqc_tmp_folder}/{metric}_output_{suffix}.parquet' for metric in metrices]
    ddf = read_data_as_ddf(tmp_files, chunk_size)
    structure_scores = ddf.sum(axis=1)
    return structure_scores


def calc_probs_pixel_score(pixel_scores, gmm_mod=3, nstds=None, t=None):
    mix = GaussianMixture(n_components=gmm_mod, tol=1e-8, max_iter=int(1e4))
    mix.fit(pixel_scores.reshape(-1, 1))
    means = mix.means_
    cov = mix.covariances_
    stds = [ np.sqrt(  np.trace(cov[i])/gmm_mod) for i in range(0,gmm_mod) ]
    max_std = stds[np.argmax(means)]

    max_mean = -1
    if ( t ):
        max_mean = t
        max_std = 1.0  # Since mean is hard picked, we will use unit variance.
    else:
        max_mean = np.max(means)

    print(f'Using std {max_std} and mean {max_mean} for pixel prior')

    # Calculate the probability density at x for each pixel clusters.
    if ( nstds ):
        return norm.pdf(pixel_scores, loc=max_mean, scale=nstds*max_std)
    else:
        return norm.pdf(pixel_scores, loc=max_mean, scale=3*max_std)  # 3 stds is default

def calc_prob_pixel_stuff(image_ddf, thresh, std, tail, col):

    # Calculate PDF values directly
    image_ddf = image_ddf.assign(
        **{f'p_{col}': image_ddf[col].map_partitions(
            lambda s: norm.pdf(s, loc=thresh, scale=std), meta=(f'p_{col}', 'f8'))}
    )

    # Apply tail logic using map_partitions (1.0 overwrite rule)
    if tail == 'left':
        image_ddf = image_ddf.assign(
            **{f'p_{col}': image_ddf[[col, f'p_{col}']].map_partitions(
                lambda df: df.apply(lambda row: 1.0 if row[col] < thresh else row[f'p_{col}'], axis=1),
                meta=(f'p_{col}', 'f8'))}
        )
    elif tail == 'right':
        image_ddf = image_ddf.assign(
            **{f'p_{col}': image_ddf[[col, f'p_{col}']].map_partitions(
                lambda df: df.apply(lambda row: 1.0 if row[col] > thresh else row[f'p_{col}'], axis=1),
                meta=(f'p_{col}', 'f8'))}
        )

    # Invert the probabilities
    image_ddf = image_ddf.assign(
        **{f'p_{col}': 1.0 - image_ddf[f'p_{col}']}
    )

    # Min-Max normalize using dask-ml
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(image_ddf[[f'p_{col}']])
    scaled_series = scaled_df.iloc[:, 0]

    image_ddf = image_ddf.assign(
        **{f'norm_p_{col}': scaled_series}
    )

    return image_ddf


def calc_prob_pixel_stuff_v2(image_ddf, thresh, std, tail, col):

    if std <= 0:
        raise ValueError("std must be > 0")

    inv_std = 1.0 / std
    norm_const = inv_std / np.sqrt(2.0 * np.pi)

    def _part(part: pd.DataFrame) -> pd.Series:
        x = part[col].to_numpy()
        # Gaussian PDF centered at `thresh`
        z = (x - thresh) * inv_std
        pdf = norm_const * np.exp(-0.5 * z * z)

        # Tail overwrite to 1.0 (then we'll invert below)
        if tail == "left":
            pdf = np.where(x < thresh, 1.0, pdf)
        elif tail == "right":
            pdf = np.where(x > thresh, 1.0, pdf)
        # else: no tail tweak

        out = 1.0 - pdf
        return pd.Series(out, index=part.index, name=f"p_{col}")

    p_series = image_ddf.map_partitions(_part, meta=(f"p_{col}", "f8"))
    image_ddf = image_ddf.assign(**{f"p_{col}": p_series})

    # Min-Max normalize using dask-ml
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(image_ddf[[f'p_{col}']])
    scaled_series = scaled_df.iloc[:, 0]

    image_ddf = image_ddf.assign(
        **{f'norm_p_{col}': scaled_series}
    )

    return image_ddf


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
        nstds_p=None
    ):

    timer = helperfuncs.Timer()

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
    image_ddf = image_ddf.assign(cluster = dask_clustering_mini_batches(spoqc_tmp_folder_metrices, tmp_suffix, 
                                                           n_clusters, seed, chunk_size, figure_path))
    timer.stop()

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

    # Based on the pixel_scores of the individual clusters figure out with GMM which clusters correspond to inforamtion.
    # Based on that you can assign a probability to each cluster that they belong to useful information.
    # Based on that you can assign to each pixel the prabolity of the pixel cluster they belong to.
    prob_densities = calc_probs_pixel_score(np.array(pixel_scores_ds), gmm_mod=3, nstds=nstds_p, t=thresh_p)    

    # Map cluster probabilites to each pixel.
    cluster_prob_map = dict(zip(clusters_ids, prob_densities))
    image_ddf = image_ddf.assign(p_informative_pixel=image_ddf['cluster'].map(cluster_prob_map))

    # Min-Max normalization
    print("[NOTE] Min-max normalization")
    timer.start()
    scaler = MinMaxScaler()
    # TODO I can do min max already before to the probs of pixel clsuter, is then faster because
    # I only normlize cluster probs and not all pixel probs.
    scaled_ddf = scaler.fit_transform(image_ddf[['p_informative_pixel']])
    image_ddf = image_ddf.assign(norm_p_informative_pixel=scaled_ddf.iloc[:,0])
    timer.stop()

    helperfuncs.plot_pixels(
        figure_path,
        image_ddf['norm_p_informative_pixel'].compute().to_numpy().reshape(dim_x, dim_y),
        imagedim,
        'norm_p_informative_pixel',
        'Normalized probability of an informative pixel', 
        'hot',
        False,
        False
    )

    # Transcript image specific
    if ( modality == 'hqtr' ):

        qv_ddf = dd.read_parquet(f'{spoqc_tmp_folder}/hqtr_output_qv_prob', columns=["norm_p_qv_density"],
                                 engine="pyarrow")
        ac_ddf = dd.read_parquet(f'{spoqc_tmp_folder}/hqtr_output_ac_prob', columns=["norm_p_ac_density"],
                                 engine="pyarrow")

        # Informative pixel probability for hqtr = p(structure) + p(good qv) + p(not ambient)
        a = image_ddf['norm_p_informative_pixel'].to_dask_array(lengths=True)
        b = qv_ddf['norm_p_qv_density'].to_dask_array(lengths=True)
        c = ac_ddf['norm_p_ac_density'].to_dask_array(lengths=True)
        series = a + b + c 
        image_ddf = image_ddf.assign(norm_p_informative_pixel=series)

        scaled_ddf = scaler.fit_transform(image_ddf[['norm_p_informative_pixel']])
        image_ddf = image_ddf.assign(norm_p_informative_pixel=scaled_ddf.iloc[:,0])

        helperfuncs.plot_pixels(
            figure_path,
            image_ddf['norm_p_informative_pixel'].compute().to_numpy().reshape(dim_x, dim_y),
            imagedim,
            'norm_p_informative_pixel_hqtr', 
            'Normalized probability of an informative pixel for HQTR', 
            'hot',
            False,
            False
        )

    helperfuncs.ddf_to_parquet(image_ddf, tmp_suffix, spoqc_tmp_folder, [], 'mask_prob')

# %%
