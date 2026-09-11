import dask
import dask.dataframe as dd
import dask.array as da
import pandas as pd
import numpy as np
import sys
import os

from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
from dask_ml.preprocessing import MinMaxScaler

from ... import helperfuncs
from ... import core


def _dask_summify(spoqc_tmp_folder, suffix, metrices, chunk_size):
    tmp_files = [f'{spoqc_tmp_folder}/{metric}_output_{suffix}.parquet' for metric in metrices]
    ddf = helperfuncs.read_data_as_dda(tmp_files, chunk_size)
    structure_scores = ddf.sum(axis=1)
    return structure_scores


def _calc_pixel_score(
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
    s_score = _dask_summify(spoqc_tmp_folder_metrices, tmp_suffix, 
                           ['edge_strength', 'energy', 'relevance', 'entropy'], chunk_size)
    as_score = _dask_summify(spoqc_tmp_folder_metrices, tmp_suffix, ['homogeneity', 'uniformity'], chunk_size)
    timer.stop()

    # Set same index on both score series
    print("[NOTE] Get intensities")
    timer.start()
    image_ddf = image_ddf.assign(s_score=s_score, as_score=as_score)

    background_intensity = 0.0
    if ( modality == 'hqpr' ):
        # Lazy Dask-native histogram instead of eagerly pulling the whole
        # full-resolution image into a numpy array just to estimate this.
        background_intensity, _, _ = helperfuncs.estimate_background_intensity_dask(
            sdata, image_type, resolution, staining
        )
        intensity_da = da.flip(
            sdata[image_type][resolution].image.data[int(staining)], axis=0
        ).flatten().rechunk((chunk_size,))
        image_ddf = image_ddf.assign(intensity=intensity_da)
    elif ( modality == 'hqtr' ):
        # Intensities already flipped
        td_file = f'{spoqc_tmp_folder_metrices}/transcript_density_output_hqtr.parquet'
        intensity = helperfuncs.read_data_as_dda([td_file], chunk_size)[:, 0]
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


def _calc_probs_pixel_score(pixel_scores, figure_path, gmm_mod=3, nstds=1, t=None, std=None):
    mix = GaussianMixture(n_components=gmm_mod, tol=1e-8, max_iter=int(1e4))
    mix.fit(pixel_scores.reshape(-1, 1))
    means = mix.means_
    cov = mix.covariances_
    stds = [ np.sqrt(np.trace(cov[i])) for i in range(0, gmm_mod) ]
    max_std = stds[np.argmax(means)]

    max_mean = -1
    if ( t ):
        max_mean = t
        max_std = 1.0  # Since mean is hard picked, we will use unit variance.
    else:
        max_mean = np.max(means)

    if ( std ):
        max_std = std

    print(f'Using std {max_std} and mean {max_mean} for pixel prior')

    helperfuncs.plot_histogram_for_array(
        pixel_scores,
        100,
        figure_path,
        f"Pixel scores: t={np.round(max_mean, 3)} with {nstds} x {np.round(max_std, 3)} std",
        "pixel_scores_prior",
        t=max_mean,
        std=max_std,
        nstds=nstds,
    )

    # Calculate the probability density at x for each pixel clusters.
    return norm.pdf(pixel_scores, loc=max_mean, scale=nstds*max_std)


def _dask_clustering_mini_batches(
        spoqc_tmp_folder,
        suffix,
        n_clusters,
        seed,
        chunk_size,
        threads,
        kmeans_sample_size,
    ):

    tmp_files = [f'{spoqc_tmp_folder}/{file}' for file in os.listdir(spoqc_tmp_folder)
             if file.endswith(f'{suffix}.parquet')]

    timer = helperfuncs.Timer()

    with dask.config.set(scheduler="threads", num_workers=threads):
        print("[NOTE] Read data")
        dask_array = helperfuncs.read_data_as_dda(tmp_files, chunk_size)
        n_rows = dask_array.shape[0]

        print("[NOTE] Clustering (fit on subsample, predict on full array in parallel)")
        timer.start()

        # A streamed/incremental fit (dask_ml.wrappers.Incremental) is
        # inherently sequential -- each chunk's partial_fit depends on the
        # previous chunk's centroid state -- which turns into thousands of
        # sequential Python-level calls at full-image pixel counts. Fitting
        # once on a large i.i.d. subsample and then predicting the rest in
        # parallel (map_blocks, no shared state) avoids that entirely.
        frac = min(1.0, kmeans_sample_size / n_rows)
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


def _calc_pixel_score_prior(
        sdata,
        figure_path,
        spoqc_tmp_folder,
        modality,
        image_type,
        resolution,
        imagedim,
        dim_x,
        dim_y,
        seed,
        threads,
        chunk_size,
        kmeans_sample_size,
        *,
        plot_all_pixel_clusters=False,
        staining=None,
        thresh_p=None,
        nstds_p=None,
        col="pixel_score",
    ):

    timer = helperfuncs.Timer()
    timer_all = helperfuncs.Timer()
    timer_all.start()

    # Just path variables
    tmp_suffix = modality
    spoqc_tmp_folder_metrices = ''
    if modality == 'hqpr':
        spoqc_tmp_folder_metrices = f'{spoqc_tmp_folder}/metrices/{modality}/{staining}'
        figure_path = f'{figure_path}/{modality}/{modality}_clustering/{staining}/'
        tmp_suffix = f'{modality}_{staining}'
    else:
        spoqc_tmp_folder_metrices = f'{spoqc_tmp_folder}/metrices/{modality}'
        figure_path = f'{figure_path}/{modality}/{modality}_clustering/'

    # Sanitycheck if files exists
    metrices = ['edge_strength', 'energy', 'relevance', 'entropy', 'homogeneity', 'uniformity']
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
    image_ddf = image_ddf.assign(cluster = _dask_clustering_mini_batches(
        spoqc_tmp_folder_metrices,
        tmp_suffix,
        n_clusters,
        seed,
        chunk_size,
        threads,
        kmeans_sample_size
    ))
    print("[NOTE] Time for the whole clustering process:")
    image_ddf = image_ddf.persist()
    timer.stop()

    #####################
    ###### Metrics ######
    #####################
    # We calculate pixel scores for each pixel cluster.
    pixel_scores_ds, clusters_ids, image_ddf = _calc_pixel_score(
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
        image_ddf,
    )

    ####################
    ###### Priors ######
    ####################
    # Based on the pixel_scores of the individual clusters figure out with GMM which clusters correspond to inforamtion.
    # Based on that you can assign a probability to each cluster that they belong to useful information.
    # Based on that you can assign to each pixel the prabolity of the pixel cluster they belong to.
    prob_densities = _calc_probs_pixel_score(
        np.array(pixel_scores_ds),
        figure_path,
        gmm_mod=3,
        nstds=nstds_p,
        t=thresh_p,
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
    
    # I only generate cluster probs and not all pixel probs.
    scaled_ddf = scaler.fit_transform(image_ddf[['p_informative_pixel']])
    image_ddf = image_ddf.assign(
        **{
            f'prob_{col}': scaled_ddf.iloc[:, 0],
            f'mask_{col}': (scaled_ddf.iloc[:, 0] > 0.5).astype(int),
        }
    )
    image_ddf = image_ddf.persist()
    timer.stop()

    helperfuncs.plot_pixels(
        figure_path,
        image_ddf[f'prob_{col}'].compute().to_numpy().reshape(dim_x, dim_y),
        imagedim,
        'prob_pixel_score',
        'Pixel score probability', 
        'hot',
        False,
        False
    )
    timer_all.stop()

    return image_ddf


def init_prior(enterprise):

    # These have to be defined.
    name = "pixel_score"
    modality = "hqtr"
    tmp_path = None
    needs_metrics = ["edge_strength", "energy", "relevance", "entropy", "homogeneity", "uniformity"]

    # These are given by your prior calc function.
    args = [enterprise.cargo.sdata, enterprise.args.output_dir, enterprise.args.tmp_dir, modality,
            enterprise.args.image_type, enterprise.args.resolution, enterprise.cargo.imagedim,
            enterprise.cargo.dim_x, enterprise.cargo.dim_y, enterprise.args.seed, enterprise.args.nthreads,
            enterprise.args.chunk_size, enterprise.args.kmeans_sample_size]
    kwargs = {"nstds_p": enterprise.args.nstds_prior_pixel, "staining": enterprise.args.staining}

    prior = core.prior.Prior(
        _calc_pixel_score_prior, 
        name,
        modality,
        needs_metrics = needs_metrics,
        tmp_path = tmp_path,
        args = args,
        kwargs = kwargs,
    )    
    
    return prior
