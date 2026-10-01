import os
import sys
import dask
import dask.array as da
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

from .. import helperfuncs
from .. import metrics
from .. import priors
from ..core import parquet

N_CLUSTERS = 100

# The k-means features of each modality, in column order: the metrics structure analysis writes,
# in the order it writes them. origin/dev took every *{suffix}.parquet in os.listdir order,
# which the filesystem decides (on weka it differs between directories).
PIXEL_FEATURE_NAMES = {
    'hqpr': ['intensity', 'lbp', 'edge_strength', 'energy', 'relevance', 'entropy', 'uniformity', 'homogenity'],
    'hqtr': ['transcript_density', 'lbp', 'edge_strength', 'energy', 'relevance', 'entropy', 'uniformity', 'homogenity'],
}


def pixel_feature_files(spoqc_tmp_folder, modality, suffix):
    """The metric files of PIXEL_FEATURE_NAMES[modality]; raises if one is missing or a stray one is present."""
    expected = [f'{name}_output_{suffix}.parquet' for name in PIXEL_FEATURE_NAMES[modality]]
    present = {file for file in os.listdir(spoqc_tmp_folder) if file.endswith(f'{suffix}.parquet')}
    missing = [file for file in expected if file not in present]
    unexpected = sorted(present - set(expected))
    if missing or unexpected:
        raise ValueError(f"[ERROR] Pixel metrics in {spoqc_tmp_folder}: missing {missing}, unexpected {unexpected}")
    return [f'{spoqc_tmp_folder}/{file}' for file in expected]


def cluster_pixels(features, n_clusters, seed, chunk_size, threads, sample_size=5_000_000):
    """MiniBatchKMeans labels (int32) for every row of `features`.

    Fits once on an i.i.d. subsample, then predicts every chunk_size block. The subsample mask
    is drawn from the same dask RNG stream (per-chunk seeds of default_rng(seed) over
    chunk_size chunks) as origin/dev, so the fit sees the same rows in the same order.
    """
    n_rows = features.shape[0]
    frac = min(1.0, sample_size / n_rows)
    sample_mask = (da.random.default_rng(seed).random(n_rows, chunks=(chunk_size,)) < frac).compute()
    sample_np = features[sample_mask]
    del sample_mask

    labels = np.empty(n_rows, dtype=np.int32)
    with threadpool_limits(limits=threads, user_api="openmp"):
        est = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            batch_size=min(chunk_size, len(sample_np)),
            reassignment_ratio=0.01,
            n_init=3,
        )
        est.fit(sample_np)
        # predict runs its own OpenMP loop over 256-row chunks of each block.
        for start in range(0, n_rows, chunk_size):
            labels[start:start + chunk_size] = est.predict(features[start:start + chunk_size])
    return labels


def load_intensity_image(sdata, image_type, resolution, staining):
    return np.asarray(sdata[image_type][resolution].image.data[int(staining)])


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
        background_intensity=None,
        gmm_n_init,
    ):
    """Cluster pixels on their metrics, score the clusters and write {prefix}_output_mask_raw.

    background_intensity: the hqpr staining's background from structure analysis; None
    recomputes it from the image. Returns the per-pixel beliefs (float64) for refinement.
    """

    timer = helperfuncs.Timer()
    timer_all = helperfuncs.Timer()
    timer_all.start()

    if ( modality not in ['hqpr', 'hqtr'] ):
        sys.exit(f'[ERROR] Modality {modality} does not exist')

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

    with dask.config.set(scheduler="threads", num_workers=threads):
        print('[NOTE] Agglomerate pixel metrices and cluster')
        timer.start()
        feature_files = pixel_feature_files(spoqc_tmp_folder_metrices, modality, tmp_suffix)
        feature_names = PIXEL_FEATURE_NAMES[modality]
        features = helperfuncs.read_pixel_features(feature_files, threads)
        print(f"[NOTE] Pixel features {feature_names}")
        timer.stop()

        print("[NOTE] Clustering (fit on subsample, predict on full array in parallel)")
        timer.start()
        clusters = cluster_pixels(features, N_CLUSTERS, seed, chunk_size, threads, sample_size)
        timer.stop()

        print("[NOTE] Structure and anti-structure scores, intensities")
        timer.start()
        s_score = metrics.image.pixel_score.summed_score(
            features, feature_names, metrics.image.pixel_score.STRUCTURE_METRICS, chunk_size, threads)
        as_score = metrics.image.pixel_score.summed_score(
            features, feature_names, metrics.image.pixel_score.ANTI_STRUCTURE_METRICS, chunk_size, threads)
        if ( modality == 'hqtr' ):
            # Intensities already flipped; hqtr has no background (constant 0.0).
            intensity = features[:, feature_names.index('transcript_density')].copy()
            background_intensity = 0.0
        del features
        if ( modality == 'hqpr' ):
            image = load_intensity_image(sdata, image_type, resolution, staining)
            if ( background_intensity is None ):
                background_intensity, _, _ = metrics.image.utility.estimate_background_intensity(image)
            intensity = np.flipud(image).ravel()
            del image
        timer.stop()

        #####################
        ###### Metrics ######
        #####################
        # We calculate pixel scores for each pixel cluster.
        pixel_scores_ds, clusters_ids = metrics.image.pixel_score.calc_pixel_score(
            figure_path,
            clusters,
            s_score,
            as_score,
            intensity,
            background_intensity,
            dim_x,
            dim_y,
            imagedim,
            plot_all_pixel_clusters,
            N_CLUSTERS,
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
            t=thresh_p,
            seed=seed,
            n_init=gmm_n_init,
        )

        ########################
        ###### Downstream ######
        ########################

        # Map cluster densities to each pixel.
        cluster_probs = np.full(N_CLUSTERS, np.nan)
        cluster_probs[clusters_ids] = prob_densities
        p_informative_pixel = cluster_probs[clusters]

        # Min-Max normalization
        print("[NOTE] Min-max normalization")
        timer.start()
        norm_p_pixel_score = helperfuncs.min_max_normalize(p_informative_pixel, threads)
        pixel_score_mask = (norm_p_pixel_score > 0.5).astype(int)
        timer.stop()

        helperfuncs.plot_pixels(
            figure_path,
            norm_p_pixel_score.reshape(dim_x, dim_y),
            imagedim,
            'norm_p_pixel_score',
            'Normalized pixel score probability',
            'hot',
            False,
            False
        )

        print("[NOTE] Combining priors")
        timer.start()
        columns = {
            'cluster': clusters,
            's_score': s_score,
            'as_score': as_score,
            'intensity': intensity,
            'p_informative_pixel': p_informative_pixel,
        }
        if modality == 'hqpr':
            belief_name = f"{modality}_{staining}_beliefs"
            prior_columns = priors.combine_priors.combine_priors_hqpr(
                norm_p_pixel_score, pixel_score_mask, belief_name, f"{modality}_{staining}_mask")
        if modality == 'hqtr':
            belief_name = f"{modality}_beliefs"
            prior_columns = priors.combine_priors.combine_priors_hqtr(
                spoqc_tmp_folder, norm_p_pixel_score, pixel_score_mask, belief_name, f"{modality}_mask")
        columns = columns | prior_columns
        n_rows = len(clusters)
        beliefs = columns[belief_name]
        timer.stop()

        helperfuncs.plot_pixels(
            figure_path,
            beliefs.reshape(dim_x, dim_y),
            imagedim,
            'norm_p_beliefs',
            'Normalized combined probability (beliefs)',
            'hot',
            False,
            False
        )

        print("[NOTE] Writing out data")
        timer.start()
        parquet.write_parts(f"{spoqc_tmp_folder}/{tmp_suffix}_output_mask_raw", n_rows,
                            parquet.columns_of(columns), range(0, n_rows, parquet.PART_ROWS), threads)
        timer.stop()

    print("[NOTE] The pixel clustering and prior estimation took:")
    timer_all.stop()
    return beliefs

# %%
