import concurrent.futures

import numpy as np
import pandas as pd

from ... import helperfuncs
from ...core import groupreduce

STRUCTURE_METRICS = ['edge_strength', 'energy', 'relevance', 'entropy']
ANTI_STRUCTURE_METRICS = ['homogenity', 'uniformity']


def summed_score(features, feature_names, metrics, chunk_size, threads):
    """Row sums of the named metric columns, summed as origin/dev did: np.sum over each
    C-ordered (chunk_size, len(metrics)) float32 block."""
    columns = [feature_names.index(metric) for metric in metrics]
    n_rows = features.shape[0]
    score = np.empty(n_rows, dtype=np.float32)

    def sum_block(start):
        block = np.ascontiguousarray(features[start:start + chunk_size, columns])
        score[start:start + chunk_size] = np.sum(block, axis=1, dtype=np.float32)

    with concurrent.futures.ThreadPoolExecutor(threads) as pool:
        list(pool.map(sum_block, range(0, n_rows, chunk_size)))
    return score


# In[]
def calc_pixel_score(
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
        n_clusters,
    ):

    timer = helperfuncs.Timer()
    timer_all = helperfuncs.Timer()
    timer_all.start()

    t=1.5
    if ( background_intensity == 0 ):
        background_intensity = 1

    # Mean intensity, s_score and as_score per cluster, as float64 sums in a fixed order.
    # origin/dev's dask groupby-mean combined partitions in task-completion order, so its
    # last bits (and its cluster order) varied run to run; this is deterministic.
    print("[NOTE] Get mean intensity, s and as scores for each cluster")
    timer.start()
    means = {}
    for name, values in (('intensity', intensity), ('s_score', s_score), ('as_score', as_score)):
        sums, counts = groupreduce.group_sum(clusters, values, n_clusters)
        means[name] = sums / counts
    # Since kmeans clusters might not find enough clusters, keep only the clusters that have pixels, in id order.
    clusters_ids = [int(k) for k in np.flatnonzero(counts)]
    cluster_means_df = pd.DataFrame({name: m[clusters_ids] for name, m in means.items()},
                                    index=pd.Index(clusters_ids, name='cluster'))
    cluster_mean_int_df = cluster_means_df['intensity'].rename('mean_cluster_intensity')
    timer.stop()

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
    if ( plot_all_pixel_clusters ):
        timer.start()
        print("[NOTE] Generate pixel cluster plots")
        for k in clusters_ids:
            cluster_selection = clusters == k

            s_score_k = np.round(cluster_mean_s_score_ds.loc[k],2)
            as_score_k = np.round(cluster_mean_as_score_ds.loc[k],2)
            pixel_score = pixel_scores_ds.loc[k]

            title = f'Pixel Cluster {k}'
            if ( k in background_clusters ):
                title = f'Background Pixel Cluster {k}'
            title = title + f' with pixel_score {pixel_score:.2f} s_core {s_score_k:.2f} and as_score {as_score_k:.2f}'

            # Check the sturucture of those pixel clusters.
            helperfuncs.plot_pixels(
                figure_path,
                cluster_selection.reshape(dim_x, dim_y),
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

    return pixel_scores_ds, clusters_ids
