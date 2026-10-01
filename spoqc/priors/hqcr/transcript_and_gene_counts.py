import pandas as pd
import numpy as np

from ... import helperfuncs
from ...core import groupreduce, spatial

# Cells closer than this (in obsm['spatial'] units) form a cell's neighbourhood.
NEIGHBOURHOOD_RADIUS = 30


def get_bad_quality_probabilities(df_coords, quality_clusters, bad_cluster, threads):
    """
    For each cell, the proportion of the other cells within NEIGHBOURHOOD_RADIUS whose quality
    cluster equals bad_cluster; 0.0 for a cell without neighbours.

    Parameters:
        df_coords (pd.DataFrame): 'x' and 'y' of the cells, in cell order.
        quality_clusters (np.ndarray): quality cluster of each cell, in cell order.
        bad_cluster: the cluster counted as bad.
        threads (int): neighbour-search threads.
    """
    n_cells = len(df_coords)
    xy = df_coords[['x', 'y']].to_numpy()
    cell_pos, neighbour_pos = spatial.pairs_within(xy, xy, NEIGHBOURHOOD_RADIUS, threads, exclude_self=True)
    n_neighbours = np.bincount(cell_pos, minlength=n_cells)
    n_bad = groupreduce.group_count(cell_pos, quality_clusters[neighbour_pos] == bad_cluster, n_cells)
    return np.divide(n_bad, n_neighbours, out=np.zeros(n_cells), where=n_neighbours != 0)


def reduce_cluster_num_for_hqcr(cell_df, qc_domains_adata, figure_path, counts):

    # Identify cluster of lowest quality and cluster of highest quality
    # df['qc_cluster'] = qc_domains_adata.obs['spatialleiden_3qclvls']
    cell_df['leiden'] = [int(x) for x in qc_domains_adata.obs['leiden']]
    clusters = np.array(list(set(cell_df['leiden'].values)))
    clusters.sort()
    helperfuncs.plot_scatter(qc_domains_adata, figure_path, 'leiden', None, 'leiden', None, None)

    # Shrink down number of leidenclusters into 3 main quality levels (low, mid, high) based QC metrices.
    mean_counts = [np.mean(cell_df.loc[cell_df['leiden'] == c][counts]) for c in clusters]

    n = len(clusters)
    sorted_clsuters = clusters[np.argsort(mean_counts)]
    low = sorted_clsuters[:n//3]
    mid = sorted_clsuters[n//3:2*n//3]
    high = sorted_clsuters[2*n//3:]

    qc_clusters = [-1] * len(cell_df) 
    for i,x in enumerate(cell_df['leiden']):
        if x in low:
            qc_clusters[i] = 0
        if x in mid:
            qc_clusters[i] = 1
        if x in high:
            qc_clusters[i] = 2

    cell_df['qc_cluster'] = qc_clusters
    cell_df['qc_cluster_str'] = [str(x) for x in qc_clusters]
    qc_domains_adata.obs['qc_cluster'] = qc_clusters
    helperfuncs.plot_scatter(qc_domains_adata, figure_path, 'qc_cluster', None, 'qc_cluster', None, None)


def calc_counts_probs(sdata, figure_path, cell_df, qc_domains_adata, counts, thres_counts, threads):

    reduce_cluster_num_for_hqcr(cell_df, qc_domains_adata, figure_path, counts)

    # Get bad cluster
    mean_counts = [np.mean(cell_df.loc[cell_df['qc_cluster'] == c][counts]) for c in [0,1,2]]
    bad_cluster = np.argmin(mean_counts)
    t = np.min(mean_counts)

    # Apply hard threshold just to check if the bad cluster is really bad and not just a specific domain.
    if ( np.min(mean_counts) > thres_counts ):
        print(f"[NOTE] Bad cluster is actually not bad." + \
            "Switching to hard theshold of {thres_counts} transcripts per cell")
        hard_qc_clusters = np.zeros(len(cell_df))
        hard_qc_clusters[cell_df[counts] > thres_counts] = 1
        cell_df['qc_cluster'] = hard_qc_clusters
        t = thres_counts 

    helperfuncs.plot_histogram_for_array(
        cell_df[counts],
        100,
        figure_path,
        f"{counts}: t={np.round(t, 3)} with {0} x {np.round(0.0, 3)} std",
        f"{counts}_prior",
        t=t
    )

    # For each cell calculate the bad quality probability, which is basically the poportion of 
    # all the cells in a distance beloning to the bad quality cluster.
    df_coords = pd.DataFrame({
        'x': sdata['table'].obsm['spatial'][:,0],
        'y': sdata['table'].obsm['spatial'][:,1],
    })

    bad_quality_probabilities = get_bad_quality_probabilities(
        df_coords, cell_df['qc_cluster'].to_numpy(), bad_cluster, threads
    )
    good_quality_probabilities = 1 - bad_quality_probabilities

    return good_quality_probabilities, cell_df