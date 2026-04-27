import pandas as pd
import numpy as np

from ... import helperfuncs

# For each cell calculate the bad quality probability, which is basically the poportion of 
# all the cells in a distance beloning to the bad quality cluster.
def get_bad_quality_probability(x, df, distance_matrix, bad_cluster, qc_cluster):
    quality_clusters = df.iloc[distance_matrix[x]][qc_cluster]
    number_of_bad_quality_cells = list(quality_clusters.values).count(bad_cluster)
    if ( len(quality_clusters) != 0 ):
        return(number_of_bad_quality_cells/len(quality_clusters))
    else:
        return(0.0)


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


def calc_transcript_counts_probs(sdata, figure_path, cell_df, qc_domains_adata, counts):

    reduce_cluster_num_for_hqcr(cell_df, qc_domains_adata, figure_path, counts)

    # Get bad cluster
    mean_counts = [np.mean(cell_df.loc[cell_df['qc_cluster'] == c][counts]) for c in [0,1,2]]
    bad_cluster = np.argmin(mean_counts)

    # Apply hard threshold just to check if the bad cluster is really bad and not just a specific domain.
    thres_transcript_counts = 100
    if ( np.min(mean_counts) > thres_transcript_counts ):
        print(f"[NOTE] Bad cluster is actually not bad." + \
            "Switching to hard theshold of {thres_transcript_counts} transcripts per cell")
        hard_qc_clusters = np.zeros(len(cell_df))
        hard_qc_clusters[cell_df[counts] > thres_transcript_counts] = 1
        cell_df['qc_cluster'] = hard_qc_clusters

    # For each cell calculate the bad quality probability, which is basically the poportion of 
    # all the cells in a distance beloning to the bad quality cluster.
    df_coords = pd.DataFrame({
        'x': sdata['table'].obsm['spatial'][:,0],
        'y': sdata['table'].obsm['spatial'][:,1],
    })

    distance_matrix = helperfuncs.points_within_radius(df_coords, 30, False)
    bad_quality_probabilities =  np.array([get_bad_quality_probability(
        x,
        cell_df,
        distance_matrix,
        bad_cluster,
        'qc_cluster'
    ) for x in range(sdata['table'].n_obs)])
    good_quality_probabilities = 1 - bad_quality_probabilities

    return good_quality_probabilities, cell_df


def calc_celltype_transcript_counts_probs(
        sdata, 
        cell_df, 
        threshold_left_dict, 
        threshold_right_dict, 
        annotation_key,
        qc_metric,
        df_coords
):
    cell_df['qc_celltype_class'] = np.array([0] * len(cell_df))
    for i, celltype in enumerate(threshold_left_dict['celltypes']):
        df_check = cell_df[cell_df[annotation_key] == celltype]

        # Apply left threshold
        idx_qc = df_check[df_check[qc_metric] < threshold_left_dict[qc_metric][i]].index
        cell_df['qc_celltype_class'][idx_qc] = 1 # 1 for beeing bad

        # Apply right threshold
        idx_qc = df_check[df_check[qc_metric] > threshold_right_dict[qc_metric][i]].index
        cell_df['qc_celltype_class'][idx_qc] = 1 # 1 for beeing bad

    # Calculate bad quality probability
    distance_matrix = helperfuncs.points_within_radius(df_coords, 30, False)
    bad_quality_probs_celltype =  np.array([get_bad_quality_probability(
        x,
        cell_df,
        distance_matrix,
        1,
        'qc_celltype_class'
    ) for x in range(sdata['table'].n_obs)])
    good_quality_probabilities = 1 - bad_quality_probs_celltype
    
    return good_quality_probabilities, cell_df