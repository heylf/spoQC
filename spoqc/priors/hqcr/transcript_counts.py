import pandas as pd
import numpy as np

from ... import helperfuncs
from ... import core

# For each cell calculate the bad quality probability, which is basically the poportion of 
# all the cells in a distance beloning to the bad quality cluster.
def _get_bad_quality_probability(x, df, distance_matrix, bad_cluster, qc_cluster):
    quality_clusters = df.iloc[distance_matrix[x]][qc_cluster]
    number_of_bad_quality_cells = list(quality_clusters.values).count(bad_cluster)
    if ( len(quality_clusters) != 0 ):
        return(number_of_bad_quality_cells/len(quality_clusters))
    else:
        return(0.0)


def _reduce_cluster_num_for_hqcr(cell_df, qc_domains_adata, figure_path, counts):

    # Identify cluster of lowest quality and cluster of highest quality
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

    return n


def _calc_counts_probs(sdata, figure_path, cell_df, qc_domains_adata, counts, thres_counts = 1.0):

    # Work on local copies, the cell_clustering_df/adata are shared with the other priors.
    shared_cell_df = cell_df
    cell_df = cell_df.copy()
    qc_domains_adata = qc_domains_adata.copy()

    n_clusters = _reduce_cluster_num_for_hqcr(cell_df, qc_domains_adata, figure_path, counts)

    # The HQCR html report groups the cells by this column, so write it back on purpose.
    shared_cell_df['qc_cluster_str'] = cell_df['qc_cluster_str'].values

    # Get bad cluster
    # With less than 3 leiden clusters some quality levels are empty (NaN means), so use the hard threshold.
    if ( n_clusters >= 3 ):
        mean_counts = [np.mean(cell_df.loc[cell_df['qc_cluster'] == c][counts]) for c in [0,1,2]]
        bad_cluster = np.argmin(mean_counts)
        t = np.min(mean_counts)

    # Apply hard threshold just to check if the bad cluster is really bad and not just a specific domain.
    if ( n_clusters < 3 or t > thres_counts ):
        print(f"[NOTE] Bad cluster is actually not bad or less than 3 leiden clusters ({n_clusters}). " + \
            f"Switching to hard theshold of {thres_counts} transcripts per cell")
        hard_qc_clusters = np.zeros(len(cell_df))
        hard_qc_clusters[cell_df[counts] > thres_counts] = 1
        cell_df['qc_cluster'] = hard_qc_clusters
        bad_cluster = 0
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

    distance_matrix = helperfuncs.points_within_radius(df_coords, 30, False)
    bad_quality_probabilities =  np.array([_get_bad_quality_probability(
        x,
        cell_df,
        distance_matrix,
        bad_cluster,
        'qc_cluster'
    ) for x in range(sdata['table'].n_obs)])
    good_quality_probabilities = 1 - bad_quality_probabilities

    return good_quality_probabilities


def init_prior(enterprise):

    # These have to be defined.
    name = "transcript_counts_prior"
    modality = "hqcr"
    tmp_path = None
    needs_metrics = ["sc_metrics"]

    counts = "transcript_counts"
    if enterprise.args.canorm:
        counts = "canorm_transcript_counts"

    # These are given by your prior calc function.
    args = [enterprise.cargo.sdata, f'{enterprise.args.output_dir}/hqcr/hqcr_ident/',
            enterprise.hqcr_set.cell_clustering_df, enterprise.hqcr_set.cell_clustering_adata,
            counts]
    kwargs = None

    prior = core.prior.Prior(
        _calc_counts_probs, 
        name,
        modality,
        needs_metrics = needs_metrics,
        tmp_path = tmp_path,
        args = args,
        kwargs = kwargs,
    )    
    
    return prior
