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