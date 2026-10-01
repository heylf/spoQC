import numpy as np

from .transcript_and_gene_counts import get_bad_quality_probabilities


def calc_celltype_transcript_counts_probs(
        sdata, 
        cell_df, 
        threshold_left_dict, 
        threshold_right_dict, 
        annotation_key,
        qc_metric,
        df_coords,
        threads
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
    bad_quality_probs_celltype = get_bad_quality_probabilities(
        df_coords, cell_df['qc_celltype_class'].to_numpy(), 1, threads
    )
    good_quality_probabilities = 1 - bad_quality_probs_celltype
    
    return good_quality_probabilities, cell_df