import numpy as np
import pandas as pd

from ... import helperfuncs
from ... import core

def calc_low_qc_transcript_count(
        sdata,
        figure_path,
        *,
        qv_tresh=20 # at 10x Genomics they use a threshold of qv < 20 (see 10x Baysor tutorial)
    ):

    transcript_df = sdata['transcripts'].compute()
    transcript_df = transcript_df.loc[transcript_df['qv'] < qv_tresh]
    cell_id_counts = transcript_df['cell_id'].value_counts()
    cell_id_counts.index.name = 'index'
    cell_id_counts = cell_id_counts[cell_id_counts.index != -1] # Remove the id -1
    cell_id_counts_df = cell_id_counts.sort_index().reset_index()
    cell_id_counts_df.columns = ['index', 'count']
    cell_id_counts_df['index'] = cell_id_counts_df['index'].astype(str)

    # Join and fill missing counts with 0
    merged_df = pd.merge(sdata['table'].obs, cell_id_counts_df, on='index', how='left')
    merged_df['count'] = merged_df['count'].fillna(0).astype(int)
    merged_df = merged_df.rename(columns={'count': 'num_low_qc_transcript'})

    sdata['table'].obs['num_low_qc_transcript'] = np.array(merged_df['num_low_qc_transcript'])
    helperfuncs.plot_scatter_density(
        sdata['table'],
        figure_path,
        'num_low_qc_transcript',
        None,
        'num_low_qc_transcript',
        None,
        'Density low quality transcripts',
    )


def init_metric(enterprise):

    # These have to be defined.
    metric_name = "low_transcript_quality_count"
    combined_metric_name = None
    needs_metrics = []
    step_when_it_is_calculated = ["cellqc", "all"]
    loaded_for_analysis = True
    loaded_for_visualization = True
    prior = False

    # These are given my your metric calc function.
    args = [enterprise.cargo.sdata, f"{enterprise.args.output_dir}/cellqc/"]
    kwargs = None

    metric = core.metric.Metric(
        calc_low_qc_transcript_count, 
        metric_name,
        combined_metric_name = combined_metric_name,
        needs_metrics = needs_metrics,
        step_when_it_is_calculated = step_when_it_is_calculated,
        loaded_for_analysis = loaded_for_analysis,
        loaded_for_visualization = loaded_for_visualization,
        prior = prior,
        args = args,
        kwargs = kwargs,
    )    
    
    return metric