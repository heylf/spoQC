
from .. import helperfuncs
from .. import metrics

def run_qc_bubble(enterprise):
    if enterprise.args.step in ['all', 'unittest', 'bubbleqc']:
        figure_path = f'{enterprise.args.output_dir}/bubbleqc/'
        
        metrics.segmentation.bubble_score.calc_bubble_score(enterprise.cargo.sdata, figure_path, 'cell_boundaries')
        
        print("[NOTE] Write results")
        helperfuncs.sdata_obs_to_parquet(
            enterprise,
            figure_path,
            'hqcr'
        )
        print("[finish]")

