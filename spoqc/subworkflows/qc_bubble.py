
from .. import helperfuncs
from .. import metrics

def run_qc_bubble(sdata, figure_path, CONST, obs_columns):
    metrics.segmentation.bubble_score.calc_bubble_score(sdata, figure_path, 'cell_boundaries')
    print("[NOTE] Write results")
    obs_columns = helperfuncs.sdata_obs_to_parquet(sdata, figure_path, CONST.TMP_PATH, 'hqcr', obs_columns)
    print("[finish]")
