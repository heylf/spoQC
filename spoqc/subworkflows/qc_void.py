
from .. import helperfuncs
from .. import metrics

def run_qc_void(sdata, figure_path, CONST, obs_columns):

    print("[NOTE] Void QC")
    # You can also provide contaminants with:
    # void.qc_void.voidqc(sdata, figure_path, CONST.TMP_PATH, 30, ['CD3D', 'CD14', 'CD68'], CONST.THREADS)
    metrics.segmentation.void.calc_void(sdata, figure_path, CONST.TMP_PATH, 30, [])

    print("[NOTE] Write results")
    obs_columns = helperfuncs.sdata_obs_to_parquet(sdata, figure_path, CONST.TMP_PATH, 'hqcr', obs_columns)
    print("[finish]")

    # helperfuncs.plot_scatter_density(
    #     sdata['table'], figure_path, None, 
    #     1, None, 'convexhull_all_transcripts', None, 'Density of convexhull'
    # )

    return obs_columns