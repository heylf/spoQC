import numpy as np

from .. import helperfuncs
from .. import metrics

def run_qc_doublets(sdata, figure_path, CONST, annotation, obs_columns):
    timer = helperfuncs.Timer()

    mean_diameter = np.mean(sdata['cell_circles']['radius'])*2

    print(f"[NOTE] Estimated cell diameter is {mean_diameter}")

    ncelltypes = -1
    if ( CONST.ANNOTATION_FILE and CONST.N_CELLTYPES == None ):
        ncelltypes = annotation.ncelltypes
    else:
        ncelltypes = CONST.N_CELLTYPES

    timer.start()
    print(f"[NOTE] Doublet QC with {annotation.ncelltypes} estimated celltypes")
    metrics.segmentation.doublet_score.calc_doublet_score(
        sdata,
        figure_path,
        CONST.TMP_PATH,
        CONST.THREADS,
        'transcripts',
        ncelltypes,
        mean_diameter,
        3,
        2,
        3,
        [10, 60],
        1,
        10,
    )
    timer.stop()

    print("[NOTE] Calculate overlap areas")
    timer.start()
    metrics.segmentation.overlap_area.calculate_overlap_areas(sdata)
    timer.stop()

    print("[NOTE] Write results")
    obs_columns = helperfuncs.sdata_obs_to_parquet(sdata, figure_path, CONST.TMP_PATH, 'hqcr', obs_columns)
    print("[finish]")

    return obs_columns