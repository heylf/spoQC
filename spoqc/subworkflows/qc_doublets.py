import numpy as np

from .. import helperfuncs
from .. import metrics

def run_qc_doublets(enterprise):
    if enterprise.args.step in ['all', 'unittest', 'doubletqc']:
        figure_path = f'{enterprise.args.output_dir}/doubletqc/'
        timer = helperfuncs.Timer()


        ncelltypes = -1

        if ( enterprise.args.annotation_file and enterprise.cargo.celltype_annotation.ncelltypes == None ):
            ncelltypes = enterprise.cargo.celltype_annotation.ncelltypes
        else:
            ncelltypes = enterprise.args.ncelltypes
        
        mean_diameter = np.mean(enterprise.cargo.sdata['cell_circles']['radius'])*2
        print(f"[NOTE] Estimated cell diameter is {mean_diameter}")

        print(f"[NOTE] Doublet QC with {ncelltypes} estimated celltypes")
        timer.start()
        metrics.segmentation.doublet_score.calc_doublet_score(
            enterprise.cargo.sdata,
            figure_path,
            enterprise.args.tmp_dir,
            enterprise.args.nthreads,
            'transcripts',
            ncelltypes,
            mean_diameter,
        )
        timer.stop()

        print("[NOTE] Calculate overlap areas")
        timer.start()
        metrics.segmentation.overlap_area.calculate_overlap_areas(enterprise.cargo.sdata)
        timer.stop()

        print("[NOTE] Write results")
        helperfuncs.sdata_obs_to_parquet(
            enterprise,
            figure_path,
            'hqcr'
        )
        print("[finish]")