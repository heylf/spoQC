import os

from .. import additional_analysis
from .. import helperfuncs

def run_qc_additional_analysis(
        sdata,
        CONST,
        annotation,
        seed,
        imagedim,
        dim_x,
        dim_y,
    ):
    
    # In[]
    performed_stainings = [int(x) for x in os.listdir(f'{CONST.TMP_PATH}/metrices/hqpr')]

    # In[]
    if ( CONST.STEP in ['all', 'analysis_overview'] and CONST.ANNOTATION_FILE):
        additional_analysis.cluster_analysis.celltype_cluster_analysis(
                sdata,
                'overview',
                CONST,
                seed,
                'raw',
                dim_x,
                dim_y,
                imagedim,
                performed_stainings,
                annotation
        )
        helperfuncs.sort_files(f'{CONST.FIGURE_PATH}/analysis/overview', 'prefix', ['res.txt', 'done.txt'])
        print(f"[finish] {CONST.STEP}")

    # In[]
    if ( CONST.STEP in ['all', 'analysis_cluster'] and CONST.ANNOTATION_FILE):
        additional_analysis.cluster_analysis.celltype_cluster_analysis(
                sdata,
                'cluster',
                CONST,
                seed,
                'raw',
                dim_x,
                dim_y,
                imagedim,
                performed_stainings,
                annotation,
        )
        helperfuncs.sort_files(f'{CONST.FIGURE_PATH}/analysis/cluster', 'prefix', ['res.txt', 'done.txt'])
        print(f"[finish] {CONST.STEP}")

    # In[]
    if ( CONST.STEP in ['all', 'analysis_category'] and CONST.ANNOTATION_FILE):
        additional_analysis.category_analysis.cell_category_analysis(
                sdata,
                'category',
                CONST,
                seed,
                'raw',
                dim_x,
                dim_y,
                imagedim,
                performed_stainings,
        )
        print(f"[finish] {CONST.STEP}")
