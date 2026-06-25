
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
        stainings
):

# In[]

    staining_list = [0]
    if ( len(stainings) > 1 ):
        staining_list = [str(x) for x in range(0, len(stainings))]
    if ( 'dummy' in stainings ):
        staining_list.remove(str(stainings.index('dummy')))

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
                staining_list,
                annotation
        )
        helperfuncs.sort_files(f'{CONST.FIGURE_PATH}/analysis/overview', 'prefix', ['res.txt', 'done.txt'])
        print(f"[finish] {CONST.STEP}")


    # # In[]
    # if ( CONST.STEP in ['all', 'analysis_cluster'] and CONST.ANNOTATION_FILE):
    #     additional_analysis.cluster_analysis.celltype_cluster_analysis(
    #             sdata,
    #             'cluster',
    #             CONST,
    #             seed,
    #             'raw',
    #             dim_x,
    #             dim_y,
    #             imagedim,
    #             staining_list,
    #             annotation,
    #     )
    #     helperfuncs.sort_files(f'{CONST.FIGURE_PATH}/analysis/cluster', 'prefix', ['res.txt', 'done.txt'])
    #     print(f"[finish] {CONST.STEP}")

    # # In[]
    # if ( CONST.STEP in ['all', 'analysis_category'] and CONST.ANNOTATION_FILE):
    #     additional_analysis.category_analysis.cell_category_analysis(
    #             sdata,
    #             'category',
    #             CONST,
    #             seed,
    #             'raw',
    #             dim_x,
    #             dim_y,
    #             imagedim,
    #             staining_list,
    #     )
    #     print(f"[finish] {CONST.STEP}")
