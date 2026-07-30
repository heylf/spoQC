import os

def create_folder_structure(CONST):

    # For the final report we do not need a folder.
    if CONST.STEP == "final_report":
        return 0

    dirs = [CONST.STEP]

    if CONST.STEP == "all":
        dirs = [
            "annotation",
            "generalqc",    
            "whole_slide_qc",
            "bubbleqc",
            "doubletqc",
            "voidqc",
            "cellqc",
            "hqcr/hqcr_ident",
            "hqcr/hqcr_ident/small_hqcr",
            "hqcr/hqcr_ident/lqcr",
            "hqcr/hqcr_ident/hqcr",
            "hqcr/hqcr_celltype",
            f"hqpr/{CONST.STAINING}/hqpr_metrices",
            f"hqpr/{CONST.STAINING}/hqpr_clustering",
            f"hqpr/{CONST.STAINING}/hqpr_refinement",
            f"hqpr/{CONST.STAINING}/hqpr_bounding_box",
            f"hqpr/{CONST.STAINING}/hqpr_bounding_box/subfigures",
            f"hqpr/{CONST.STAINING}/hqpr_celltype",
            f"hqtr/hqtr_metrices",
            "hqtr/hqtr_qv",
            "hqtr/hqtr_ac",
            "hqtr/hqtr_clustering",
            "hqtr/hqtr_refinement",
            "hqtr/hqtr_bounding_box",
            "hqtr/hqtr_bounding_box/subfigures",
            "hqtr/hqtr_celltype",
            "transcriptqc",
            "cellcycleqc",
            "modelqc",
            "ambientqc",
            "markerqc",
            f"combine_masks/{CONST.STAINING}",
            "analysis/overview",
            "analysis/cluster",
            "analysis/category",
        ]
    else:
        prefix_dir = CONST.STEP.split("_")[0]

        if ( prefix_dir in ["hqcr", "hqtr"] ):
            dirs = [f"{prefix_dir}/{CONST.STEP}"]
        if ( prefix_dir == "hqpr" ):
            dirs = [f"{prefix_dir}/{CONST.STAINING}/{CONST.STEP}"]
            
        if ( CONST.STEP == "hqcr_ident" ):
            dirs.extend([
                "hqcr/hqcr_ident/small_hqcr",
                "hqcr/hqcr_ident/lqcr",
                "hqcr/hqcr_ident/hqcr"
            ])

        if ( CONST.STEP == f"{prefix_dir}_bounding_box" ):
            if ( prefix_dir == "hqpr" ):
                dirs.append(f"{prefix_dir}/{CONST.STAINING}/{CONST.STEP}/subfigures")
            else:
                dirs.append(f"{prefix_dir}/{CONST.STEP}/subfigures")

        if ( CONST.STEP == "combine_masks" ):
            dirs = [f"combine_masks/{CONST.STAINING}"]

        if ( prefix_dir == "analysis" ):
            dirs = [f"analysis/{CONST.STEP.split("_")[1]}"]

    for dir in dirs:
        if ( not os.path.exists(f"{CONST.FIGURE_PATH}/{dir}") ):
            os.makedirs(f"{CONST.FIGURE_PATH}/{dir}")

    if ( not os.path.exists(f"{CONST.TMP_PATH}/metrices/hqpr/{CONST.STAINING}") ):
        os.makedirs(f"{CONST.TMP_PATH}/metrices/hqpr/{CONST.STAINING}")
    if ( not os.path.exists(f"{CONST.TMP_PATH}/metrices/hqtr/") ):
        os.makedirs(f"{CONST.TMP_PATH}/metrices/hqtr/")