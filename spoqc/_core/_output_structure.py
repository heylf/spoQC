import os

def create_output_structure(args):

    if ( not os.path.exists(f"{args.output_dir}") ):
            os.makedirs(f"{args.output_dir}")

    # For the final report we do not need a folder.
    if args.step == "final_report":
        return 0

    dirs = [args.step]

    if args.step == "all":
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
            f"hqpr/hqpr_metrices/{args.staining}",
            f"hqpr/hqpr_clustering/{args.staining}",
            f"hqpr/hqpr_refinement/{args.staining}",
            f"hqpr/hqpr_bounding_box/{args.staining}",
            f"hqpr/hqpr_bounding_box/{args.staining}/subfigures",
            f"hqpr/hqpr_celltype/{args.staining}",
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
            f"combine_masks/{args.staining}",
            "analysis/overview",
            "analysis/cluster",
            "analysis/category",
        ]
    else:
        prefix_dir = args.step.split("_")[0]

        if ( prefix_dir in ["hqcr", "hqtr"] ):
            dirs = [f"{prefix_dir}/{args.step}"]
        if ( prefix_dir == "hqpr" ):
            dirs = [f"{prefix_dir}/{args.step}/{args.staining}"]
            
        if ( args.step == "hqcr_ident" ):
            dirs.extend([
                "hqcr/hqcr_ident/small_hqcr",
                "hqcr/hqcr_ident/lqcr",
                "hqcr/hqcr_ident/hqcr"
            ])

        if ( args.step == f"{prefix_dir}_bounding_box" ):
            if ( prefix_dir == "hqpr" ):
                dirs.append(f"{prefix_dir}/{args.step}/{args.staining}/subfigures")
            else:
                dirs.append(f"{prefix_dir}/{args.step}/subfigures")

        if ( args.step == "combine_masks" ):
            dirs = [f"combine_masks/{args.staining}"]

        if ( prefix_dir == "analysis" ):
            dirs = [f"analysis/{args.step.split("_")[1]}"]

    for dir in dirs:
        if ( not os.path.exists(f"{args.output_dir}/{dir}") ):
            os.makedirs(f"{args.output_dir}/{dir}")

    if ( not os.path.exists(f"{args.tmp_dir}/metrices/hqpr/{args.staining}") ):
        os.makedirs(f"{args.tmp_dir}/metrices/hqpr/{args.staining}")
    if ( not os.path.exists(f"{args.tmp_dir}/metrices/hqtr/") ):
        os.makedirs(f"{args.tmp_dir}/metrices/hqtr/")