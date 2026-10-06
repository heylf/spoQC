import os

from .. import additional_analysis
from .. import helperfuncs

def run_qc_additional_analysis(enterprise):
    
    if 'analysis' in enterprise.args.step or enterprise.args.step == 'all':

        performed_stainings = [int(x) for x in os.listdir(f'{enterprise.args.tmp_dir}/metrices/hqpr')]

        if enterprise.args.step in ['all', 'analysis_overview'] and enterprise.args.annotation_file:
            additional_analysis.cluster_analysis.celltype_cluster_analysis(
                    enterprise.cargo.sdata,
                    enterprise.args.output_dir,
                    enterprise.args.tmp_dir,
                    'overview',
                    enterprise.args.seed,
                    'raw',
                    enterprise.cargo.dim_x,
                    enterprise.cargo.dim_y,
                    enterprise.cargo.imagedim,
                    performed_stainings,
                    enterprise.cargo.celltype_annotation,
                    enterprise.args.image_type,
                    enterprise.args.resolution,
                    enterprise.args.canorm,
                    enterprise.args.cluster_celltype,
            )
            helperfuncs.sort_files(f'{enterprise.args.output_dir}/analysis/overview', 'prefix', ['res.txt', 'done.txt'])
            print(f"[finish] {enterprise.args.step}")

        if enterprise.args.step in ['all', 'analysis_cluster'] and enterprise.args.annotation_file:
            additional_analysis.cluster_analysis.celltype_cluster_analysis(
                    enterprise.cargo.sdata,
                    enterprise.args.output_dir,
                    enterprise.args.tmp_dir,
                    'cluster',
                    enterprise.args.seed,
                    'raw',
                    enterprise.cargo.dim_x,
                    enterprise.cargo.dim_y,
                    enterprise.cargo.imagedim,
                    performed_stainings,
                    enterprise.cargo.celltype_annotation,
                    enterprise.args.image_type,
                    enterprise.args.resolution,
                    enterprise.args.canorm,
                    enterprise.args.cluster_celltype,
            )
            helperfuncs.sort_files(f'{enterprise.args.output_dir}/analysis/cluster', 'prefix', ['res.txt', 'done.txt'])
            print(f"[finish] {enterprise.args.step}")

        if enterprise.args.step in ['all', 'analysis_category'] and enterprise.args.annotation_file:
            additional_analysis.category_analysis.cell_category_analysis(
                    enterprise.cargo.sdata,
                    enterprise.args.output_dir,
                    enterprise.args.tmp_dir,
                    'category',
                    'raw',
                    enterprise.cargo.dim_x,
                    enterprise.cargo.dim_y,
                    enterprise.cargo.imagedim,
                    performed_stainings,
                    enterprise.cargo.celltype_annotation,
                    enterprise.args.image_type,
                    enterprise.args.resolution,
                    enterprise.args.canorm,
            )
            print(f"[finish] {enterprise.args.step}")
