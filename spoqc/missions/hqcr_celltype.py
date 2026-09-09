import pandas as pd
import numpy as np
import plotly.express as px
import spatialdata as sd
import scanpy as sc
import plotly.graph_objects as go

from scipy import ndimage
from shapely.geometry import Polygon, mapping
from rasterio.features import rasterize, MergeAlg
from rasterio.transform import from_origin
from scipy.stats import median_abs_deviation

from .. import hqr
from .. import helperfuncs
from .. import priors
from .. import missions


def celltype_artefact_analysis_for_hqcr(sdata, figure_path, cell_df, annotation_file, annotation_key):

    qc_metrics = list(cell_df.columns)
    if ( annotation_file != "" ):
        ncat = len(set(cell_df[annotation_key]))
        catnames = list(set(cell_df[annotation_key]))
        catnames.sort()
        missions.plots_hqcr.generate_hqcr_html(figure_path, cell_df, annotation_key, ncat, catnames, qc_metrics)
        celltypes = cell_df[annotation_key].unique()
        helperfuncs.cell_artefact_assignment(cell_df, sdata)

        figures = []
        min_num_cells = 100 # minimum number of cells needed for multiplet and nucleus free cell distribution to estiamte thresh
        threshold_left_dict = {}
        threshold_right_dict = {}

        total_artefact_scores = np.zeros(len(celltypes))

        for qc_metric in qc_metrics:
            if qc_metric in ["transcript_counts", "n_genes_by_counts", "canorm_transcript_counts",
                             "canorm_n_genes_by_counts", "num_low_qc_transcript"]:

                threshold_log_file = open(f'{figure_path}/threshold_log.txt', 'w')

                thresholds_left = []
                thresholds_right = []
                artefact_scores = []
                nmads = 1  # Parameter for MAD calculation

                for celltype in celltypes:

                    cell_df_check = cell_df[cell_df[annotation_key] == celltype].copy()
                    
                    # Distributions
                    celltype_whole_distribution = cell_df_check[qc_metric]
                    celltype_doublet_distribution = cell_df_check[cell_df_check['artefact'] == 'doublet'][qc_metric]
                    celltype_nucleusfree_distribution = cell_df_check[cell_df_check['artefact'] == 'nucleusfree'][qc_metric]
                    celltype_cell_distribution = cell_df_check[cell_df_check['artefact'] == 'cell'][qc_metric]
                    
                    bins = 100

                    # Compute artefact score
                    a, _ = np.histogram(celltype_cell_distribution, bins=bins, density=True)
                    a += 1e-10 # Avoid division by zero
                    b, _ = np.histogram(celltype_doublet_distribution, bins=bins, density=True)
                    b += 1e-10 # Avoid division by zero
                    c, _ = np.histogram(celltype_nucleusfree_distribution, bins=bins, density=True)
                    c += 1e-10 # Avoid division by zero

                    # Take abs of KL. I am not interested which kind of skewe I have in b or c vs a.
                    if ( len(celltype_cell_distribution) > 0 ):

                        if ( len(celltype_doublet_distribution) > min_num_cells and len(celltype_nucleusfree_distribution) > min_num_cells ):
                            threshold_log_file.write(f"Left and right theshold adjustment since {celltype} had {min_num_cells} for both doublet and nucleus free cells. \n")
                            artefact_scores.append( abs(helperfuncs.KL(a, b)) + abs(helperfuncs.KL(a, c)) )
                            thresholds_right.append( np.median(celltype_doublet_distribution) - nmads * median_abs_deviation(celltype_doublet_distribution) )
                            thresholds_left.append( np.median(celltype_nucleusfree_distribution) + nmads * median_abs_deviation(celltype_nucleusfree_distribution) )
                        elif ( len(celltype_doublet_distribution) > min_num_cells and len(celltype_nucleusfree_distribution) == 0 ):
                            threshold_log_file.write(f"Only right theshold adjustment since {celltype} had not enough {min_num_cells} nucleus free cells. \n")
                            artefact_scores.append( abs(helperfuncs.KL(a, b)) )
                            thresholds_right.append( np.median(celltype_doublet_distribution) - nmads * median_abs_deviation(celltype_doublet_distribution) )
                            thresholds_left.append( np.median(celltype_cell_distribution) - nmads * median_abs_deviation(celltype_cell_distribution) )
                        elif ( len(celltype_doublet_distribution) == 0 and len(celltype_nucleusfree_distribution) > min_num_cells ):
                            threshold_log_file.write(f"Only left theshold adjustment since {celltype} had not enough {min_num_cells} doublet cells. \n")
                            artefact_scores.append( abs(helperfuncs.KL(a, c)) )
                            thresholds_right.append( np.median(celltype_cell_distribution) + nmads * median_abs_deviation(celltype_cell_distribution) )
                            thresholds_left.append( np.median(celltype_nucleusfree_distribution) + nmads * median_abs_deviation(celltype_nucleusfree_distribution) )
                        else:
                            threshold_log_file.write(f"No theshold adjustment since {celltype} had not enough {min_num_cells} doublet and nucleus free cells. \n")
                            artefact_scores.append( 0.0 )
                            thresholds_right.append( np.median(celltype_cell_distribution) + nmads * median_abs_deviation(celltype_cell_distribution) )
                            thresholds_left.append( np.median(celltype_cell_distribution) - nmads * median_abs_deviation(celltype_cell_distribution) )
                    else:
                        artefact_scores.append( 0.0 )
                        thresholds_right.append( 0.0 )
                        thresholds_left.append( 0.0 )

                total_artefact_scores += np.array(artefact_scores)

                threshold_left_dict[qc_metric] = thresholds_left
                threshold_left_dict['celltypes'] = celltypes
                threshold_right_dict[qc_metric] = thresholds_right
                threshold_right_dict['celltypes'] = celltypes

                fig = px.violin(
                    cell_df,
                    x=qc_metric,
                    y=annotation_key,
                    color='artefact',
                    box=False,
                    title=f'Multiplet and nucleus free cell disbtributions for {qc_metric}',
                    color_discrete_map={
                        'doublet': 'red',
                        'nucleus_free': 'orange',
                        'cell': 'blue'
                    }
                )
                fig.add_trace(
                    go.Scatter(
                        y=celltypes,
                        x=thresholds_left,
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='line-ns', line=dict(width=2, color='red')),
                        name='Left Threshold'
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        y=celltypes,
                        x=thresholds_right,
                        mode='markers',
                        marker=dict(color='blue', size=10, symbol='line-ns', line=dict(width=2, color='blue')),
                        name='Right Threshold'
                    )
                )
                fig.update_layout(width=800, height=2500, violinmode='overlay')
                figures.append(fig)
                fig.write_image(f"{figure_path}/split_violinplot_{qc_metric}.png", scale=3)
                fig.write_image(f"{figure_path}/split_violinplot_{qc_metric}.pdf", scale=3)

                # Bar plot of artefact scores
                df_artefact_scores = pd.DataFrame({'celltype': celltypes, 'artefact_scores': artefact_scores })
                fig_bar = px.bar(
                    df_artefact_scores,
                    x='artefact_scores',
                    y='celltype',
                    orientation='h',
                    title=f'Artefact Scores per Celltype for {qc_metric}'
                )
                figures.append(fig_bar)
                fig_bar.write_image(f"{figure_path}/barplot_artefact_scores_{qc_metric}.png", scale=3)

            elif qc_metric in ['convexity_metric_cell', 'convexity_min_nuceli', 'border_scores',
                            'thinness_score', 'island_score', 'cell_overlap_area',
                            'convexhull_outside_trnascripts', 'convexhull_all_trnascripts']:
                fig = px.violin(
                    cell_df,
                    x=qc_metric,
                    y=annotation_key,
                    color='artefact',
                    box=False,
                    title=f'Multiplet and nucleus free cell disbtributions for {qc_metric}'
                )
                fig.update_layout(width=800, height=2500, violinmode='overlay')
                figures.append(fig)
                fig.write_image(f"{figure_path}/split_violinplot_{qc_metric}.png", scale=3)
                fig.write_image(f"{figure_path}/split_violinplot_{qc_metric}.pdf", scale=3)

            else:
                print(f"[NOTE] {qc_metric} is not implemented yet for doublet and nucelus free cell check.")

        # Bar plot of artefact scores
        df_artefact_scores = pd.DataFrame({'celltype': celltypes, 'artefact_scores': total_artefact_scores })
        fig_bar = px.bar(
            df_artefact_scores,
            x='artefact_scores',
            y='celltype',
            orientation='h',
            title=f'Artefact Scores per Celltype for all considered QC metrices'
        )
        figures.append(fig_bar)
        fig_bar.write_image(f"{figure_path}/barplot_total_artefact_scores.png", scale=3)
        fig_bar.write_image(f"{figure_path}/barplot_total_artefact_scores.pdf", scale=3)

        # Generate plotly HTML
        html_content = ''.join(fig.to_html(full_html=False) for fig in figures)
        with open(f"{figure_path}/celltype_qc_analysis.html", "w") as f:
            f.write(html_content)

        threshold_log_file.close()

    return threshold_left_dict, threshold_right_dict

# For each cell calculate the bad quality probability, which is basically the poportion of 
# all the cells in a distance beloning to the bad quality cluster.
def get_bad_quality_probability(x, df, distance_matrix, bad_cluster, qc_cluster):
    quality_clusters = df.iloc[distance_matrix[x]][qc_cluster]
    number_of_bad_quality_cells = list(quality_clusters.values).count(bad_cluster)
    if ( len(quality_clusters) != 0 ):
        return(number_of_bad_quality_cells/len(quality_clusters))
    else:
        return(0.0)

def calc_celltype_transcript_counts_probs(
        sdata, 
        cell_df, 
        threshold_left_dict, 
        threshold_right_dict, 
        annotation_key,
        qc_metric,
        df_coords
):
    cell_df['qc_celltype_class'] = np.array([0] * len(cell_df))
    for i, celltype in enumerate(threshold_left_dict['celltypes']):
        df_check = cell_df[cell_df[annotation_key] == celltype]

        # Apply left threshold
        idx_qc = df_check[df_check[qc_metric] < threshold_left_dict[qc_metric][i]].index
        cell_df['qc_celltype_class'][idx_qc] = 1 # 1 for beeing bad

        # Apply right threshold
        idx_qc = df_check[df_check[qc_metric] > threshold_right_dict[qc_metric][i]].index
        cell_df['qc_celltype_class'][idx_qc] = 1 # 1 for beeing bad

    # Calculate bad quality probability
    distance_matrix = helperfuncs.points_within_radius(df_coords, 30, False)
    bad_quality_probs_celltype =  np.array([get_bad_quality_probability(
        x,
        cell_df,
        distance_matrix,
        1,
        'qc_celltype_class'
    ) for x in range(sdata['table'].n_obs)])
    good_quality_probabilities = 1 - bad_quality_probs_celltype
    
    return good_quality_probabilities, cell_df


def refine_hqcr_with_celltype_thresholds(
        sdata,
        canorm,
        figure_path,
        spoqc_tmp_folder,
        cell_df,
        threshold_left_dict,
        threshold_right_dict,
        annotation_key,
        imagedim,
        image_type,
        resolution
    ):

    df_coords = pd.DataFrame({
        'x': sdata['table'].obsm['spatial'][:,0],
        'y': sdata['table'].obsm['spatial'][:,1],
    })

    # Lets first investigate what we can do with the celltype informed threhsholds.
    # Refine HQCR based on cell type thresholds.
    # Now I have to find out which of those multiplets and emtplets are true and which are real cells still.

    qc_metric = 'transcript_counts'
    if canorm:
        qc_metric = 'canorm_transcript_counts'

    # Prior calculation done here.
    # TODO this is experimental and will change over time.
    good_quality_probs_celltype, cell_df = calc_celltype_transcript_counts_probs(
        sdata, 
        cell_df, 
        threshold_left_dict, 
        threshold_right_dict, 
        annotation_key,
        qc_metric,
        df_coords
    )
    sdata['table'].obs['good_quality_probs_celltype'] = good_quality_probs_celltype
    sdata['table'].obs['bad_quality_probs_celltype'] = 1 - good_quality_probs_celltype

    # Refine good_quality_probs_celltype assignment per cell based on the bad quality probability.
    missions.hqcr.cell_quality_probability_refinement(
        sdata,
        imagedim,
        image_type,
        resolution,
        figure_path,
        'good_quality_probs_celltype',
        'refine_qc_celltype_class',
        spoqc_tmp_folder,
        'celltype_refined'
    )

    # Generate plots
    helperfuncs.plot_scatter_density(
        sdata['table'], figure_path, 'refine_qc_celltype_class',
        'refine_qc_celltype_class', 'bad_quality_probs_celltype', ['red', 'lightblue'], 'Density of Bad Cell Quality Informed by Celltype'
    )

    helperfuncs.plot_scatter_density(
        sdata['table'], figure_path, 'artefact',
        'artefact', 'bad_quality_probs_celltype', ['lightblue', 'red', 'black'], 'Density of Bad Cell Quality Informed by Celltype'
    )

def start_exploration(enterprise):

    if enterprise.args.step in ['all', 'hqcr_celltype']:
        if enterprise.args.annotation_file:
    
            figure_path = f'{enterprise.args.output_dir}/hqcr/hqcr_celltype/'

            # Load data
            helperfuncs.read_sdata_parquet_tmp_files(enterprise.cargo.sdata, enterprise.args.tmp_dir, 'hqcr')

            # I have to reload the cell_clustering_df
            enterprise.hqcr_set.load_cell_clustering_df(enterprise)

            # Add extra columns
            annotation_key = enterprise.cargo.celltype_annotation.annotation_key
            enterprise.hqcr_set.cell_clustering_df[annotation_key] = enterprise.cargo.sdata['table'].obs[annotation_key]
            enterprise.hqcr_set.cell_clustering_df['cell_area'] = enterprise.cargo.sdata['table'].obs['cell_area']
            enterprise.hqcr_set.cell_clustering_df['nulleus_area'] = enterprise.cargo.sdata['table'].obs['nucleus_area']
            enterprise.hqcr_set.cell_clustering_df['nucleus_free'] = enterprise.cargo.sdata['table'].obs['wnucleus_free']
            enterprise.hqcr_set.cell_clustering_df['doublet'] = enterprise.cargo.sdata['table'].obs['wdoublet']

            threshold_left_dict, threshold_right_dict = celltype_artefact_analysis_for_hqcr(
                enterprise.cargo.sdata,
                figure_path,
                enterprise.hqcr_set.cell_clustering_df,
                enterprise.args.annotation_file,
                enterprise.cargo.celltype_annotation.annotation_key,
            )

            refine_hqcr_with_celltype_thresholds(
                enterprise.cargo.sdata,
                enterprise.args.canorm,
                figure_path,
                enterprise.args.tmp_dir,
                enterprise.hqcr_set.cell_clustering_df,
                threshold_left_dict, 
                threshold_right_dict,
                enterprise.cargo.celltype_annotation.annotation_key,
                enterprise.cargo.imagedim,
                enterprise.args.image_type,
                enterprise.args.resolution,
            )

            print("[finish]")
        else:
            print("[NOTE] No annotation file provided so I will not perform start_hqcr_celltype")