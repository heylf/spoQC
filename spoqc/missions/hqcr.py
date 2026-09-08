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

from .. import core
from .. import hqr
from .. import helperfuncs
from .. import priors
from .. import missions


def cell_artefact_assignment(cell_df, sdata):
    cell_df['artefact'] = 'cell'

    # Assign artefacts
    cell_df.loc[cell_df['doublet'] == 1, 'artefact'] = 'doublet'
    mean_overlap = cell_df['cell_overlap_area'].mean()

    cell_df.loc[(cell_df['nuceli_count'] > 1) & (cell_df['cell_overlap_area'] > mean_overlap), 'artefact'] = 'doublet'
    cell_df.loc[cell_df['nucleus_free'] == 1, 'artefact'] = 'nucleus_free'
    sdata['table'].obs['artefact'] = cell_df['artefact']


def create_polygon_dataframe(sdata, imagedim, object, prob_col=None):

    # Get all polygon coordinates in the real coordinate system.
    polys = sd.transform(sdata[object], to_coordinate_system='global')

    if ( prob_col != None ):
        polys[prob_col] = list(sdata['table'].obs[prob_col])

    # The image matrix has not the same index range as the poly coords, so we need to
    # offset coordinates into the matrix's index space. x_idx/y_idx used to be materialized
    # as Python lists searched with list.index() (O(image_width)/O(image_height) per vertex);
    # since they are contiguous ranges starting at bb_xmin-1/bb_ymin-1, the index is just an
    # O(1) arithmetic offset.
    x_offset = int(imagedim.bb_xmin - 1)
    y_offset = int(imagedim.bb_ymin - 1)

    # Translate poly coords.
    for index, row in polys.iterrows():
        poly = row['geometry']
        translated_poly = []
        for tuple in list(poly.exterior.coords):
            x = int(tuple[0])
            y = int(tuple[1])

            if ( x >= int(imagedim.bb_xmin) and x <= int(imagedim.bb_xmax) -1 and \
                 y >= int(imagedim.bb_ymin) and y <= int(imagedim.bb_ymax) - 1 ):
                translated_poly.append((x - x_offset, y - y_offset))
                
        translated_poly = Polygon(translated_poly)

        # Problem is that I translate float coorinates into integer cooridnate for the image.
        # This might generate polygons that are invalid which I have to correct.
        if( not translated_poly.is_valid ):
            translated_poly = translated_poly.convex_hull

        polys.loc[index, 'geometry'] = translated_poly

    return polys


def _create_cell_probability_image(sdata, polys, img, resolution, prob_col):

    dim_x = len(sdata[img][resolution].image.y.values)
    dim_y = len(sdata[img][resolution].image.x.values)

    # Define output matrix size
    height, width = int(dim_x), int(dim_y)

    # Define transform: (origin_x, origin_y, pixel_width, pixel_height)
    transform = from_origin(0, height, 1, 1)  # top-left at (0, height), cell size = 1

    # Define your list of (polygon, value) tuples
    polygons_with_values = [ (row['geometry'], row[prob_col]) for index, row in polys.iterrows() ]

    # Get for each pixel the summed cell quality probabilties from each cell polygon.
    shapes = ((mapping(poly), val) for poly, val in polygons_with_values)

    # Switching to all_touched=False avoids boundary multi-writes that might inflate overlaps and thus inflate counts.
    value_prob_sum = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype='float64',
        merge_alg=MergeAlg.add
    )

    # Get for each pixel the overlap count of each cell polygon.
    shapes = ((mapping(poly), 1) for poly, val in polygons_with_values)

    # Switching to all_touched=False avoids boundary multi-writes that might inflate overlaps and thus inflate counts.
    value_count = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype='int32',
        merge_alg=MergeAlg.add
    )

    # Calcualte the average cell quality probability for each pixel.
    # This create the cell quality probability image.
    average_cell_probability_image = np.zeros((dim_x, dim_y))
    with np.errstate(divide='ignore', invalid='ignore'):
        average_cell_probability_image = np.where(value_count > 0, value_prob_sum / value_count, 0)

    return average_cell_probability_image


def map_values_to_cells(
        sdata,
        polys,
        img,
        resolution,
        labels,
        res_col,
        figure_path,
        mode,
        *,
        tresh_polgon_score=15,
        true_false_binary=False
    ):
    
    dim_x = len(sdata[img][resolution].image.y.values)
    dim_y = len(sdata[img][resolution].image.x.values)

    # Define output matrix size
    height, width = int(dim_x), int(dim_y)

    # Define transform: (origin_x, origin_y, pixel_width, pixel_height)
    transform = from_origin(0, height, 1, 1)  # top-left at (0, height), cell size = 1

    polygons_with_values = zip(polys['geometry'], polys.index)

    shapes = [(mapping(geom), value) for geom, value in polygons_with_values]

    # Switching to all_touched=False avoids boundary multi-writes that might inflate overlaps and thus inflate counts.
    index_map = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=False
    )

    # Flatten the arrays
    flat_index = index_map.ravel()
    flat_labels = labels.ravel()

    # Extract only valid pixels (index_map > 0 or > -1 depending on background)
    valid = flat_index >= 0  # change to >0 if background is 0

    flat_index = flat_index[valid]
    flat_labels = flat_labels[valid]

    # Get unique polygon indices from your GeoDataFrame
    polygon_ids = polys.index.to_numpy()

    if ( mode == 'markov_labels' ):

        # Compute per-polygon sum of label values using ndimage.sum
        polygon_scores = ndimage.sum(
            input=flat_labels,
            labels=flat_index,
            index=polygon_ids
        )

        # Apply threshold
        refine_qc_celltype_class = (polygon_scores > tresh_polgon_score).astype(int).tolist()

        if ( true_false_binary ):
            refine_qc_celltype_class = ["True" if str(x) == "1" else "False" for x in refine_qc_celltype_class]

        # I have to do it that way because there can be less nuceli than cells.
        if ( len(refine_qc_celltype_class) < sdata['table'].n_obs ):
            sdata['table'].obs[res_col] = ["False"] * sdata['table'].n_obs
            polygon_ids = [str(x) for x in polygon_ids]
            sdata['table'].obs.loc[polygon_ids, res_col] = refine_qc_celltype_class
        else:
            sdata['table'].obs[res_col] = refine_qc_celltype_class

        # Plot
        colors = ['lightblue', 'red']
        if ( len(list(set(sdata['table'].obs[res_col]))) < 2 ):
            colors = ['red']
        helperfuncs.plot_scatter(sdata['table'], figure_path, res_col, None, res_col, colors, None)

    if ( mode == 'mean_values' ):

        # Compute per-polygon mean of values
        polygon_scores = ndimage.mean(
            input=flat_labels,
            labels=flat_index,
            index=polygon_ids
        )

        if ( len(polygon_scores) < sdata['table'].n_obs ):
            sdata['table'].obs[res_col] = [0] * sdata['table'].n_obs
            polygon_ids = [str(x) for x in polygon_ids]
            sdata['table'].obs.loc[polygon_ids, res_col] = polygon_scores
        else:
            sdata['table'].obs[res_col] = polygon_scores


    if ( mode == 'mean_values_nonzero' ):

        # Exclude zero-valued pixels before computing mean
        nonzero_mask = flat_labels != 0
        flat_labels_nz = flat_labels[nonzero_mask]
        flat_index_nz = flat_index[nonzero_mask]

        polygon_scores = ndimage.mean(
            input=flat_labels_nz,
            labels=flat_index_nz,
            index=polygon_ids
        )

        # Polygons with no non-zero pixels produce NaN — treat as 0
        polygon_scores = np.nan_to_num(polygon_scores, nan=0.0)

        if ( len(polygon_scores) < sdata['table'].n_obs ):
            sdata['table'].obs[res_col] = [0] * sdata['table'].n_obs
            polygon_ids = [str(x) for x in polygon_ids]
            sdata['table'].obs.loc[polygon_ids, res_col] = polygon_scores
        else:
            sdata['table'].obs[res_col] = polygon_scores

    # This mode is for hqtr and hqpr because a lot of pixels are not informative.
    # This you can see in the distribution of the pixel beliefs for hqtr and hqpr.
    # I think the issue is currently the PS score.
    if ( mode == 'mean_values_informative' ):

        # Use pixels with a bliefs of > 0.2 before computing mean
        non_t_mask = flat_labels > 0.2
        flat_labels_nz = flat_labels[non_t_mask]
        flat_index_nz = flat_index[non_t_mask]

        polygon_scores = ndimage.mean(
            input=flat_labels_nz,
            labels=flat_index_nz,
            index=polygon_ids
        )

        # Polygons with no non-zero pixels produce NaN — treat as 0
        polygon_scores = np.nan_to_num(polygon_scores, nan=0.0)

        if ( len(polygon_scores) < sdata['table'].n_obs ):
            sdata['table'].obs[res_col] = [0] * sdata['table'].n_obs
            polygon_ids = [str(x) for x in polygon_ids]
            sdata['table'].obs.loc[polygon_ids, res_col] = polygon_scores
        else:
            sdata['table'].obs[res_col] = polygon_scores


def _cell_quality_probability_refinement(sdata, imagedim, image_type, resolution, figure_path, 
                                        prob_col, res_col, spoqc_tmp_folder, suffix):
    
    polys = create_polygon_dataframe(sdata, imagedim, 'cell_boundaries', prob_col)
    average_cell_probability_image = _create_cell_probability_image(sdata, polys, image_type, resolution, prob_col)

    # This is a sanity check
    has_values_over_1 = np.any(np.array(polys[prob_col]) > 1)
    print(f'Are there any values bigger than 1: {has_values_over_1}')

    has_values_over_1 = np.any(average_cell_probability_image > 1)
    print(f'Are there any values bigger than 1: {has_values_over_1}')

    # Plot input prior image
    helperfuncs.plot_pixels(
        figure_path,
        (average_cell_probability_image > 0).astype(np.uint8),
        imagedim,
        'input_priors',
        'input_priors',
        'gray',
        False,
        True,
        legend_dict={"mask": "#FFFFFF", "empty": "#000000"}
    )
        
    beliefs, labels = hqr.markov_random_field_zarr_parallel.first_version_loopy_belief_propagation_parallel(
        average_cell_probability_image,
        spoqc_tmp_folder,
        'hqcr',
        beta=1.5,
        max_iter=15,
        normalize='total'
    )

    hqr.markov_random_field_zarr_parallel.visualize_markov_calculation(average_cell_probability_image, labels[:], figure_path)
    map_values_to_cells(sdata, polys, image_type, resolution, labels[:], res_col, figure_path, 'markov_labels')

    # Write out hqcr mask
    df_smoothed = pd.DataFrame({
        'hqcr_beliefs_smoothed': beliefs[:].flatten(),
        'hqcr_mask_smoothed': labels[:].flatten(),
    })
    df_smoothed.to_parquet(f"{spoqc_tmp_folder}/hqcr_output_mask_smoothed_{suffix}.parquet")

    df = pd.DataFrame({
        'hqcr_beliefs': average_cell_probability_image.flatten(),
        'hqcr_mask': (average_cell_probability_image.flatten() > 0.5).astype(np.uint8),
    })
    df.to_parquet(f"{spoqc_tmp_folder}/hqcr_output_mask_{suffix}.parquet")

    # This is in cell dimension.
    if 'hqcr_traffic_light' in sdata['table'].obs.columns :
        df = pd.DataFrame({
            'hqcr_traffic_light': sdata['table'].obs['hqcr_traffic_light'],
        })
        df.index = sdata['table'].obs.index
        df.to_parquet(f"{spoqc_tmp_folder}/traffic_light_output_hqcr.parquet")



def start_hqcr(enterprise, test_res=False):

    if enterprise.args.step in ['all', 'unittest', 'hqcr_ident']:

        figure_path = f'{enterprise.args.output_dir}/hqcr/hqcr_ident/'

        # Generate first the input cell and nucleus segmentation figures
        for seg in ['cell_labels', 'nucleus_labels']:
            values = enterprise.cargo.sdata.labels[seg][enterprise.args.resolution].image.values
            values = (values > 0.0).astype(np.uint8)
            helperfuncs.plot_pixels(
                figure_path,
                values,
                enterprise.cargo.imagedim,
                f'input_segmentation_{seg}',
                f'input_segmentation_{seg}',
                'gray',
                False,
                True,
                legend_dict={"mask": "#FFFFFF", "empty": "#000000"},
                flip=True
            )

        print("[NOTE] Gather cell QC metrices")
        helperfuncs.read_sdata_parquet_tmp_files(enterprise.cargo.sdata, enterprise.args.tmp_dir, 'hqcr')

        enterprise.initialize_hqcr_set()

        # Leiden clustering.
        print("[NOTE] Cell QC clustering")
        sc.pp.neighbors(enterprise.hqcr_set.cell_clustering_adata, n_neighbors=20, random_state=enterprise.args.seed)
        sc.tl.umap(enterprise.hqcr_set.cell_clustering_adata, random_state=enterprise.args.seed)
        if test_res: # Use this to optimize the step.
            helperfuncs.test_resolutions_leiden(
                enterprise.hqcr_set.cell_clustering_adata,
                figure_path,
                enterprise.args.nthreads,
                k=10
            )
        sc.tl.leiden(enterprise.hqcr_set.cell_clustering_adata, resolution=1.2)

        # Here we combine available.
        missions.combine_priors.combine_priors_hqcr(enterprise, figure_path)

        # Cell quality probability refinement
        _cell_quality_probability_refinement(
            enterprise.cargo.sdata,
            enterprise.cargo.imagedim,
            enterprise.args.image_type,
            enterprise.args.resolution,
            figure_path,
            'good_quality_probabilities',
            'refined_qc_class',
            enterprise.args.tmp_dir,
            'raw'
        )

        ###################
        ###### Plots ######
        ###################
        print("[NOTE] Generate plots for HQCRs")
        missions.plots_hqcr.plot_hqcr(enterprise.cargo.sdata, figure_path, 50, 20)

        # HTML report for qc metrices
        enterprise.hqcr_set.cell_clustering_df['cell_area'] = enterprise.cargo.sdata['table'].obs['cell_area']
        enterprise.hqcr_set.cell_clustering_df['cell_region'] = enterprise.cargo.sdata['table'].obs['cell_region']

        # Add for a better visualiation all of the data again as an all data cluster
        cell_df_all = enterprise.hqcr_set.cell_clustering_df.copy()
        cell_df_all['qc_cluster_str'] = ['all'] * len(cell_df_all)
        cell_df_combined_with_all = pd.concat([enterprise.hqcr_set.cell_clustering_df, cell_df_all])

        print("[NOTE] Generate plots for HTMLs")
        # Generate html for qc metrices
        ncat = len(set(cell_df_combined_with_all['qc_cluster_str']))
        catnames = list(set(cell_df_combined_with_all['qc_cluster_str']))
        catnames.sort()
        missions.plots_hqcr.generate_hqcr_html(
            figure_path,
            cell_df_combined_with_all,
            'qc_cluster_str',
            ncat,
            catnames,
            enterprise.hqcr_set.metrics,
        )

        # Generate html for quality cell regions (bad, small and hqcrs)
        ncat = len(set(enterprise.hqcr_set.cell_clustering_df['cell_region']))
        catnames = list(set(enterprise.hqcr_set.cell_clustering_df['cell_region']))
        catnames.sort()
        missions.plots_hqcr.generate_hqcr_html(
            figure_path,
            enterprise.hqcr_set.cell_clustering_df,
            'cell_region',
            ncat,
            catnames,
            enterprise.hqcr_set.metrics,
        )

        # Generate html for just all the data
        cell_df_all['data'] = list(cell_df_all['qc_cluster_str'])
        ncat = len(set(cell_df_all['data']))
        catnames = list(set(cell_df_all['data']))
        catnames.sort()
        missions.plots_hqcr.generate_hqcr_html(figure_path, cell_df_all, 'data', ncat, catnames, enterprise.hqcr_set.metrics)

        print("[finish]")


def load_data_for_hqcr_celltype(sdata, spoqc_tmp_folder, counts, annotation_key):

    # Load data
    helperfuncs.read_sdata_parquet_tmp_files(sdata, spoqc_tmp_folder, 'hqcr')

    cell_df = helperfuncs.load_cell_df(counts, sdata)
    cell_df[annotation_key] = sdata['table'].obs[annotation_key]
    cell_df['cell_area'] = sdata['table'].obs['cell_area']
    cell_df['nulleus_area'] = sdata['table'].obs['nucleus_area']
    cell_df['nucleus_free'] = sdata['table'].obs['wnucleus_free']

    df_coords = pd.DataFrame({
        'x': sdata['table'].obsm['spatial'][:,0],
        'y': sdata['table'].obsm['spatial'][:,1],
    })

    return cell_df, df_coords


def celltype_artefact_analysis_for_hqcr(sdata, figure_path, cell_df, annotation_file, annotation_key, counts):

    qc_metrics = list(cell_df.columns)
    if ( annotation_file != "" ):
        ncat = len(set(cell_df[annotation_key]))
        catnames = list(set(cell_df[annotation_key]))
        catnames.sort()
        missions.plots_hqcr.generate_hqcr_html(figure_path, cell_df, annotation_key, ncat, catnames, qc_metrics)
        celltypes = cell_df[annotation_key].unique()
        cell_artefact_assignment(cell_df, sdata)

        figures = []
        min_num_cells = 100 # minimum number of cells needed for multiplet and nucleus free cell distribution to estiamte thresh
        threshold_left_dict = {}
        threshold_right_dict = {}

        total_artefact_scores = np.zeros(len(celltypes))

        for qc_metric in qc_metrics:
            if qc_metric in [counts, 'n_genes_by_counts', 'num_low_qc_transcript']:

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


def refine_hqcr_with_celltype_thresholds(
        sdata,
        figure_path,
        spoqc_tmp_folder,
        cell_df,
        df_coords,
        counts,
        threshold_left_dict,
        threshold_right_dict,
        annotation_key,
        imagedim,
        image_type,
        resolution
    ):

    # Lets first investigate what we can do with the celltype informed threhsholds.
    # Refine HQCR based on cell type thresholds.
    # Now I have to find out which of those multiplets and emtplets are true and which are real cells still.
    qc_metric = counts
    good_quality_probs_celltype, cell_df = priors.hqcr.transcript_counts_celltype.calc_celltype_transcript_counts_probs(
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
    _cell_quality_probability_refinement(
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

def start_hqcr_celltype(enterprise):

    if enterprise.args.step in ['all', 'hqcr_celltype']:
        if enterprise.args.annotation_file:
    
            figure_path = f'{enterprise.args.output_dir}/hqcr/hqcr_celltype/'

            counts = 'transcript_counts'
            if enterprise.args.canorm:
                counts = 'canorm_transcript_counts'

            cell_df, df_coords = load_data_for_hqcr_celltype(
                enterprise.cargo.sdata,
                enterprise.args.tmp_dir,
                counts,
                enterprise.cargo.celltype_annotation.annotation_key
            )

            threshold_left_dict, threshold_right_dict = celltype_artefact_analysis_for_hqcr(
                enterprise.cargo.sdata,
                figure_path,
                cell_df,
                enterprise.args.annotation_file,
                enterprise.cargo.celltype_annotation.annotation_key,
                counts
            )

            refine_hqcr_with_celltype_thresholds(
                enterprise.cargo.sdata,
                figure_path,
                enterprise.args.tmp_dir,
                cell_df,
                df_coords,
                counts,
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