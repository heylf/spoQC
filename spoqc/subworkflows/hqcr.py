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

# Function to print all HQCRs
def plot_hqcr(sdata, figure_path, min_number_good_cells_hqcr, minimum_number_of_total_cells):
    islands = list(set(sdata['table'].obs['island_index']))
    sdata['table'].obs['cell_region'] = np.array(['undefined'] * len(sdata['table']))

    for island in islands:
        island_idxs = sdata['table'].obs['island_index'] == island
        island_adata = sdata['table'][island_idxs]

        num_good_qc_cells = list(island_adata.obs['refined_qc_class']).count(1) # good
        num_bad_qc_cells = list(island_adata.obs['refined_qc_class']).count(0) # bad

        good_bad_ration = 0.0
        if ( num_bad_qc_cells == 0 ):
            good_bad_ration = 100
        else:
            good_bad_ration = num_good_qc_cells / num_bad_qc_cells

        title = ''
        island_suffix = ''
        if ( good_bad_ration > 1.0 ):
            if ( num_good_qc_cells > min_number_good_cells_hqcr ):
                title = f'HQCR {island+1}'
                island_suffix = 'hqcr'
                sdata['table'].obs.loc[island_idxs, 'cell_region'] = 'hqcr'
            else:
                title = f'Small HQCR {island+1}'
                island_suffix = 'small_hqcr'
                sdata['table'].obs.loc[island_idxs, 'cell_region'] = 'small_hqcr'
        else:
            title = f'LQCR {island+1}'
            island_suffix = 'lqcr'
            sdata['table'].obs.loc[island_idxs, 'cell_region'] = 'lqcr'

        # Here I just setelect all the cells that are part of the island to mark them later in the plot.
        island_select = np.array([0] * sdata['table'].n_obs)
        island_select[island_idxs] = 1
        sdata['table'].obs['island_select'] = island_select

        if ( num_good_qc_cells + num_bad_qc_cells > minimum_number_of_total_cells ):
            helperfuncs.plot_scatter(
                island_adata,
                f'{figure_path}/{island_suffix}/',
                f'zoomed_{island_suffix}_{island+1}',
                None,
                None,
                None,
                title
            )
            helperfuncs.plot_scatter(
                sdata['table'],
                f'{figure_path}/{island_suffix}/',
                f'{island_suffix}_{island+1}',
                None,
                'island_select',
                ['lightblue', 'red'],
                title
            )

    hqcr_df = pd.DataFrame({
        'islands': sdata['table'].obs['island_index'], 
        'cell_region': sdata['table'].obs['cell_region']
    })

    hqcr_df.to_json(f"{figure_path}/hqcr.json", orient="columns")


def generate_hqcr_html(figure_path, df_plot, cat, ncat, catnames, qc_metrics):

    figures = []

    qc_metrics = [x for x in qc_metrics if x not in ['celltype']]

    for level in qc_metrics:
        print(level)
        plotname = 'violinplot'
        title = 'Distribution of'
        if ( level == 'doublet' or level == 'nucleus_free' ):
            num_doublets_qc_cluster = [-1] * ncat
            for c in range(0, ncat):
                ndoublets = len([True for x in df_plot[df_plot[cat] == catnames[c]][level] if x == 1])
                num_doublets_qc_cluster[c] = ndoublets

            plot_doublet_df = pd.DataFrame({
                cat: [str(x) for x in catnames],
                level: num_doublets_qc_cluster
            })
            fig = px.bar(
                plot_doublet_df,
                x=level,
                y=cat,
                width=800,
                height=800
            )
            plotname = 'barplot'
            title = 'Count of'
        else:
            if ( level == 'island_score' ):
                fig = px.violin(
                    x=helperfuncs.min_max_normalize(df_plot['island_score']), 
                    y=df_plot[cat],
                    width=800,
                    height=800
                )
                fig.update_layout(
                    xaxis=dict(range=[0, 1.1], title='min-max normalized island score'),
                    yaxis=dict(title=cat)
                )
            else:
                fig = px.violin(
                    df_plot,
                    x=level,
                    y=cat,
                    width=800,
                    height=800
                )

        fig.update_layout(title=f"{title} {level} for all {cat}", showlegend=True)
        helperfuncs.apply_general_plotly_layout(fig, True)

        figures.append(fig)
        fig.write_image(f"{figure_path}/{plotname}_{level}.png", scale=3)
        fig.write_image(f"{figure_path}/{plotname}_{level}.pdf", scale=3)

    with open(f'{figure_path}/hqcr_{cat}.html', 'w') as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


def load_cell_df(counts, sdata):

    cell_df = pd.DataFrame({
        counts: sdata['table'].obs[counts],
        'control_probe_counts': sdata['table'].obs['control_probe_counts'],
        'n_genes_by_counts': sdata['table'].obs['canorm_n_genes_by_counts'],
        'convexity_metric_cell': sdata['table'].obs['convexity_metric_cell'],
        'convexity_min_nuceli': sdata['table'].obs['convexity_min_nuceli'],
        'nuceli_count': sdata['table'].obs['nuceli_count'],
        'border_scores': sdata['table'].obs['border_scores'],
        'thinness_score': sdata['table'].obs['thinness_score'],
        'island_score': sdata['table'].obs['island_score'],
        'doublet': sdata['table'].obs['wdoublet'],
        'cell_overlap_area': sdata['table'].obs['cell_overlap_area'],
        'convexhull_outside_trnascripts': sdata['table'].obs['convexhull_outside_trnascripts'],
        #'convexhull_all_trnascripts': sdata['table'].obs['convexhull_all_trnascripts'],
        'num_low_qc_transcript': sdata['table'].obs['num_low_qc_transcript']
    })

    return cell_df


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
        polys['good_quality_probabilities'] = list(1 - sdata['table'].obs[prob_col])

    # These list I need later because the image matrix has not the same index range as the poly coords.
    x_idx = [i for i in range(int(imagedim.bb_xmin-1), int(imagedim.bb_xmax+1))]
    y_idx = [i for i in range(int(imagedim.bb_ymin-1), int(imagedim.bb_ymax+1))]

    # Translate poly coords.
    for index, row in polys.iterrows():
        poly = row['geometry']
        translated_poly = []
        for tuple in list(poly.exterior.coords):
            x = int(tuple[0])
            y = int(tuple[1])

            if ( x >= int(imagedim.bb_xmin) and x <= int(imagedim.bb_xmax) -1 and \
                 y >= int(imagedim.bb_ymin) and y <= int(imagedim.bb_ymax) - 1 ):
                translated_poly.append((x_idx.index(x),y_idx.index(y)))
                
        translated_poly = Polygon(translated_poly)

        # Problem is that I translate float coorinates into integer cooridnate for the image.
        # This might generate polygons that are invalid which I have to correct.
        if( not translated_poly.is_valid ):
            translated_poly = translated_poly.convex_hull

        polys.loc[index, 'geometry'] = translated_poly

    return polys


def create_cell_probability_image(sdata, polys, img, resolution):

    dim_x = len(sdata[img][resolution].image.y.values)
    dim_y = len(sdata[img][resolution].image.x.values)

    # Define output matrix size
    height, width = int(dim_x), int(dim_y)

    # Define transform: (origin_x, origin_y, pixel_width, pixel_height)
    transform = from_origin(0, height, 1, 1)  # top-left at (0, height), cell size = 1

    # Define your list of (polygon, value) tuples
    polygons_with_values = [ (row['geometry'], row['good_quality_probabilities']) for index, row in polys.iterrows() ]

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

    print(height)
    print(width)

    # Define transform: (origin_x, origin_y, pixel_width, pixel_height)
    transform = from_origin(0, height, 1, 1)  # top-left at (0, height), cell size = 1

    polygons_with_values = [ (row['geometry'], index) for index, row in polys.iterrows() ]

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

    print(len(flat_index))
    print(len(flat_labels))

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


def cell_quality_probability_refinement(sdata, imagedim, image_type, resolution, figure_path, 
                                        prob_col, res_col, spoqc_tmp_folder, suffix):
    
    polys = create_polygon_dataframe(sdata, imagedim, 'cell_boundaries', prob_col)
    average_cell_probability_image = create_cell_probability_image(sdata, polys, image_type, resolution)

    # This is a sanity check
    has_values_over_1 = np.any(np.array(polys['good_quality_probabilities']) > 1)
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
        'hqcr_mask': (average_cell_probability_image.flatten() > 0.5).astype(np.uint8)
    })
    df.to_parquet(f"{spoqc_tmp_folder}/hqcr_output_mask_{suffix}.parquet")


def load_data_for_hqcr(sdata, spoqc_tmp_folder, counts):
    print("[NOTE] Gather cell QC metrices")
    helperfuncs.read_sdata_parquet_tmp_files(sdata, spoqc_tmp_folder, 'hqcr')
    qc_domains_adata = sdata['table']

    cell_df = load_cell_df(counts, sdata)

    # Check for NaNs
    print("[DEBUG] Number of NaNs in each column:")
    print(cell_df.isna().sum())

    # This I have to do to avoid an error because of the number of features I have selected.
    qc_metrices = list(cell_df.columns)
    qc_domains_adata = qc_domains_adata[:,0:len(qc_metrices)]
    qc_domains_adata.X = cell_df

    return qc_domains_adata, cell_df, qc_metrices


def clustering_for_hqcr(qc_domains_adata, figure_path, seed, test_res_n_clusters=10, test_res=False):
    # leiden clustering
    print("[NOTE] Cell QC clustering")
    sc.pp.neighbors(qc_domains_adata, n_neighbors=20, random_state=seed)
    sc.tl.umap(qc_domains_adata, random_state=seed)

    if ( test_res ):
        ss = helperfuncs.test_resolutions_leiden(qc_domains_adata, figure_path, test_res_n_clusters)

    sc.tl.leiden(qc_domains_adata, resolution=1.2)


def start_hqcr(sdata, spoqc_tmp_folder, imagedim, CONST, seed):
    figure_path = f'{CONST.FIGURE_PATH}/hqcr/hqcr_ident/'

    # Generate first the input cell and nucleus segmentation figures
    for seg in ['cell_labels', 'nucleus_labels']:
        values = sdata.labels[seg][CONST.RESOLUTION].image.values
        values = (values > 0.0).astype(np.uint8)
        helperfuncs.plot_pixels(
            figure_path,
            values,
            imagedim,
            f'input_segmentation_{seg}',
            f'input_segmentation_{seg}',
            'gray',
            False,
            True,
            legend_dict={"mask": "#FFFFFF", "empty": "#000000"},
            flip=True
        )

    counts = 'transcript_counts'
    if ( CONST.CANORM ):
        counts = 'canorm_transcript_counts'

    qc_domains_adata, cell_df, qc_metrices = load_data_for_hqcr(sdata, spoqc_tmp_folder, counts)
    clustering_for_hqcr(qc_domains_adata, figure_path, seed)
    
    bad_quality_probabilities, cell_df = priors.hqcr.transcript_counts.calc_transcript_counts_probs(
        sdata, 
        figure_path,
        cell_df,
        qc_domains_adata,
        counts
    )

    # priors.combine_priors.combine_priors_hqtr(spoqc_tmp_folder, image_ddf)

    sdata['table'].obs['bad_quality_probabilities'] = bad_quality_probabilities

    # Cell quality probability refinement
    cell_quality_probability_refinement(
        sdata,
        imagedim,
        CONST.IMAGE_TYPE,
        CONST.RESOLUTION,
        figure_path,
        'bad_quality_probabilities',
        'refined_qc_class',
        spoqc_tmp_folder,
        'raw'
    )

    ###################
    ###### Plots ######
    ###################
    print("[NOTE] Generate plots for HQCRs")
    plot_hqcr(sdata, figure_path, 50, 20)

    # HTML report for qc metrices
    cell_df['cell_area'] = sdata['table'].obs['cell_area']
    cell_df['cell_region'] = sdata['table'].obs['cell_region']

    # Add for a better visualiation all of the data again as an all data cluster
    cell_df_all = cell_df.copy()
    cell_df_all['qc_cluster_str'] = ['all'] * len(cell_df_all)
    cell_df_combined_with_all = pd.concat([cell_df, cell_df_all])

    print("[NOTE] Generate plots for HTMLs")
    # Generate html for qc metrices
    ncat = len(set(cell_df_combined_with_all['qc_cluster_str']))
    catnames = list(set(cell_df_combined_with_all['qc_cluster_str']))
    catnames.sort()
    generate_hqcr_html(figure_path, cell_df_combined_with_all, 'qc_cluster_str', ncat, catnames, qc_metrices)

    # Generate html for quality cell regions (bad, small and hqcrs)
    ncat = len(set(cell_df['cell_region']))
    catnames = list(set(cell_df['cell_region']))
    catnames.sort()
    generate_hqcr_html(figure_path, cell_df, 'cell_region', ncat, catnames, qc_metrices)

    # Generate html for just all the data
    cell_df_all['data'] = list(cell_df_all['qc_cluster_str'])
    ncat = len(set(cell_df_all['data']))
    catnames = list(set(cell_df_all['data']))
    catnames.sort()
    generate_hqcr_html(figure_path, cell_df_all, 'data', ncat, catnames, qc_metrices)


def load_data_for_hqcr_celltype(sdata, spoqc_tmp_folder, counts, annotation_key):

    # Load data
    helperfuncs.read_sdata_parquet_tmp_files(sdata, spoqc_tmp_folder, 'hqcr')

    cell_df = load_cell_df(counts, sdata)
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
        generate_hqcr_html(figure_path, cell_df, annotation_key, ncat, catnames, qc_metrics)
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
    bad_quality_probs_celltype, cell_df = priors.hqcr.transcript_counts.calc_celltype_transcript_counts_probs(
        sdata, 
        cell_df, 
        threshold_left_dict, 
        threshold_right_dict, 
        annotation_key,
        qc_metric,
        df_coords
    )
    sdata['table'].obs['bad_quality_probs_celltype'] = bad_quality_probs_celltype

    # Refine bad_quality_probs_celltype assignment per cell based on the bad quality probability.
    cell_quality_probability_refinement(
        sdata,
        imagedim,
        image_type,
        resolution,
        figure_path,
        'bad_quality_probs_celltype',
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

def start_hqcr_celltype(sdata, spoqc_tmp_folder, imagedim, CONST):

    figure_path = f'{CONST.FIGURE_PATH}/hqcr/hqcr_celltype/'

    counts = 'transcript_counts'
    if ( CONST.CANORM ):
        counts = 'canorm_transcript_counts'

    cell_df, df_coords = load_data_for_hqcr_celltype(sdata, spoqc_tmp_folder, counts, CONST.ANNOTATION_KEY)
    threshold_left_dict, threshold_right_dict = celltype_artefact_analysis_for_hqcr(
        sdata,
        figure_path,
        cell_df,
        CONST.ANNOTATION_FILE,
        CONST.ANNOTATION_KEY,
        counts
    )

    refine_hqcr_with_celltype_thresholds(
        sdata,
        figure_path,
        spoqc_tmp_folder,
        cell_df,
        df_coords,
        counts,
        threshold_left_dict, 
        threshold_right_dict,
        CONST.ANNOTATION_KEY,
        imagedim,
        CONST.IMAGE_TYPE,
        CONST.RESOLUTION
    )