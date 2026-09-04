import numpy as np
import plotly.express as px
import geopandas as gpd

from typing import Tuple, List

from ... import helperfuncs
from ... import core

def is_convex(polygon: List[Tuple[float, float]]) -> Tuple[bool, float]:
    """
    Determines if a polygon is convex and measures its convexity.
    
    Parameters:
        polygon (List[Tuple[float, float]]): List of (x, y) vertices defining the polygon.
        
    Returns:
        Tuple[bool, float]: 
            - A boolean indicating whether the polygon is convex.
            - A float representing the convexity metric (range [0, 1]; 1 is fully convex).
    """
    n = len(polygon)
    if n < 3:
        raise ValueError("A polygon must have at least three vertices.")
    
    cross_products = []
    for i in range(n):
        # Get three consecutive points
        p1 = np.array(polygon[i])
        p2 = np.array(polygon[(i + 1) % n])
        p3 = np.array(polygon[(i + 2) % n])
        
        # Compute vectors
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Compute the cross product of vectors
        cross_product = np.cross(v1, v2)
        cross_products.append(cross_product)
    
    # Check if all cross products have the same sign
    all_positive = all(cp > 0 for cp in cross_products)
    all_negative = all(cp < 0 for cp in cross_products)
    
    is_polygon_convex = all_positive or all_negative

    # Measure convexity as the ratio of consistent angles
    neg = 0
    pos = 0
    if all_positive or all_negative:
        pos = sum(cp != 0 for cp in cross_products)
    else:
        # Sum up the boolean vectore (i.e., sum up all Ture)
        neg = sum(cp < 0 for cp in cross_products)
        pos = sum(cp > 0 for cp in cross_products)

    # Just take the maximum amount of consistent angles
    convexity_metric = np.max([neg,pos]) / len(cross_products)
    
    return is_polygon_convex, convexity_metric


# Function to assign nulcei to cell
def find_overlapping_nuclei(cells: gpd.GeoDataFrame, nucleus: gpd.GeoDataFrame):
    print("[NOTE] Find overlapping nuceli for cells")
    timer = helperfuncs.Timer()
    timer.start()
    overlaps = []
    nucleus_centroids = nucleus.geometry.centroid
    for cell in cells.geometry:
        overlapping_indices = nucleus[nucleus_centroids.geometry.intersects(cell)].index.tolist()
        overlaps.append(overlapping_indices)
    timer.stop()
    return overlaps


def calc_convexity(sdata, figure_path):

    timer = helperfuncs.Timer()

    # Convexity calculation for cell polygon
    print("[NOTE] Calculate convexity for cells")
    timer.start()
    cell_convexity_metric_list = []
    for poly in sdata['cell_boundaries']['geometry']:
        cell_is_polygon_convex, cell_convexity_metric = is_convex(list(poly.exterior.coords))
        cell_convexity_metric_list.append(cell_convexity_metric)
    timer.stop()

    sdata['table'].obs['convexity_cell'] = [True if x > 0.5 else False for x in cell_convexity_metric_list]
    sdata['table'].obs['convexity_metric_cell'] = cell_convexity_metric_list

    # Find nuceli cell overlaps
    nulcei_of_the_cell = find_overlapping_nuclei(sdata['cell_boundaries'], sdata['nucleus_boundaries'])
    sdata['table'].obs['nuclei_idxs'] = nulcei_of_the_cell

    # Convexity calcualteion for nuclei associated with cell
    print("[NOTE] Calculate convexity for cells")
    timer.start()
    nulcei_convexity_metric_list = []
    min_convexity_metric_list = []
    nuclei_idxs = sdata['table'].obs['nuclei_idxs']
    for cell_nuclei in nuclei_idxs:
            if ( len(cell_nuclei) != 0 ):
                # Since we might have more then one nuclei in a cell we take the mean.
                convexities = []
                for nuceuls_idx in cell_nuclei:
                    nuceuls_poly = sdata['nucleus_boundaries']['geometry'].loc[nuceuls_idx]
                    is_polygon_convex, convexity_metric = is_convex(list(nuceuls_poly.exterior.coords))
                    convexities.append(convexity_metric)
                nulcei_convexity_metric_list.append(np.mean(convexities))
                min_convexity_metric_list.append(np.min(convexities))
            else:
                nulcei_convexity_metric_list.append(0)
                min_convexity_metric_list.append(0)
    timer.stop()

    sdata['table'].obs['convexity_mean_nuceli'] = nulcei_convexity_metric_list
    sdata['table'].obs['convexity_min_nuceli'] = min_convexity_metric_list
    sdata['table'].obs['convexity_nuclei'] = [True if x > 0.5 else False for x in min_convexity_metric_list]

    plotcats = [['convexity_cell', 'convexity_metric_cell'], 
                ['convexity_nuclei', 'convexity_mean_nuceli']]

    figures = []
    for i, cat in enumerate(plotcats):

        helperfuncs.plot_scatter(sdata['table'], figure_path, cat[0], None, 
                                cat[0], ['black', 'lightblue'], None)
        helperfuncs.plot_scatter_density(sdata['table'], figure_path, f'{cat[0]}_{cat[1]}', 
                                         cat[0], cat[1], ['black', 'lightblue'], '' \
                                         f'Density of {cat[1]}')
        
        fig = px.histogram(sdata['table'].obs, x=cat[1], nbins=100, width=800, height=800)
        fig.update_layout(
            title=f"Total distribution {cat[1]} for cells of all samples"
        )
        helperfuncs.apply_general_plotly_layout(fig, True)
        figures.append(fig)
        fig.write_image(f"{figure_path}/histogram_{cat[1]}.png", scale=3)
        fig.write_image(f"{figure_path}/histogram_{cat[1]}.pdf", scale=3)

    with open(f'{figure_path}/convexity.html', 'w') as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


def init_metric(enterprise):

    # These have to be defined.
    metric_name = "convexity"
    combined_metric_name = None
    needs_metrics = []
    step_when_it_is_calculated = ["cellqc", "all"]
    loaded_for_analysis = True
    loaded_for_visualization = True
    prior = False

    # These are given my your metric calc function.
    args = [enterprise.cargo.sdata, f"{enterprise.args.output_dir}/cellqc/"]
    kwargs = None

    metric = core.metric.Metric(
        calc_convexity, 
        metric_name,
        combined_metric_name = combined_metric_name,
        needs_metrics = needs_metrics,
        step_when_it_is_calculated = step_when_it_is_calculated,
        loaded_for_analysis = loaded_for_analysis,
        loaded_for_visualization = loaded_for_visualization,
        prior = prior,
        args = args,
        kwargs = kwargs,
    )    
    
    return metric
