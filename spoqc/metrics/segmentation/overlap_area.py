import numpy as np
import shapely

from ... import core

def calc_overlap_areas(sdata):
    print("[NOTE] Calculate overlap areas")
    cells = sdata['cell_boundaries']
    geometries = cells.geometry.values
    n = len(geometries)
    overlap_areas = np.zeros(n)

    tree = shapely.STRtree(geometries)
    left, right = tree.query(geometries, predicate='intersects')

    # keep each pair once — eliminates self-pairs and duplicates
    mask = left < right
    left, right = left[mask], right[mask]

    pair_areas = shapely.area(shapely.intersection(geometries[left], geometries[right]))

    np.add.at(overlap_areas, left, pair_areas)
    np.add.at(overlap_areas, right, pair_areas)

    sdata['table'].obs['cell_overlap_area'] = overlap_areas



def init_metric(enterprise):

    # These have to be defined.
    metric_name = "overlap_area"
    combined_metric_name = None
    needs_metrics = []
    step_when_it_is_calculated = ["doubletqc", "all"]
    loaded_for_analysis = True
    loaded_for_visualization = True
    prior = False

    # These are given my your metric calc function.
    args = [enterprise.cargo.sdata]
    kwargs = None

    metric = core.metric.Metric(
        calc_overlap_areas, 
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
