import numpy as np
import shapely

def calculate_overlap_areas(sdata):
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
