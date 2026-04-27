
def calculate_overlap_areas(sdata):
    cells = sdata['cell_boundaries']
    overlap_areas = []
    
    for i, cell in cells.iterrows():
        overlapping_area = sum(cells.loc[cells.index != i, "geometry"].intersection(cell.geometry).area)
        overlap_areas.append(overlapping_area)

    sdata['table'].obs['cell_overlap_area'] = overlap_areas