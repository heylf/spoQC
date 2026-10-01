import numpy as np
import plotly.express as px
import shapely

from concurrent.futures import ThreadPoolExecutor

from ... import helperfuncs
from ...core import groupreduce, spatial
from spoqc.core.figures import save_figure

def convexity_metrics(polygons: np.ndarray, threads: int) -> np.ndarray:
    """
    Measures the convexity of every polygon's exterior ring at once.

    For each ring (closing vertex included) the cross product of each pair of
    consecutive edges is computed; the metric is the larger of the number of
    positive and negative cross products divided by the number of vertices
    (range [0, 1]; 1 is fully convex).

    Parameters:
        polygons (np.ndarray): Array of shapely Polygons.
        threads (int): Number of threads; each handles a contiguous chunk of polygons.

    Returns:
        np.ndarray: float64 convexity metric per polygon.
    """
    with ThreadPoolExecutor(threads) as executor:
        return np.concatenate(list(executor.map(_ring_convexity, np.array_split(polygons, threads))))


def _ring_convexity(polygons: np.ndarray) -> np.ndarray:
    coords, ring = shapely.get_coordinates(shapely.get_exterior_ring(polygons), return_index=True)
    offsets = groupreduce.group_offsets(ring, len(polygons))
    n = np.diff(offsets)
    if (n < 3).any():
        raise ValueError("A polygon must have at least three vertices.")

    # Three consecutive points per vertex, wrapping around within each ring
    p2 = coords[groupreduce.cyclic_shift(offsets, 1)]
    p3 = coords[groupreduce.cyclic_shift(offsets, 2)]
    v1 = p2 - coords
    v2 = p3 - p2
    cross_products = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    pos = groupreduce.group_count(ring, cross_products > 0, len(polygons))
    neg = groupreduce.group_count(ring, cross_products < 0, len(polygons))

    # Just take the maximum amount of consistent angles
    return np.maximum(neg, pos) / n


def calc_convexity(sdata, figure_path, threads):

    timer = helperfuncs.Timer()

    # Convexity calculation for cell polygon
    print("[NOTE] Calculate convexity for cells")
    timer.start()
    cell_convexity_metric = convexity_metrics(np.asarray(sdata['cell_boundaries'].geometry.values), threads)
    timer.stop()

    sdata['table'].obs['convexity_cell'] = cell_convexity_metric > 0.5
    sdata['table'].obs['convexity_metric_cell'] = cell_convexity_metric

    # Find nuceli cell overlaps: nuclei whose centroid lies in the cell
    print("[NOTE] Find overlapping nuceli for cells")
    timer.start()
    n_cells = len(sdata['cell_boundaries'])
    cell_pos, nucleus_pos = spatial.polygons_containing(
        np.asarray(sdata['cell_boundaries'].geometry.values),
        np.asarray(sdata['nucleus_boundaries'].geometry.centroid.values),
        threads)
    offsets = groupreduce.group_offsets(cell_pos, n_cells)
    nucleus_labels = sdata['nucleus_boundaries'].index.values[nucleus_pos].tolist()
    sdata['table'].obs['nuclei_idxs'] = groupreduce.ragged_lists(nucleus_labels, offsets)
    timer.stop()

    # Convexity calcualteion for nuclei associated with cell
    print("[NOTE] Calculate convexity for cells")
    timer.start()
    nuclei_convexity = convexity_metrics(np.asarray(sdata['nucleus_boundaries'].geometry.values)[nucleus_pos], threads)
    # Cells without nuclei get 0 (an integer column if no cell has a nucleus).
    # Since we might have more then one nuclei in a cell we take the mean.
    no_nuclei = np.zeros(n_cells, dtype=float if len(nucleus_pos) else int)
    nulcei_convexity_metric = groupreduce.ragged_reduce(nuclei_convexity, offsets, np.mean, no_nuclei.copy())
    min_convexity_metric = groupreduce.ragged_reduce(nuclei_convexity, offsets, np.min, no_nuclei.copy())
    timer.stop()

    sdata['table'].obs['convexity_mean_nuceli'] = nulcei_convexity_metric
    sdata['table'].obs['convexity_min_nuceli'] = min_convexity_metric
    sdata['table'].obs['convexity_nuclei'] = min_convexity_metric > 0.5

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
        save_figure(fig, f"{figure_path}/histogram_{cat[1]}.png", f"{figure_path}/histogram_{cat[1]}.pdf", scale=3)

    with open(f'{figure_path}/convexity.html', 'w') as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))