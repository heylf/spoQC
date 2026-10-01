"""Differential test: vectorised convexity QC vs a verbatim copy of the original (db00d98)."""

from typing import List, Tuple

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from shapely.geometry import Polygon, box

from spoqc import helperfuncs
from spoqc.metrics.segmentation import convexity


# ---------------------------------------------------------------------------
# Verbatim reference: spoqc/metrics/segmentation/convexity.py at db00d98
# (calc_convexity is copied up to, not including, the plotting section).
# ---------------------------------------------------------------------------
def is_convex(polygon: List[Tuple[float, float]]) -> Tuple[bool, float]:
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


def find_overlapping_nuclei(cells: gpd.GeoDataFrame, nucleus: gpd.GeoDataFrame):
    overlaps = []
    nucleus_centroids = nucleus.geometry.centroid
    for cell in cells.geometry:
        overlapping_indices = nucleus[nucleus_centroids.geometry.intersects(cell)].index.tolist()
        overlaps.append(overlapping_indices)
    return overlaps


def reference_calc_convexity(sdata):
    cell_convexity_metric_list = []
    for poly in sdata['cell_boundaries']['geometry']:
        cell_is_polygon_convex, cell_convexity_metric = is_convex(list(poly.exterior.coords))
        cell_convexity_metric_list.append(cell_convexity_metric)

    sdata['table'].obs['convexity_cell'] = [True if x > 0.5 else False for x in cell_convexity_metric_list]
    sdata['table'].obs['convexity_metric_cell'] = cell_convexity_metric_list

    nulcei_of_the_cell = find_overlapping_nuclei(sdata['cell_boundaries'], sdata['nucleus_boundaries'])
    sdata['table'].obs['nuclei_idxs'] = nulcei_of_the_cell

    nulcei_convexity_metric_list = []
    min_convexity_metric_list = []
    nuclei_idxs = sdata['table'].obs['nuclei_idxs']
    for cell_nuclei in nuclei_idxs:
            if ( len(cell_nuclei) != 0 ):
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

    sdata['table'].obs['convexity_mean_nuceli'] = nulcei_convexity_metric_list
    sdata['table'].obs['convexity_min_nuceli'] = min_convexity_metric_list
    sdata['table'].obs['convexity_nuclei'] = [True if x > 0.5 else False for x in min_convexity_metric_list]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
OUTPUT_COLUMNS = ['convexity_cell', 'convexity_metric_cell', 'nuclei_idxs',
                  'convexity_mean_nuceli', 'convexity_min_nuceli', 'convexity_nuclei']
PIXEL_SIZE = 0.2125  # Xenium micron-per-pixel, gives realistic non-integer coordinates


def make_sdata(cells, nuclei, nucleus_index=None):
    cell_gdf = gpd.GeoDataFrame(geometry=cells)
    nucleus_gdf = gpd.GeoDataFrame(geometry=nuclei, index=nucleus_index)
    table = ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(len(cells))]))
    return {'cell_boundaries': cell_gdf, 'nucleus_boundaries': nucleus_gdf, 'table': table}


def copy_sdata(sdata):
    return {'cell_boundaries': sdata['cell_boundaries'].copy(),
            'nucleus_boundaries': sdata['nucleus_boundaries'].copy(),
            'table': sdata['table'].copy()}


@pytest.fixture
def no_plots(monkeypatch):
    monkeypatch.setattr(helperfuncs, 'plot_scatter', lambda *a, **k: None)
    monkeypatch.setattr(helperfuncs, 'plot_scatter_density', lambda *a, **k: None)
    monkeypatch.setattr(go.Figure, 'write_image', lambda *a, **k: None)


def assert_identical(sdata, tmp_path, threads):
    expected = copy_sdata(sdata)
    with pytest.warns(DeprecationWarning):  # np.cross on 2-vectors, numpy >= 2.0
        reference_calc_convexity(expected)
    actual = copy_sdata(sdata)
    convexity.calc_convexity(actual, str(tmp_path), threads)
    for column in OUTPUT_COLUMNS:
        exp, act = expected['table'].obs[column], actual['table'].obs[column]
        assert exp.dtype == act.dtype, f"{column}: dtype {exp.dtype} != {act.dtype}"
        if column == 'nuclei_idxs':
            assert act.tolist() == exp.tolist(), column
            assert [type(x) for labels in act for x in labels] == [type(x) for labels in exp for x in labels], column
        else:
            np.testing.assert_array_equal(act.to_numpy(), exp.to_numpy(), err_msg=column, strict=True)
            # bit-level equality (distinguishes -0.0 / NaN payloads)
            assert act.to_numpy().tobytes() == exp.to_numpy().tobytes(), column


def star(cx, cy, n_points, r_out, r_in, clockwise=False):
    angles = np.linspace(0, 2 * np.pi, 2 * n_points, endpoint=False)
    radii = np.where(np.arange(2 * n_points) % 2 == 0, r_out, r_in)
    xy = np.column_stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)])
    xy = np.round(xy / PIXEL_SIZE) * PIXEL_SIZE
    return Polygon(xy[::-1] if clockwise else xy)


def blob(rng, cx, cy, radius, n_vertices):
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_vertices))
    radii = radius * rng.uniform(0.5, 1.0, n_vertices)
    xy = np.column_stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)])
    return Polygon(np.round(xy / PIXEL_SIZE) * PIXEL_SIZE)


def nucleus_at(cx, cy, size=0.5):
    """Square nucleus whose centroid is exactly (cx, cy)."""
    return box(cx - size, cy - size, cx + size, cy + size)


@pytest.fixture
def edge_case_sdata():
    rng = np.random.default_rng(0)
    cells, nuclei = [], []
    # Row of adjacent 10x10 cells sharing edges
    for i in range(6):
        cells.append(box(10 * i, 0, 10 * i + 10, 10))
    nuclei += [nucleus_at(5, 5)]                                   # cell 0: one nucleus
    nuclei += [nucleus_at(12, 3), nucleus_at(17, 7)]               # cell 1: two
    nuclei += [nucleus_at(20, 5)]                                  # shared edge of cells 1 and 2 (touching)
    nuclei += [nucleus_at(30, 10)]                                 # shared vertex of cells 2, 3 on top edge
    nuclei += [nucleus_at(41 + j, 1 + j, 0.3) for j in range(9)]   # cell 4: nine nuclei (pairwise-sum regime)
    nuclei += [blob(rng, 55, 5, 2, 12) for _ in range(12)]         # cell 5: twelve concave nuclei
    # Cells without nuclei, concave and clockwise shapes, collinear vertices, a hole
    cells.append(star(100, 100, 5, 8, 3))
    cells.append(star(120, 100, 7, 8, 5, clockwise=True))
    cells.append(Polygon([(0, 20), (5, 20), (10, 20), (10, 30), (0, 30)]))           # collinear vertex
    cells.append(Polygon([(0, 40), (10, 40), (10, 50), (0, 50)][::-1]))              # clockwise square
    cells.append(Polygon([(20, 20), (40, 20), (40, 40), (20, 40)],
                         [[(25, 25), (35, 25), (35, 35), (25, 35)]]))                # hole: exterior only
    nuclei += [nucleus_at(30, 30)]                                                   # centroid in the hole
    nuclei += [nucleus_at(26, 22)]                                                   # centroid in the ring
    # Invalid bow-tie cell as loaded, and the convex hull correct_for_valid_geometries makes of it
    bowtie = Polygon([(60, 20), (70, 30), (70, 20), (60, 30)])
    cells += [bowtie, bowtie.convex_hull]
    nuclei += [nucleus_at(62, 25, 0.2), nucleus_at(68, 25, 0.2), nucleus_at(65, 29, 0.2)]
    # Invalid nucleus: a concave bow-tie nucleus in cell 0 and its corrected hull in cell 1
    nuclei += [Polygon([(2, 2), (4, 4), (4, 2), (2, 4)])]
    # Many random cells with random concave nuclei around them
    for j in range(40):
        cx, cy = rng.uniform(200, 400, 2)
        cells.append(blob(rng, cx, cy, 6, int(rng.integers(8, 40))))
        for _ in range(int(rng.integers(0, 15))):
            nuclei.append(blob(rng, cx + rng.normal(0, 3), cy + rng.normal(0, 3), 1.5, int(rng.integers(5, 20))))
    return cells, nuclei


@pytest.mark.parametrize('threads', [1, 2, 3, 8])
def test_calc_convexity_matches_original_on_edge_cases(edge_case_sdata, no_plots, tmp_path, threads):
    cells, nuclei = edge_case_sdata
    # Shuffled, non-contiguous labels so positions and labels cannot be confused
    labels = np.random.default_rng(1).permutation(len(nuclei)) * 3 + 1
    assert_identical(make_sdata(cells, nuclei, labels), tmp_path, threads)


def test_calc_convexity_matches_original_with_string_labels(edge_case_sdata, no_plots, tmp_path):
    cells, nuclei = edge_case_sdata
    labels = [f'nuc-{i}' for i in np.random.default_rng(2).permutation(len(nuclei))]
    assert_identical(make_sdata(cells, nuclei, labels), tmp_path, 2)


def test_calc_convexity_matches_original_when_no_cell_has_a_nucleus(no_plots, tmp_path):
    cells = [box(0, 0, 10, 10), star(50, 50, 5, 8, 3)]
    nuclei = [nucleus_at(100, 100), nucleus_at(-20, 5)]
    assert_identical(make_sdata(cells, nuclei), tmp_path, 2)


def test_calc_convexity_matches_original_without_nuclei(no_plots, tmp_path):
    cells = [box(0, 0, 10, 10), star(50, 50, 5, 8, 3)]
    assert_identical(make_sdata(cells, []), tmp_path, 2)


def test_calc_convexity_matches_original_on_random_dense_tissue(no_plots, tmp_path):
    rng = np.random.default_rng(3)
    centers = rng.uniform(0, 300, (400, 2))
    cells = [blob(rng, x, y, 8, int(rng.integers(6, 60))) for x, y in centers]
    nuclei = [blob(rng, x + rng.normal(0, 2), y + rng.normal(0, 2), 2.5, int(rng.integers(4, 30)))
              for x, y in np.repeat(centers, rng.integers(0, 4, len(centers)), axis=0)]
    assert_identical(make_sdata(cells, nuclei, np.arange(1, len(nuclei) + 1)), tmp_path, 3)


def test_convexity_metrics_rejects_degenerate_polygon():
    with pytest.raises(ValueError, match="at least three vertices"):
        convexity.convexity_metrics(np.array([box(0, 0, 1, 1), Polygon()]), 2)
