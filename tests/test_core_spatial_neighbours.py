"""Differential tests: spoqc.core.spatial radius/nearest searches vs the verbatim per-object originals.

Each `original_*` below is copied verbatim from spoQC db00d98 (only renamed and given its
inputs as arguments). Fixtures put points exactly at the radius, 1-2 ULPs either side of
it, on duplicates and on ties, in float32 and float64, at Xenium-scale offsets.
"""


import numpy as np
import pandas as pd
import pytest
from scipy.spatial import KDTree, cKDTree

from spoqc.core import spatial
from spoqc.metrics.segmentation import border_score, doublet_score, island_score
from spoqc.metrics.transcript_density import local_moran_I
from spoqc.priors.hqcr import transcript_and_gene_counts

assert not pd.core.computation.expressions.NUMEXPR_INSTALLED, (
    "numexpr changes the originals' dtypes"
)

THREADS = 4
ORIGIN = 20_000.0


# ---------------------------------------------------------------- verbatim originals (db00d98)


def original_points_within_radius(df, radius, num):
    """helperfuncs.points_within_radius."""
    points_in_radius = []
    for i, point in df.iterrows():
        x1, y1 = point["x"], point["y"]
        distances = np.sqrt((df["x"] - x1) ** 2 + (df["y"] - y1) ** 2)
        close_points = df[distances <= radius].index.tolist()
        close_points.remove(i)
        if num:
            points_in_radius.append(len(close_points))
        else:
            points_in_radius.append(close_points)
    return points_in_radius


def original_get_bad_quality_probability(
    x, df, distance_matrix, bad_cluster, qc_cluster
):
    """priors/hqcr/transcript_and_gene_counts.py (byte-identical copy in transcript_counts_celltype.py)."""
    quality_clusters = df.iloc[distance_matrix[x]][qc_cluster]
    number_of_bad_quality_cells = list(quality_clusters.values).count(bad_cluster)
    if len(quality_clusters) != 0:
        return number_of_bad_quality_cells / len(quality_clusters)
    else:
        return 0.0


def original_bad_quality_probabilities(df_coords, cell_df, bad_cluster, qc_cluster):
    """The calc_counts_probs / calc_celltype_transcript_counts_probs call pattern."""
    distance_matrix = original_points_within_radius(df_coords, 30, False)
    return np.array(
        [
            original_get_bad_quality_probability(
                x, cell_df, distance_matrix, bad_cluster, qc_cluster
            )
            for x in range(len(df_coords))
        ]
    )


def original_find_connected_groups_iterative(points, distance_threshold):
    """island_score.find_connected_groups_iterative."""
    points_array = np.array(points)
    tree = KDTree(points_array)
    neighbors = tree.query_ball_point(points_array, distance_threshold)
    visited = set()
    connected_groups = []
    for i in range(len(points)):
        if i not in visited:
            group = []
            queue = [i]
            while queue:
                node = queue.pop(0)
                if node not in visited:
                    visited.add(node)
                    group.append(node)
                    queue.extend(
                        [
                            neighbor
                            for neighbor in neighbors[node]
                            if neighbor not in visited
                        ]
                    )
            connected_groups.append(group)
    return connected_groups


def original_island_scores(spatial_xy, distance_threshold):
    """island_score.calc_island_score, up to the obs writes."""
    adata_x = spatial_xy[:, 0]
    adata_y = spatial_xy[:, 1]
    adata_coordinates = list(zip(adata_x, adata_y))
    groups = original_find_connected_groups_iterative(
        adata_coordinates, distance_threshold
    )
    island_indices = np.array([-1] * len(spatial_xy))
    island_scores = np.array([-1] * len(spatial_xy))
    for i, group in enumerate(groups):
        island_indices[group] = i
        island_scores[group] = len(group)
    return island_indices, island_scores


def original_compute_border_score_for_point(
    point_idx, points, rotation_matrices, radius, tree
):
    x1, y1 = points[point_idx]
    indices = tree.query_ball_point([x1, y1], r=radius)
    if not indices:
        return 0, point_idx
    relevant_points = points[indices]
    diffs = relevant_points - np.array([x1, y1])
    scores = []
    for rotation_matrix in rotation_matrices:
        rotated = diffs @ rotation_matrix
        x_coords = rotated[:, 0]
        num_left = np.count_nonzero(x_coords > 0) + 1
        num_right = np.count_nonzero(x_coords < 0) + 1
        score = abs(np.log2(num_left / num_right))
        scores.append(score)
    return max(scores), point_idx


def original_border_scores(spatial_xy, radius, step):
    """border_score.get_border_scores_optimized + the define_border_cells unpacking (serial map)."""
    df = pd.DataFrame({"x": spatial_xy[:, 0], "y": spatial_xy[:, 1]})
    points = df[["x", "y"]].values
    tree = cKDTree(points)
    angles = np.radians(np.arange(0, 360, step))
    rotation_matrices = np.stack(
        [np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) for a in angles]
    )
    results = np.array(
        [
            original_compute_border_score_for_point(
                idx, points, rotation_matrices, radius, tree
            )
            for idx in range(len(points))
        ]
    )
    indices = results[:, 1].astype(int)
    scores = results[:, 0]
    border_scores = np.full(len(spatial_xy), -1.0)
    border_scores[indices] = scores
    return border_scores


def original_fill_outside(coords, feat, local_I, outside_mask, dist_thresh=100.0):
    """The outside-transcript block of local_moran_I.calculate_local_moran_I_values."""
    inside_mask = ~outside_mask
    features_outside = np.unique(feat[outside_mask])
    for f in features_outside:
        out_idx = np.flatnonzero(outside_mask & (feat == f))
        if out_idx.size == 0:
            continue
        in_idx = np.flatnonzero(inside_mask & (feat == f))
        if in_idx.size == 0:
            local_I[out_idx] = 0.0
            continue
        tree = cKDTree(coords[in_idx])
        dists, nn = tree.query(coords[out_idx], k=1, distance_upper_bound=dist_thresh)
        has_neighbor = np.isfinite(dists) & (nn < in_idx.size)
        local_I[out_idx] = 0.0
        local_I[out_idx[has_neighbor]] = local_I[in_idx[nn[has_neighbor]]]
    return local_I


def original_cells_near_doublets(cell_xy, corrected_doublet_df, distance_thresh):
    """The cell loop of doublet_score.calc_doublet_score."""
    n_obs = len(cell_xy)
    cell_dobulet_df = pd.DataFrame(
        {
            "x": list(cell_xy[:, 0]),
            "y": list(cell_xy[:, 1]),
            "doublet": [False] * n_obs,
            "wdoublet": [0] * n_obs,
            "doublet_distance": [100_000.0] * n_obs,
        }
    )
    final_distances = np.array([100_000.0] * n_obs)
    for i, doublet in corrected_doublet_df.iterrows():
        x1, y1 = doublet["x"], doublet["y"]
        distances = np.sqrt(
            (cell_dobulet_df["x"] - x1) ** 2 + (cell_dobulet_df["y"] - y1) ** 2
        )
        final_distances = np.minimum(final_distances, distances)
        cell_dobulet_df.loc[distances <= distance_thresh, "doublet"] = True
        cell_dobulet_df.loc[distances <= distance_thresh, "wdoublet"] = 1
    cell_dobulet_df["doublet_distance"] = final_distances
    return (
        np.array(cell_dobulet_df["doublet"]),
        np.array(cell_dobulet_df["wdoublet"]),
        np.array(cell_dobulet_df["doublet_distance"]),
    )


# ---------------------------------------------------------------- fixtures


def ulp_steps(v, steps, dtype):
    for _ in range(abs(steps)):
        v = np.nextafter(v, dtype(np.inf * steps), dtype=dtype)
    return v


def boundary_cloud(rng, dtype, radius, n_centres=12, n_random=600, extent=300.0):
    """Centres, points at exactly `radius` (axis and 3-4-5 diagonals) and 1-2 ULPs off, duplicates, random fill."""
    dtype = np.dtype(dtype).type
    centres = (ORIGIN + rng.integers(0, int(extent), (n_centres, 2))).astype(dtype)
    pts = [centres]
    offsets = [
        (radius, 0),
        (0, -radius),
        (-0.6 * radius, 0.8 * radius),
        (0.8 * radius, -0.6 * radius),
    ]
    for cx, cy in centres:
        for ox, oy in offsets + [
            (radius * np.cos(t), radius * np.sin(t))
            for t in rng.uniform(0, 2 * np.pi, 3)
        ]:
            bx, by = dtype(cx + ox), dtype(cy + oy)
            for sx in (-2, -1, 0, 1, 2):
                for sy in (-1, 0, 1):
                    pts.append(
                        np.array(
                            [[ulp_steps(bx, sx, dtype), ulp_steps(by, sy, dtype)]],
                            dtype=dtype,
                        )
                    )
    pts.append(centres[:3])  # exact duplicates of centres
    pts.append(np.array([[ORIGIN - 10 * extent, ORIGIN]], dtype=dtype))  # isolated: no neighbours
    pts.append((ORIGIN + rng.random((n_random, 2)) * extent).astype(dtype))
    xy = np.concatenate(pts)
    return xy[rng.permutation(len(xy))]


def mutants(radius, dtype=np.float64):
    """Wrong radii: one ULP below (drops the pairs exactly at r) and 0.01 above (adds the ULP-outside pairs)."""
    return [np.nextafter(dtype(radius), dtype(0)), radius + 0.01]


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def frame(xy):
    return pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]})


# ---------------------------------------------------------------- pairs_within / neighbour_lists


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("radius", [20, 30, 100])
def test_neighbour_lists_match_points_within_radius(rng, dtype, radius):
    xy = boundary_cloud(rng, dtype, radius)
    expected = original_points_within_radius(frame(xy), radius, False)
    actual = spatial.neighbour_lists(xy, radius, THREADS)
    assert actual == expected
    assert all(type(v) is int for lst in actual for v in lst)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_boundary_fixture_straddles_radius(rng, dtype):
    """Some pairs sit exactly at r and some a coordinate ULP beyond it: the fixture is live."""
    xy = boundary_cloud(rng, dtype, 30)
    d = np.sqrt(
        (xy[:, None, 0] - xy[None, :, 0]) ** 2 + (xy[:, None, 1] - xy[None, :, 1]) ** 2
    )
    assert (d == 30).any() and ((d > 30) & (d < 30.01)).any()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("which", [0, 1])
def test_wrong_radius_is_detected(rng, dtype, which):
    """Mutant: one ULP more radius must change the neighbour lists on this fixture."""
    xy = boundary_cloud(rng, dtype, 30)
    expected = original_points_within_radius(frame(xy), 30, False)
    mutant = spatial.neighbour_lists(
        xy, mutants(30, dtype)[which], THREADS
    )
    assert mutant != expected


def test_pairs_within_cross_sets_in_query_dtype_matches_pandas(rng):
    """float32 reference Series vs float64 scalars: pandas computes in float32 (the doublet case)."""
    ref = boundary_cloud(rng, np.float32, 10)
    query = ref[:40].astype(np.float64) + rng.integers(-3, 4, (40, 2))
    series = frame(ref)
    expected = [
        np.flatnonzero(
            (
                np.sqrt((series["x"] - x1) ** 2 + (series["y"] - y1) ** 2) <= 10
            ).to_numpy()
        )
        for x1, y1 in query
    ]
    q, r = spatial.pairs_within(query, ref, 10, THREADS, dtype=np.float32)
    offsets = np.searchsorted(q, np.arange(len(query) + 1))
    assert [r[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])] == [
        e.tolist() for e in expected
    ]


def test_pairs_within_is_thread_invariant(rng):
    xy = boundary_cloud(rng, np.float32, 30)
    one = spatial.pairs_within(xy, xy, 30, 1)
    many = spatial.pairs_within(xy, xy, 30, 7)
    assert all(np.array_equal(a, b) for a, b in zip(one, many))


# ---------------------------------------------------------------- nearest


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nearest_distance_is_the_brute_force_minimum(rng, dtype):
    ref = boundary_cloud(rng, dtype, 10)[:300]
    # queries equidistant from several refs (ties) and 1 ULP off them
    query = np.concatenate(
        [ref[:50] + dtype(5), ref[50:80], boundary_cloud(rng, dtype, 7)[:200]]
    ).astype(dtype)
    series = frame(ref)
    expected = np.array(
        [
            np.sqrt((series["x"] - x1) ** 2 + (series["y"] - y1) ** 2).min()
            for x1, y1 in query
        ]
    )
    _, actual = spatial.nearest(query, ref, THREADS)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected, strict=True)


def test_nearest_index_is_cKDTree_query(rng):
    ref = boundary_cloud(rng, np.float64, 10)
    query = boundary_cloud(rng, np.float64, 10)
    d, i = cKDTree(ref).query(query, k=1, distance_upper_bound=12.0)
    pos, _ = spatial.nearest(query, ref, THREADS, distance_upper_bound=12.0)
    np.testing.assert_array_equal(pos, i, strict=True)


# ---------------------------------------------------------------- callers


@pytest.mark.parametrize(
    "clusters,bad",
    [
        ("int", 0),  # reduce_cluster_num_for_hqcr: 0/1/2, bad = np.argmin(...)
        ("float", 1),  # hard threshold branch: np.zeros(len) float clusters
        ("celltype", 1),  # qc_celltype_class
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_bad_quality_probabilities_match_original(rng, clusters, bad, dtype):
    xy = boundary_cloud(rng, dtype, 30)
    n = len(xy)
    values = {
        "int": rng.integers(0, 3, n),
        "float": rng.integers(0, 2, n).astype(float),
        "celltype": np.where(rng.random(n) < 0.3, 1, 0),
    }[clusters]
    cell_df = pd.DataFrame({"qc": values}, index=[f"cell{i}" for i in range(n)])
    bad_cluster = np.int64(bad)
    df_coords = frame(xy)
    expected = original_bad_quality_probabilities(df_coords, cell_df, bad_cluster, "qc")
    actual = transcript_and_gene_counts.get_bad_quality_probabilities(
        df_coords, cell_df["qc"].to_numpy(), bad_cluster, THREADS
    )
    np.testing.assert_array_equal(actual, expected, strict=True)
    assert (expected == 0.0).any() and ((expected > 0) & (expected < 1)).any()


@pytest.mark.parametrize("which", [0, 1])
def test_bad_quality_probabilities_detect_wrong_radius(rng, monkeypatch, which):
    xy = boundary_cloud(rng, np.float64, 30)
    cell_df = pd.DataFrame({"qc": rng.integers(0, 3, len(xy))})
    expected = original_bad_quality_probabilities(frame(xy), cell_df, np.int64(0), "qc")
    monkeypatch.setattr(
        transcript_and_gene_counts, "NEIGHBOURHOOD_RADIUS", mutants(30)[which]
    )
    mutant = transcript_and_gene_counts.get_bad_quality_probabilities(
        frame(xy), cell_df["qc"].to_numpy(), np.int64(0), THREADS
    )
    assert not np.array_equal(mutant, expected)


def chain_cloud(rng, dtype, step):
    """Chains spaced exactly `step` apart (connected) and one coordinate ULP more (broken), plus a random fill."""
    dtype = np.dtype(dtype).type
    pts = []
    for k in range(6):
        x, y0 = dtype(ORIGIN + 200 * k), dtype(ORIGIN)
        for j in range(5 + k):
            pts.append((x, y0))
            x = dtype(x + dtype(step))
            if k % 2:
                x = np.nextafter(x, dtype(np.inf), dtype=dtype)
    random = ORIGIN + rng.random((800, 2)) * 1500
    xy = np.concatenate([np.array(pts, dtype=dtype), random.astype(dtype)])
    return xy[rng.permutation(len(xy))]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_island_scores_match_original(rng, dtype):
    xy = np.concatenate([chain_cloud(rng, dtype, 15), boundary_cloud(rng, dtype, 15)])
    expected_idx, expected_score = original_island_scores(xy, 15)
    actual_idx = island_score.find_connected_groups(xy, 15, THREADS)
    actual_score = np.bincount(actual_idx)[actual_idx]
    np.testing.assert_array_equal(actual_idx, expected_idx, strict=True)
    np.testing.assert_array_equal(actual_score, expected_score, strict=True)
    assert len(np.unique(expected_score)) > 3


@pytest.mark.parametrize("which", [0, 1])
def test_island_scores_detect_wrong_radius(rng, which):
    xy = chain_cloud(rng, np.float64, 15)
    expected_idx, _ = original_island_scores(xy, 15)
    assert not np.array_equal(
        island_score.find_connected_groups(
            xy, mutants(15)[which], THREADS
        ),
        expected_idx,
    )


def border_cloud(rng, dtype):
    """Grid points (rotated neighbours land exactly on 0 at multiples of 90 degrees), boundary rings and a random fill."""
    grid = ORIGIN + np.stack(
        np.meshgrid(np.arange(0, 200, 10.0), np.arange(0, 120, 10.0)), -1
    ).reshape(-1, 2)
    return np.concatenate(
        [grid, boundary_cloud(rng, np.float64, 50, n_random=900, extent=400)]
    ).astype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("step", [10, 45])
def test_border_scores_match_original(rng, dtype, step):
    xy = border_cloud(rng, dtype)
    expected = original_border_scores(xy, 50, step)
    actual = border_score.get_border_scores(xy, 50, step, THREADS)
    np.testing.assert_array_equal(actual, expected, strict=True)
    assert len(np.unique(expected)) > 10


@pytest.mark.parametrize("which", [0, 1])
def test_border_scores_detect_wrong_radius(rng, which):
    xy = border_cloud(rng, np.float64)
    expected = original_border_scores(xy, 50, 10)
    assert not np.array_equal(
        border_score.get_border_scores(
            xy, mutants(50)[which], 10, THREADS
        ),
        expected,
    )


def moran_case(rng):
    """float32 transcript coordinates read as float64, three genes, outside transcripts with ties and at the cutoff."""
    n = 6000
    xy32 = (ORIGIN + rng.random((n, 2)) * 1500).astype(np.float32)
    feat = rng.choice(np.array(["A", "B", "C", "D"], dtype=object), n)
    outside = rng.random(n) < 0.3
    feat[(feat == "D") & ~outside] = "A"  # 'D' has no inside transcripts
    extra_xy, extra_feat, extra_out = [], [], []
    for k in range(
        40
    ):  # an outside point equidistant from two inside points of the same gene
        cx, cy = ORIGIN + 3000 + 30 * k, ORIGIN
        extra_xy += [(cx, cy), (cx - 4, cy), (cx + 4, cy), (cx, cy + 3)]
        extra_feat += ["B", "B", "B", "C"]
        extra_out += [True, False, False, True]
    for k in range(20):  # outside points exactly 100 from the only nearby inside point
        cx, cy = ORIGIN + 6000 + 400 * k, ORIGIN
        extra_xy += [(cx, cy), (cx + 100, cy), (cx, cy + 60), (cx + 80, cy + 60)]
        extra_feat += ["C", "C", "C", "C"]
        extra_out += [False, True, False, True]
    xy32 = np.concatenate([xy32, np.array(extra_xy, dtype=np.float32)])
    coords = xy32.astype(np.float64)
    feat = np.concatenate([feat, np.array(extra_feat, dtype=object)])
    outside = np.concatenate([outside, extra_out])
    local_I = np.where(outside, -1.0, rng.random(len(coords))).astype(np.float32)
    return coords, feat, local_I, outside


def test_fill_outside_from_nearest_inside_matches_original(rng):
    coords, feat, local_I, outside = moran_case(rng)
    expected = original_fill_outside(coords, feat, local_I.copy(), outside)
    actual = local_moran_I.fill_outside_from_nearest_inside(
        coords, feat, local_I.copy(), outside, THREADS
    )
    np.testing.assert_array_equal(actual, expected, strict=True)
    assert (expected[outside] == 0.0).any() and (expected[outside] > 0).any()


def doublet_case(rng):
    """Cells with float64 centroids; ovrlpy-style doublets; cells on, 1 ULP inside/outside 10 of a doublet, ties, far cells."""
    n_cells = 3000
    cells = ORIGIN + rng.random((n_cells, 2)) * 800
    doublets = pd.DataFrame(
        {
            "x": rng.integers(0, 800, 40).astype(np.int64) + ORIGIN,
            "y": rng.integers(0, 800, 40).astype(np.int64) + ORIGIN,
            "integrity": rng.random(40).astype(np.float32),
            "signal": rng.random(40).astype(np.float32),
        }
    )
    extra = []
    for x1, y1 in doublets[["x", "y"]].to_numpy()[:15]:
        for ox, oy in [(10, 0), (6, 8), (0, -10), (-8, 6)]:
            bx, by = x1 + ox, y1 + oy
            extra += [
                (bx, by),
                (np.nextafter(bx, np.inf), by),
                (np.nextafter(bx, -np.inf), by),
            ]
    x1, y1 = doublets.loc[0, ["x", "y"]]
    doublets.loc[1, ["x", "y"]] = [
        x1 + 20,
        y1,
    ]  # a cell midway between two doublets: a tie
    extra += [
        (x1 + 10, y1),
        (ORIGIN + 500_000, ORIGIN),
    ]  # and one farther than 100_000 from all
    cells = np.concatenate([cells, np.array(extra)])
    return cells, doublets


@pytest.mark.parametrize("n_doublets", [None, 0, 1])
def test_cells_near_doublets_match_original(rng, n_doublets):
    cells, doublets = doublet_case(rng)
    doublets = doublets if n_doublets is None else doublets.iloc[:n_doublets]
    expected = original_cells_near_doublets(cells, doublets, 10)
    actual = doublet_score.flag_cells_near_doublets(
        cells, doublets[["x", "y"]].to_numpy(), 10, THREADS
    )
    for exp, act in zip(expected, actual):
        np.testing.assert_array_equal(act, exp, strict=True)
    if n_doublets is None:
        assert (
            expected[0].any()
            and not expected[0].all()
            and (expected[2] == 100_000.0).any()
        )


@pytest.mark.parametrize("which", [0, 1])
def test_cells_near_doublets_detect_wrong_radius(rng, which):
    cells, doublets = doublet_case(rng)
    expected = original_cells_near_doublets(cells, doublets, 10)
    mutant = doublet_score.flag_cells_near_doublets(
        cells, doublets[["x", "y"]].to_numpy(), mutants(10)[which], THREADS
    )
    assert not np.array_equal(mutant[0], expected[0])


# ---------------------------------------------------------------- review fixes

def straddle_cloud(rng, radius, origin=5000.0, n_centres=60, n_angles=24):
    """float32 centroids (Xenium-scale obsm) with pairs whose float64 squared distance lies
    within one float32 ULP of radius**2 on either side: the float32 sqrt formula and the
    tree's float64 sum of squares disagree on some of them."""
    centres = (origin + rng.random((n_centres, 2)) * 400).astype(np.float32)
    theta = rng.uniform(0, 2 * np.pi, (n_centres, n_angles))
    jitter = rng.uniform(-1, 1, (n_centres, n_angles)) * np.spacing(np.float32(radius))
    ring = centres[:, None, :] + (radius + jitter)[..., None] * np.stack((np.cos(theta), np.sin(theta)), -1)
    xy = np.concatenate([centres, ring.reshape(-1, 2).astype(np.float32)])
    return xy[rng.permutation(len(xy))]


def straddle_pairs(rng, radius, origin=5000.0, n_pairs=2000, spacing=100.0):
    """Isolated float32 pairs, one straddling point each: a changed decision changes the islands."""
    grid = np.stack(np.meshgrid(np.arange(50), np.arange(n_pairs // 50)), -1).reshape(-1, 2)
    centres = (origin + grid * spacing).astype(np.float32)
    theta = rng.uniform(0, 2 * np.pi, len(centres))
    dist = radius + rng.uniform(-1, 1, len(centres)) * np.spacing(np.float32(radius))
    partners = (centres + dist[:, None] * np.stack((np.cos(theta), np.sin(theta)), -1)).astype(np.float32)
    xy = np.concatenate([centres, partners])
    return xy[rng.permutation(len(xy))]


def formula_and_tree_disagree(xy, radius):
    q_f, r_f = spatial.pairs_within(xy, xy, radius, THREADS)
    q_t, r_t = spatial.pairs_within(xy, xy, radius, THREADS, decide="tree")
    return not (np.array_equal(q_f, q_t) and np.array_equal(r_f, r_t))


def test_straddle_fixture_separates_formula_from_tree(rng):
    assert formula_and_tree_disagree(straddle_cloud(rng, 15), 15)
    assert formula_and_tree_disagree(straddle_cloud(rng, 50), 50)


@pytest.mark.parametrize("leafsize", [10, 16])
def test_pairs_within_tree_is_query_ball_point(rng, leafsize):
    xy = straddle_cloud(rng, 15)
    tree = cKDTree(xy.astype(np.float64), leafsize=leafsize)
    lists = tree.query_ball_point(xy.astype(np.float64), 15)
    q, r = spatial.pairs_within(xy, xy, 15, THREADS, decide="tree", leafsize=leafsize)
    offsets = np.searchsorted(q, np.arange(len(xy) + 1))
    assert [r[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])] == [sorted(x) for x in lists]


@pytest.mark.parametrize("fixture", [straddle_cloud, straddle_pairs])
def test_island_scores_match_original_on_float32_straddle(rng, fixture):
    xy = fixture(rng, 15)
    expected_idx, expected_score = original_island_scores(xy, 15)
    actual_idx = island_score.find_connected_groups(xy, 15, THREADS)
    np.testing.assert_array_equal(actual_idx, expected_idx, strict=True)
    np.testing.assert_array_equal(np.bincount(actual_idx)[actual_idx], expected_score, strict=True)


def test_island_scores_detect_formula_decision_on_float32_straddle(rng, monkeypatch):
    """Mutant: deciding with the float32 sqrt formula changes the islands."""
    xy = straddle_pairs(rng, 15)
    expected_idx, _ = original_island_scores(xy, 15)
    formula = spatial.pairs_within
    monkeypatch.setattr(spatial, "pairs_within", lambda q, r, rad, threads, **_: formula(q, r, rad, threads))
    assert not np.array_equal(island_score.find_connected_groups(xy, 15, THREADS), expected_idx)


def test_border_scores_detect_formula_decision_on_float32_straddle(rng, monkeypatch):
    xy = straddle_cloud(rng, 50)
    expected = original_border_scores(xy, 50, 10)
    formula = spatial.pairs_within
    monkeypatch.setattr(spatial, "pairs_within", lambda q, r, rad, threads, **_: formula(q, r, rad, threads))
    assert not np.array_equal(border_score.get_border_scores(xy, 50, 10, THREADS), expected)


@pytest.mark.parametrize("step", [10, 45])
def test_border_scores_match_original_on_float32_straddle(rng, step):
    xy = straddle_cloud(rng, 50)
    expected = original_border_scores(xy, 50, step)
    actual = border_score.get_border_scores(xy, 50, step, THREADS)
    np.testing.assert_array_equal(actual, expected, strict=True)


def test_border_scores_match_original_on_symmetric_offsets(rng):
    """Neighbours at (d, -d) etc.: the 45-degree rotation puts them within rounding of 0, where an
    FMA and a separately rounded product-sum can disagree in sign; the per-point matmul decides."""
    base = ORIGIN + np.stack(np.meshgrid(np.arange(0, 120, 7.0), np.arange(0, 120, 7.0)), -1).reshape(-1, 2)
    xy = np.concatenate([base, base + [3.0, -3.0], base + [-5.0, 5.0]])
    for step in (5, 45):
        expected = original_border_scores(xy, 12, step)
        np.testing.assert_array_equal(border_score.get_border_scores(xy, 12, step, THREADS), expected, strict=True)


@pytest.mark.parametrize("threads", [1, 2, 3, 4, 5, 8, 16, 33])
@pytest.mark.parametrize("n_radii", [1, 2, 5, 7])
def test_split_threads_stays_within_budget(threads, n_radii):
    from spoqc.subworkflows.qc_marker import split_threads
    workers, per_task = split_threads(threads, n_radii)
    assert workers * per_task <= threads and workers >= 1 and per_task >= 1
    assert workers == min(threads, n_radii)


def test_border_matmuls_run_on_one_blas_thread(rng, monkeypatch):
    from threadpoolctl import threadpool_info
    seen = []
    original = border_score.np.count_nonzero

    def recording(*args, **kwargs):
        seen.append(max((lib["num_threads"] for lib in threadpool_info() if lib["user_api"] == "blas"), default=1))
        return original(*args, **kwargs)

    monkeypatch.setattr(border_score.np, "count_nonzero", recording)
    border_score.get_border_scores(border_cloud(rng, np.float64)[:200], 50, 90, THREADS)
    assert seen and max(seen) == 1


@pytest.mark.parametrize("where", ["cell", "doublet"])
def test_cells_near_doublets_fail_loudly_on_nan(rng, where):
    """The original silently gave NaN doublet_distance; scipy's KD-tree rejects non-finite input."""
    cells, doublets = doublet_case(rng)
    doublet_xy = doublets[["x", "y"]].to_numpy()
    if where == "cell":
        cells[5, 0] = np.nan
    else:
        doublet_xy[3, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        doublet_score.flag_cells_near_doublets(cells, doublet_xy, 10, THREADS)
