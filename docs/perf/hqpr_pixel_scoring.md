# hqpr/hqtr pixel scoring: differences from origin/dev

Pixel scoring (`image_analysis/pixel_scoring_dask.start_pixel_qc`) is bit-identical to origin/dev db00d98 except for the two changes below.
Both were measured on a real 4000 x 4000 crop of `breast2.zarr` (morphology_focus s0, staining 0, 16,000,000 pixels).

## 1. Cluster means: a deterministic reduction

**origin/dev:** the per-cluster mean intensity, s_score and as_score came from a dask `groupby('cluster').mean()`.
Dask runs it as a disk shuffle, and the partition results are combined in task-completion order.
The means therefore changed in their last bits from run to run: 5 runs on the crop gave 5 different results.
pandas also summed the float32 scores in float32.

**Now:** `core/groupreduce.group_sum` computes a float64 sum and a count per cluster.
Rows are added in order within fixed chunks of 2^20 rows, and the chunks are then added in chunk order.
The result is identical across runs and thread counts (tested with 1, 2 and 4 numba threads).
The clusters are passed to the GMM prior in cluster-id order; origin/dev used its shuffle order.

**Difference on the crop** (on origin/dev's own labels):

| Quantity | Max abs diff | Max rel diff | Clusters differing |
| --- | --- | --- | --- |
| Mean intensity | 0 | 0 | 0 of 100 |
| Mean s_score | 5.4e-7 | 5.6e-8 | 100 of 100 |
| Mean as_score | 1.2e-7 | 4.9e-8 | 100 of 100 |
| Pixel score after `np.round(..., 2)` | 0 | 0 | 0 clusters, 0 pixels |

Downstream, end to end with origin/dev's feature order:
- Cluster labels, s_score, as_score and intensity are identical.
- The GMM prior differs by at most 1.7e-15 relative, because it receives the clusters in a different order.
- `hqpr_0_beliefs` differs by at most 4.4e-16.
- `hqpr_0_mask`: 0 pixels differ.
- After MRF refinement, the smoothed mask and smoothed beliefs are identical.

## 2. K-means features: a fixed list instead of `os.listdir`

**origin/dev:** every `*{suffix}.parquet` file in the metrics folder became a k-means feature, in `os.listdir` order.
- That order is set by the filesystem. On overlayfs it is the creation order. On weka it differs between directories, even when the files were created in the same order.
- Any stray matching file silently became an extra feature.

**Now:** `pixel_scoring_dask.PIXEL_FEATURE_NAMES` fixes the features and their column order:
- **hqpr:** intensity, lbp, edge_strength, energy, relevance, entropy, uniformity, homogenity
- **hqtr:** transcript_density, lbp, edge_strength, energy, relevance, entropy, uniformity, homogenity

This is the order structure analysis writes the files in, so it equals origin/dev's order on overlayfs.
A missing file or an unexpected `*{suffix}.parquet` raises an error.

**Difference on the crop:** the weka directory's listdir order was edge_strength, homogenity, intensity, energy, entropy, relevance, lbp, uniformity.

| Output | Difference against that origin/dev run |
| --- | --- |
| Cluster assignment | 103,655 pixels (0.65%) differ, after optimally matching cluster ids |
| `hqpr_0_mask` | 12,123 pixels (0.076%) differ; mask=1 count 3,448,229 → 3,456,198 |
| `hqpr_0_beliefs` | max abs diff 0.62, mean abs diff 9.1e-4 |
| MRF smoothed mask | 4,896 pixels (0.031%) differ |

origin/dev itself can produce a difference of this size between two runs on different directories.

## Memory (full scale, 913M pixels, extrapolated)

These figures are estimates, not measurements.
- **Metric handoff:** structure analysis keeps each metric as float32 in `helperfuncs.PIXEL_FEATURES`, 8 x 3.65 GB = 29 GB.
  - Pixel scoring moves the columns one by one into a column-major matrix, so the handoff and the matrix together peak at about 33 GB, not 58 GB.
  - When clustering will not run in the same process (`-s hqpr_metrices` / `hqtr_metrices`), the subworkflow clears the handoff at the end of structure analysis.
- **Pixel QC:**
  - About 40 GB during clustering: the matrix plus labels and scores.
  - About 35 GB afterwards: the per-pixel output columns.
  - Each full-image `plot_pixels` call adds about 33 GB transiently (measured at 36 B/pixel on the crop).
- **After pixel QC returns:** about 14 GB stays resident, of which 7.3 GB is the beliefs handed to refinement. origin/dev left about 150 GB resident.
- **In-RAM MRF:** adds about 41 GB on top.

The actual full-scale peak is not measured here.
