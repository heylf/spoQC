# ambientqc and hqtr transcript images: changes from origin/dev

Everything in `subworkflows/qc_ambient.py` and the hqtr-specific steps (transcript density, qv and ac images, global and local Moran's I, the qv/ac priors) gives the same values as origin/dev db00d98, bit for bit, except for the changes below.
Measurements are on a real 4000 x 4000 crop of `breast2.zarr` (x 16000-20000, y 10000-14000: 16,000,000 pixels, 547,140 transcripts, 2,222 cells), `-n 4`, with the tmp folder on NFS.
Full-slide numbers (912,950,048 pixels) are linear extrapolations, x57.06.

## 1. Prior parquet layout: 1,000,000-row parts, only the columns that are read

The qv and ac steps write `{tmp}/hqtr_output_qv_prob` and `{tmp}/hqtr_output_ac_prob`.

**origin/dev:** dask part files of 10,000 rows, each with three float64 columns: `{x}_density`, `d_{x}_density` and `norm_p_{x}_density`.

**Now:** part files of 1,000,000 rows (`priors.hqtr.ac_or_qv.PART_ROWS`), with two columns: `{x}_density` and `norm_p_{x}_density`.
The index is unchanged (int64 pixel position, `__null_dask_index__`), and so are the values.

Readers (all found by grep over `spoqc/` and `tests/`):

| Reader | Columns it reads | Change |
| --- | --- | --- |
| `priors/combine_priors.combine_priors_hqtr` (hqtr clustering) | `norm_p_{qv,ac}_density` | reads any layout; no longer returns the prior divisions |
| `additional_analysis/analysis_funcs` (with an annotation file) | `qv_density`, `ac_density` | none (`dd.read_parquet(columns=...)`) |
| `unittests/test_all.py` (`-s unittest` against stored reference data) | whole directory, row-hash sum | its reference data must be regenerated: `d_*` is gone |

`d_{x}_density` had no reader, so it is no longer written.

| Measure (crop) | origin/dev | now |
| --- | --- | --- |
| qv files, size | 1,600 files, 133.6 MB | 16 files, 95.0 MB |
| ac files, size | 1,600 files, 146.4 MB | 16 files, 105.4 MB |
| Write time per prior (dask, 2 runs) | 6.8-9.1 s | 0.31-0.35 s (write_parts, 3.3-3.6 cores) |
| Same layout through write_parts | 2.5-5.0 s (NFS latency, 0.5-1.0 cores) | |
| Read time of `norm_p` (random data, 16 M rows) | 2.8-3.1 s | 0.22-0.24 s |

Full slide (extrapolated): 91,295 files and ~7.6-8.4 GB per prior become 913 files and ~5.4-6.0 GB; the dask write of ~390-520 s per prior becomes ~18-20 s.

Tests: `tests/test_hqtr_ambient_differential.py::TestPrior` (values, columns and divisions against origin/dev's prior), and the real crop in `tests/realdata_hqtr_ambient.py`.

### hqtr/hqpr mask directories

`mask_raw` and `mask_smoothed_raw` kept origin/dev's 10,000-row layout here (91,296 files per directory at full slide).
They are now written in `core.parquet.PART_ROWS`-row parts and decoded in parallel: see `docs/perf/mask_layout.md`.

## Writers: one implementation

`core/parquet.write_parts` is spoQC's only writer of these per-pixel parquet directories: the qv/ac priors, `mask_raw` (`pixel_scoring_dask`) and `mask_smoothed_raw` (`pixel_scoring_refinement`).
For a given part layout it writes the bytes dask's `to_parquet` wrote, including NaN written as null.
`helperfuncs.ddf_to_parquet` is deleted; its verbatim copy in `tests/legacy/parquet_writer.py` is the reference in the tests.

## 2. One Gaussian prior: `priors/gaussian.py`

**origin/dev:** four copies of "score values by a normal density, then min-max scale":

| Copy | Density | Tail / peak | Min-max |
| --- | --- | --- | --- |
| `priors/hqcr/negative_probe_counts.calc_probs` (per cell) | `scipy.stats.norm.pdf` | right tail set to `np.max(pdf)`, then `np.max(pdf) - pdf` | `(x - min) / (max - min)` |
| `priors/hqcr/doublet_distance.calc_probs_doublet_distance` (per cell) | `norm.pdf` | none | `(x - min) / (max - min)`, skipped when there are no doublets |
| `priors/hqpr/pixel_score.calc_probs_pixel_score` (hqpr and hqtr, per pixel cluster) | `norm.pdf` | none | dask_ml MinMaxScaler over the pixels: `x * (1 / range) + (0 - min / range)` |
| `priors/hqtr/ac_or_qv.calc_prob_pixel_stuff_v2` (qv/ac, per pixel) | by hand: `inv_std / sqrt(2 pi) * exp(-0.5 z^2)` | left tail set to that constant, then constant `- pdf` | dask_ml MinMaxScaler |

The GMM fit of the negative-probe and pixel-score copies was the same code twice.

**Now:** `priors/gaussian.py` holds the GMM parameter choice (`gmm_parameters`) and the density (`gaussian_density`), and `helperfuncs.min_max_normalize` is the one min-max (threaded, `out=` for in place). All four callers use them. `pixel_scoring_dask.min_max_normalize` and every `dask_ml` import are deleted.

The formulation, the most correct of the four:
- **Density:** `scipy.stats.norm.pdf`, the library implementation, rather than the formula typed out by hand.
- **Peak:** `norm.pdf(mean)`, the density's true maximum. `np.max(pdf)` over the values is below the peak whenever no value sits exactly at the mean. For the negative probes it then made the tail cells and the cells as far below the mean equally "worst": with no cell at exactly t = 1 probe, every prior was 0 and min-max gave 0 / 0 = NaN for every cell (`tests/test_gaussian_prior.py`).
- **Min-max:** `(x - min) / (max - min)`: one rounding, and exactly 0 and 1 at the extremes. MinMaxScaler's `x * (1 / range) + (0 - min / range)` has three roundings. A zero range now scales to 0 (MinMaxScaler's rule) instead of 0 / 0 = NaN, and NaN values are skipped.
- **GMM:** the mixture is still fitted where origin/dev fitted it, even when `t` and `std` override its result (negative probes). origin/dev's fits drew their k-means initialisation from numpy's global random state; every fit now takes `random_state=seed`, the run's seed, so it neither reads nor advances the global state. With origin/dev's global state seeded to the same seed, the pixel-score parameters are identical (tested).

**Difference on the crop**, on the real inputs of a run of origin/dev-equivalent code, each prior computed both ways:

| Prior | Values differing | Max abs diff | Max rel diff |
| --- | --- | --- | --- |
| Negative probes (A, cells) | 0 of 2,222 (325 cells have exactly t = 1 probe, so `np.max(pdf)` was the true peak) | 0 | 0 |
| Doublet distance (D, cells) | 0 of 2,222 | 0 | 0 |
| hqpr pixel score, `norm_p` (B, pixels) | 8,700,228 of 16,000,000 | 1.1e-16 | 2.9e-16 |
| hqtr pixel score, `norm_p` (B, pixels) | 6,352,763 of 16,000,000 | 1.1e-16 | 3.1e-16 |
| qv prior, `norm_p_qv_density` (C, pixels) | 355,579 of 16,000,000 | 3.3e-16 | 2.6e-4 (at values near 1e-12) |
| ac prior, `norm_p_ac_density` (C, pixels) | 430,040 of 16,000,000 | 2.2e-16 | 7.6e-6 |

**Downstream, end to end:** the crop was run step by step through the CLI (bubbleqc, doubletqc, voidqc, cellqc, generalqc, hqcr_ident, hqpr_metrices .. hqpr_bounding_box, ambientqc, hqtr_metrices .. hqtr_bounding_box) three times: twice with origin/dev-equivalent code and once with these changes.
The two origin/dev-equivalent runs gave identical outputs, all 95 parquet columns, so every difference below comes from these changes.

| Output | Differs from origin/dev |
| --- | --- |
| GMM thresholds (mean, std) of the hqpr and hqtr pixel priors and of the negative probes | identical |
| hqcr: `hqcr_beliefs`, `hqcr_mask`, smoothed beliefs and mask, `hqcr_traffic_light`, `hqcr.json` | identical |
| `hqpr_0_beliefs` (mask_raw and mask_smoothed_raw) | 8,700,228 pixels, max abs 1.1e-16 |
| `hqpr_0_mask` | 0 pixels |
| hqtr `norm_p_pixel_score` (mask_raw) | 6,352,763 pixels, max abs 1.1e-16 |
| `norm_p_qv_density`, `norm_p_ac_density` (prior parquets) | 355,579 and 430,040 pixels, max abs 3.3e-16 |
| `hqtr_beliefs` (mask_raw and mask_smoothed_raw) | 4,516,210 pixels, max abs 2.2e-16 |
| `hqtr_mask`, `pixel_score_mask` | 0 pixels |
| MRF-refined `*_beliefs_smoothed`, `*_mask_smoothed` (hqpr and hqtr) | 0 pixels |
| Bounding boxes (`hqprs_0.txt`, `hqtrs.txt`) | identical |
| Every other parquet column (88 of 95) | identical |

Tests: `tests/test_gaussian_prior.py` (each copy vs its verbatim origin/dev module in `tests/legacy/`), `tests/test_pixel_scoring_differential.py` (mask_raw vs the origin/dev pipeline: only the prior-derived columns differ, within 2 ULP of 1, and masks flip only at beliefs within 2 ULP of 0.5).
