# Summary of Changes

# 0.1.1

### `Dependencies`
- `pyproject.toml` dependencies are now compatible ranges (`>=tested,<next major`, `<next minor` for 0.x packages) instead of exact `==` pins, so spoQC can be installed next to other packages. The exact tested environment stays pinned in `requirements.txt`.
- `requirements.txt` (used by the Dockerfile) now only pins the runtime dependencies of spoQC. GPU/CUDA, torch, napari/Qt, squidpy and other development-only packages (~4 GB) moved to `requirements-dev.txt`.
- `scikit-misc` is now a declared dependency, it is required by scanpy's `highly_variable_genes(flavor="seurat_v3")` but was only installed by chance.

### `Fixed`
- Markov random field (loopy belief propagation): incoming messages were read from the wrong direction slot (e.g. the upper neighbour's UP message instead of its DOWN message), so a pixel's own message was never excluded. Affects all smoothed HQCR/HQPR/HQTR masks.
- Convexity metric: the closing vertex of shapely exteriors was passed to `_is_convex`, producing two zero cross products and making every polygon non-convex (square scored 0.6, triangle 0.5).
- Doublet score: without detected doublets, `doublet_density` and `doublet_ddd` were set to a 100 000 sentinel that downstream consumers (clustering, visualization) treated as a real "maximum doublet" value. They are now 0 (no doublet signal).
- `map_values_to_cells`: polygon index 0 collided with the rasterize background value (`fill=0`), so all background pixels were credited to cell 0. Polygons are now burned as `index + 1` and only covered pixels are kept.
- Celltype image analysis: intensities and scores were flipped a second time (already flipped in `pixel_score.py`), so per-cell intensity/score means were taken from the vertically mirrored location.
- HQTR pixel clustering: the `*hqtr.parquet` glob also picked up `transcript_density_output_hqtr.parquet` as a 7th KMeans feature. Clustering now uses the explicit list of the 6 image metrics.
- Gene/transcript count priors: with fewer than 3 Leiden clusters a quality level was empty, its NaN mean went into `argmin` and every cell got probability 1 (prior silently disabled). They now fall back to the hard count threshold. The priors also work on local copies of the shared `cell_clustering_df`/`cell_clustering_adata` (only `qc_cluster_str` is still written back for the HQCR report).
- `helperfuncs.min_max_normalize`: constant input (e.g., all negative probe counts 0) divided by zero and produced all-NaN priors. It now returns 1.0 for every value.
- Pixel relevance (HQPR/HQTR): `background_intensity` was passed as the threshold to `cv2.threshold` together with `THRESH_OTSU`, which ignores it. Otsu's threshold is now computed first and the background intensity is used as an explicit lower bound.
- Final report: `analysis/overview` figures were loaded unconditionally, so `--step all` without `--annotation_file` raised `FileNotFoundError`. These sections are now skipped (with a note in the report) when the cell type analysis did not run.
- Celltype image analysis: the percentage barplot used a hard-coded `"celltype"` column instead of `annotation_key`, failing for annotations with a different key.
- `hqcr_celltype.py`: the threshold log was reopened (truncated) for every QC metric and closed outside the loop (`UnboundLocalError` if no count metric was present); nucleus-free cells were looked up as `'nucleusfree'` instead of `'nucleus_free'`, so they were never used for the left threshold; chained assignment for `qc_celltype_class` was a no-op under Copy-on-Write.
- Valid nucleus geometries: with fewer nuclei than cells the nucleus `RangeIndex` was used as cell id (flags landed on the wrong cells), and unassigned nuclei (`cell_id -1`) raised a `KeyError`. `correct_indexing` now keeps the nucleus' cell as `cell_id` column, which is used for the mapping.
- HQR metadata in `adata.uns` (cluster analysis): `uns` was overwritten with the metadata dict, wiping `pca`/`umap`/colors; it is now updated instead. The metadata files were also looked up under `analysis/overview/` instead of the output root, and the HQCR file as `hqcrs.json` instead of `hqcr.json`, so no metadata was ever found.
- Markov random field: the `lbp_messages_*.mmap` file (~32 B/pixel, tens of GB for whole slides) was never deleted. It is now removed in a `finally` block after the beliefs are computed, also when the LBP fails.
- HQTR transcript density cache never hit: it was looked up as `transcript_density_output_hqtr` but written as `transcript_density_output_hqtr.parquet`, so the density image was recomputed on every run. The cache is now used unless `--overwrite` is set or its size does not match the current image.
- Local Moran's I (HQTR ambient contamination) was computed on raw counts (`X`), while the global Moran's I uses `normlog`. Spatially varying library sizes inflated the local autocorrelation. It now uses the `normlog` layer as well.
- HQCR prior combination: a NaN prior turned the aggregated quality of the cell into NaN and counted as "green" in the traffic light (`NaN < t` is always False). NaN priors are now treated as missing evidence and ignored (with a warning per prior); cells without any valid prior get 0.5 and "yellow".
- Raw counts were not copied properly `spoqc/core/starship.py:63 adata.layers['raw'] = adata.X`
- Bug for annotation cell types count: `spoqc/metrics/hqcr/doublet_score.py:261 is inverted (== None)`. Fix: is not None.

### `Removed`
- Unused MRF variants `hqr/markov_random_field.py` and `hqr/markov_random_field_zarr.py`

### `Deprecated`
- Changing threshold for negative probes counts to `t=2`.

# 0.1.0

### `Added`
- New "cell traffic light" QC level system, with a dedicated summary panel in the final report and subcluster spatial plots, adding new tmp data `traffic_light_output_hqcr.parquet`
- New HQCR priors: invalid geometry and negative probe counts
- Redesigned transcript count prior, split into a transcript-and-gene-count prior and a cell-type-level transcript-count prior
- Asymmetric evidence aggregation added for HQCR prior combination
- New parameters for the doublet distance prior
    - `--doublet_prior_std` (default: 0.125)
    - `--doublet_prior_mean` (default: None), this is estimated automatically, but can be set by the user
- Funky heatmap now renders the minimum value as a circle marker
- Performance improvements across pixel scoring/clustering, void calculation, global Moran's I, prior combination, and Leiden clustering; increased prior bin size
- The clustering for HQCR now takes all metrics in the `metrics/hqcr` folder into account.
- Metric calculationg for HQPR now takes all metrics in the `metrics/hqpr` folder into account.
- Metric calculationg for HQTR now takes all metrics in the `metrics/hqtr` folder into account.
- Combine priors now takes all metrics in the `priors/` folder into account.
- Adding `rich-click` instead of `argparse`
    - Added parameter: `--pixel_qc_chunk_size`, `--kmeans_sample_size` for that. The default values do not have to be changed.
- New doublet metrics (ddd = doublet density divided by doublet distance) used for doublet prior estimation.
- Adding script for a quick version bump of the tool.
- More comfort functions which will soon replace older functions in helperfuncs: `plot_spatial_density_of_category`, `plot_marker_density_in_spatial_context`, and `plot_marker_boxplot`.

### `Fixed`
- `combine_priors`: replaced min/max weighting with an absolute average weighted by number of priors
- AC (ambient contamination) image prior: fixed missing absolute-value calculation
- Fixed two rendering bugs in the funky heatmap
- Fixed a bug in HQCR combination logic
- Added edge-case guards and bugfixes in `process_datasets.py`, `cluster_analysis.py`, `final_report.py`, and `helperfuncs.py` (e.g. empty-category and missing-second-page handling)
- Adding `n_unique = len(np.unique(pixel_scores))` to pixel_score for hqpr and hqtr for small datasets.
- Fixed a potential unsync of sdata and annotation in `cli.py`

### `Dependencies`
- No dependency changes in this cycle

### `Deprecated`
- Removed the unused `hqtr_memopt.py` module (superseded HQTR memory-optimization path)
- BIG REDESIGN: spoqc is now object oriented! Metrics and prior can now esiear implemented.
- Changed names in `metrics` to respective HQR.
- `subworkflows` are now called `missions`.
- remove `subworkflows/qc_ambient` as it was merged with metrics calculations
- remove `bubbleqc` as an extra step, it is now part of `cellqc`
- Renameding output tmp files:
    - `hqcr_output_mask_raw.parquet` to `mask_raw_output_hqcr.paruqet`
    - `mask_smoothed_raw_output_hqcr` to `mask_smoothed_raw_output_hqcr`
    - `hqcr_output_mask_smoothed_celltype_refined` to `mask_smoothed_celltype_refined_output_hqcr`
    - `hqcr_output_mask_celltype_refined` to `mask_celltype_refined_output_hqcr`
    - `hqpr_{channel}_output_mask_raw` to `mask_raw_output_hqpr_{channel}`
    - `hqpr_{channel}_output_mask_raw` to `mask_smoothed_raw_output_hqpr_{channel}`
    - `hqtr_output_qv_prob` to `qv_density_output_hqtr`
    - `hqtr_output_ac_prob` to `ac_density_output_hqtr`
    - `hqtr_output_mask_raw` to `mask_raw_output_hqtr`
    - `hqtr_output_mask_smoothed_raw` to `mask_smoothed_raw_output_hqtr`
- Update to Documentation

# 0.0.1

Initial version