# Summary of Changes

# 0.0.2

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

# 0.0.1

Initial version