# Summary of Changes

# 0.0.2

### `Added`
- New "cell traffic light" QC level system, with a dedicated summary panel in the final report and subcluster spatial plots
- New HQCR priors: invalid geometry and negative probe counts
- Redesigned transcript count prior, split into a transcript-and-gene-count prior and a cell-type-level transcript-count prior
- Asymmetric evidence aggregation added for HQCR prior combination
- New parameters for the doublet distance prior
    - `--doublet_prior_std` (default: 0.125)
    - `--doublet_prior_mean` (default: None), this is estimated automatically, but can be set by the user
- Funky heatmap now renders the minimum value as a circle marker
- Performance improvements across pixel scoring/clustering, void calculation, global Moran's I, prior combination, and Leiden clustering; increased prior bin size
    - Added parameter: `--pixel_qc_chunk_size`, `--kmeans_sample_size` for that. The default values do not have to be changed.
- New doublet metrics (ddd = doublet density divided by doublet distance) used for doublet prior estimation.
- Adding script for a quick version bump of the tool.

### `Fixed`
- `combine_priors`: replaced min/max weighting with an absolute average weighted by number of priors
- AC (ambient contamination) image prior: fixed missing absolute-value calculation
- Fixed two rendering bugs in the funky heatmap
- Fixed a bug in HQCR combination logic
- Added edge-case guards and bugfixes in `process_datasets.py`, `cluster_analysis.py`, `final_report.py`, and `helperfuncs.py` (e.g. empty-category and missing-second-page handling)
- Fixed a potential unsync of sdata and annotation in `cli.py`

### `Dependencies`
- No dependency changes in this cycle

### `Deprecated`
- Removed the unused `hqtr_memopt.py` module (superseded HQTR memory-optimization path)


# 0.0.1

Initial version