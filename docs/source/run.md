# Run

spoQC is designed to process large spatial transcriptomics (SRT) datasets at full resolution. Running the complete pipeline typically requires access to an HPC (High Performance Computing) environment.

If you do not have access to an HPC system, you may still be able to run spoQC locally by:

- Using a lower-resolution dataset.
- Running spoQC on a subset of your data.
- Testing individual pipeline steps before processing the full dataset.

## Step 1: Generate a Cell Type Annotation (Optional)

If your dataset does not already contain a cell type annotation, spoQC can create one automatically using unsupervised Leiden clustering.

Run:

```bash
python3 -m spoqc -s "annotation" -i [input_spatial_data_bundle] -o [output_folder] -t [spoqc_tmp_folder] -n [n_cores]
```

After the analysis finishes, spoQC will create an annotation file:

```text
[output_folder]/report/annotation/unsupervised_cell_annotation.tsv
```

You can use this file as the value for the `[annotation_file]` parameter in later steps.

---

## Step 2: Run the Complete spoQC Pipeline

To execute all spoQC analyses in the correct order, run:

```bash
python3 -m spoqc -s all -i [input_spatial_data_bundle] -o [output_folder] -t [spoqc_tmp_folder] -n [n_cores] -a [annotation_file]
```

This is the recommended option for most users.

---

## Options

spoQC uses a [rich-click](https://github.com/ewels/rich-click) command line interface. To list all options with their defaults, run:

```bash
python3 -m spoqc --help
```

### Required

| Option | Description |
|---|---|
| `-i`, `--input_file` | Path to the input SpatialData bundle (see [Input](input.md)). |
| `-o`, `--output_dir` | Output directory. spoQC writes all results to `[output_dir]/report/`. |
| `-t`, `--tmp_dir` | Directory for temporary files (see [Output temporary data files](output_tmp.md)). |
| `-s`, `--step` | Step to run (default: `all`). |
| `-d`, `--datatype` | Type of the input data (default: `xenium`). Currently only `xenium` is supported. |

### Optional

| Option | Default | Description |
|---|---|---|
| `-a`, `--annotation_file` | – | Cell type annotation file (see [Input](input.md)). Required for the `analysis_*` steps. |
| `-n`, `--nthreads` | `1` | Number of cores. |
| `--staining` | `0` | Integer index of the staining used for the HQPR steps. The mapping of indices to staining names is written to `report/staining_log.txt`. |
| `--cellcycle_gene_file` | – | JSON file with `"S"` and `"G2M"` keys listing the S-phase and G2M-phase genes. |
| `--cluster_celltype` | – | Name of a cell type you want to analyse specifically. |
| `--reference_file` | – | Transcript reference file for the transcript QC. |
| `--overwrite` | off | Overwrite existing temporary files. |
| `--doublet_prior_std` | `0.125` | Standard deviation of the doublet prior. Increasing it increases the impact of doublet events, i.e., doublet events affect more cells and give them a lower quality. |
| `--doublet_prior_mean` | estimated | Mean of the doublet prior. By default it is estimated automatically from the data, but it can be set manually. |
| `--thresh_prior_pixel` | estimated | Threshold for the pixel score prior distribution (HQPR/HQTR). Only set this if you understand the pixel prior (see [Image metrics](metrics_image_layer.md)). |
| `--nstds_prior_pixel` | `6` | Number of standard deviations for the pixel score prior distribution (HQPR/HQTR). |
| `--kmeans_sample_size` | `5000000` | Number of randomly subsampled pixels used to fit the MiniBatchKMeans model of the pixel clustering (HQPR/HQTR). All pixels are then labeled in parallel with the fitted model. |
| `--chunk_size` | `200000` | Row-chunk size of the dask arrays/dataframes for the HQPR/HQTR clustering and scoring. Larger values reduce dask overhead but increase the peak memory per chunk. |

```{note}
The defaults of `--kmeans_sample_size` and `--chunk_size` usually do not have to be changed.
```

---

## (Run Individual Pipeline Steps)

Advanced users can execute individual spoQC steps separately.

Run:

```bash
python3 -m spoqc -s [step] -i [input_spatial_data_bundle] -o [output_folder] -t [spoqc_tmp_folder] -n [n_cores] -a [annotation_file]
```

Replace `[step]` with one of the following pipeline stages.

> **Important:** These steps must be executed in the exact order shown below. Running steps out of order will cause downstream analyses to fail.

| # | Step | What it does |
|---|---|---|
| 1 | generalqc | General single cell metrics (transcript, gene and control probe counts) and the check for invalid cell/nucleus geometries. |
| 2 | doubletqc | Doublet detection with [ovrl.py](https://github.com/HiDiHlabs/ovrl.py) and the cell overlap area. |
| 3 | voidqc | Unassigned transcripts (uRNAs) within the convex hull of each cell. |
| 4 | cellqc | Cell shape and neighbourhood metrics: border score, thinness (bubble) score, convexity, island score, nuclei count and low quality transcripts. |
| 5 | ambientqc | Global Moran's *I* per gene, needed for the ambient RNA (AC) image in `hqtr_ac`. |
| 6 | hqcr_ident | Calculates and combines the HQCR priors (incl. the cell traffic light system) and identifies HQCRs. |
| 7 | hqcr_celltype | Cell type refinement of the HQCR mask. |
| 8 | hqpr_metrices | Image metrics for one staining (has to be run for each staining). |
| 9 | hqpr_clustering | Pixel clustering and HQPR priors (has to be run for each staining). |
| 10 | hqpr_refinement | Markov random field smoothing of the HQPR mask (has to be run for each staining). |
| 11 | hqpr_bounding_box | HQPR bounding boxes (has to be run for each staining). |
| 12 | hqpr_celltype | Cell type analysis of the HQPR (has to be run for each staining). |
| 13 | hqtr_metrices | Transcript density image and its image metrics. |
| 14 | hqtr_ac | Ambient RNA (autocorrelation) image. |
| 15 | hqtr_qv | Transcript quality (QV) image. |
| 16 | hqtr_clustering | Pixel clustering and HQTR priors. |
| 17 | hqtr_refinement | Markov random field smoothing of the HQTR mask. |
| 18 | hqtr_bounding_box | HQTR bounding boxes. |
| 19 | hqtr_celltype | Cell type analysis of the HQTR. |
| 20 | combine_masks | Combines the HQCR, HQPR and HQTR masks (has to be run for each staining). |
| 21 | transcriptqc | Transcript QC (optionally against `--reference_file`). |
| 22 | modelqc | QC model analysis. |
| 23 | cellcycleqc | Cell cycle QC. |
| 24 | whole_slide_qc | Whole slide QC plots. |
| 25 | analysis_overview | Overview analysis (requires `-a`). |
| 26 | analysis_cluster | Subcluster analysis (requires `-a`). |
| 27 | analysis_category | Category analysis (requires `-a`). |
| 28 | final_report | Builds `report.html`. |

The following steps are not part of `all` and can be run additionally:

- `annotation`: unsupervised cell type annotation (see Step 1).
- `markerqc`: marker gene QC.
- `combine_masks_zoom`: zoomed-in plots of the combined masks.

### Example

To run the first pipeline step (`generalqc`), execute:

```bash
python3 -m spoqc -s generalqc -i [input_spatial_data_bundle] -o [output_folder] -t [spoqc_tmp_folder] -n [n_cores] -a [annotation_file]
```

Wait until the step has completed successfully before continuing with the next step in the list.