# Output data

SpoQC's output data that is useful.

## staining_log
`report/` holds a file named `staining_log.txt`.
Currently spoQC analysis all image channels and uses and integer index because channel names can be very arbitrary in their naming.
Therefore, some of the folder and files for the analysis of pixel quality (HQPR) will have integers standing for the individual image channels.
To know which integer belongs to which channel, please look in to `staining_log.txt`.

## Anndata: rna_qc_annotated
`report/analsis` holds a file named `rna_qc_annotated.h5ad`.
SpoQC maps QC metrics and masks to cells with a designed function so you can invstigate spoQC metrics per cell.
The file has the following columnn in `anndata.obs`:

### Masks

For each modality the pixel-level values are aggregated to the cell polygon in two variants: the raw output (`hqcr_*`, `hqpr_i_*`, `hqtr_*`) and a Markov-smoothed version (`*_smoothed`). `_mask` columns are a binary label derived by summing the pixel mask over the cell polygon and thresholding at t = 15 pixels; `_mask_mean` is the fraction of the cell's pixels that are labeled positive (i.e. the mean of the binary mask).

| Column | Description |
|---|---|
| hqcr_mask / hqcr_mask_smoothed | Binary label if a cell was defined as a high quality cell (1) else (0), using the raw / Markov-smoothed belief mask. |
| hqcr_mask_mean / hqcr_mask_mean_smoothed | Fraction of the cell's pixels labeled as HQCR. |
| hqcr_beliefs / hqcr_beliefs_smoothed | Mean value of all cell segmentation pixel beliefs over the cell polygon (raw / smoothed), including zero-belief pixels. |
| hqpr_i_mask / hqpr_i_mask_smoothed | Binary label if a cell had n > t image pixels identified as high quality pixel (1) else (0) using staining channel i. Currently t = 15. |
| hqpr_i_mask_mean / hqpr_i_mask_mean_smoothed | Fraction of the cell's pixels labeled as HQPR for staining channel i. |
| hqpr_i_beliefs / hqpr_i_beliefs_smoothed | Mean value of all image pixel beliefs using staining channel i, excluding zero-belief pixels. |
| hqpr_i_beliefs_mean_informative / hqpr_i_beliefs_mean_informative_smoothed | Mean image pixel belief using staining channel i, considering only "informative" pixels (belief > 0.2). |
| hqtr_mask / hqtr_mask_smoothed | Binary label if a cell had n > t transcript density spots identified as high quality spots (1) else (0). Currently t = 15. |
| hqtr_mask_mean / hqtr_mask_mean_smoothed | Fraction of the cell's pixels labeled as HQTR. |
| hqtr_beliefs / hqtr_beliefs_smoothed | Mean value of all transcript density beliefs, excluding zero-belief pixels. |
| hqtr_beliefs_mean_informative / hqtr_beliefs_mean_informative_smoothed | Mean transcript density belief, considering only "informative" pixels (belief > 0.2). |

### Cell metrics

| Column | Description |
|---|---|
| transcript_counts (or canorm_transcript_counts if canorm normalization was used) | Number of transcripts assigned to the cell. |
| control_probe_counts | Number of control probe counts assigned to the cell. |
| n_genes_by_counts | Number of genes detected in the cell. |
| doublet | If cell was identified by ovrlpy being involved in a potential doublet event. |
| thinness_score | Bubble score. |
| convexhull_outside_trnascripts | Number of transcript in the convex hull of the cell not assigned to a cell (uRNAs). |
| convexity_cell | Binary label if the cell is convex (convexity_metric_cell > 0.5). |
| convexity_metric_cell | Convexity score of the cell. |
| convexity_mean_nuceli | Mean convexity of all nuclei if at least one is present. |
| convexity_min_nuceli | Min convexity of all nuclei if at least one is present. |
| multi_nuceli | If the cell has more than one nuclei. |
| nucleus_free | If the cell is nucelus free. |
| nuceli_count | Number of nuclei in the cell. |
| border_cell | If the cell is a border cell (border_scores above threshold). |
| border_scores | Border score of the cell, used to derive border_cell. |
| island_index | Index of the cell island it belongs to. |
| island_score | Size (number of cells) of the cell island it belongs to. |
| small_islands | If the cell belongs to a small cell island (island_score below threshold). |
| num_low_qc_transcript | Number of low quality transcripts (qv < 20). |
| cell_overlap_area | Overlap area of the cell with neighbouring cells. |

### Image metrics

| Column | Description |
|---|---|
| hqpr_i_intensity | Mean pixel intensity for the cell polygon using staining channel i. |
| edge_strength_hqpr_i, energy_hqpr_i, relevance_hqpr_i, entropy_hqpr_i, homogenity_hqpr_i, uniformity_hqpr_i | Mean of the respective pixel (anti)structure/texture metric over the cell polygon, using staining channel i. |

### Transcript metrics

| Column | Description |
|---|---|
| hqtr_intensity | Mean transcript density for the cell polygon. |
| hqtr_qv_density | Mean transcript quality (QV) density for the cell polygon. |
| hqtr_ac_density | Mean ambient RNA (AC) density for the cell polygon. |
| edge_strength_hqtr, energy_hqtr, relevance_hqtr, entropy_hqtr, homogenity_hqtr, uniformity_hqtr | Mean of the respective pixel (anti)structure/texture metric over the cell polygon, computed on the transcript density image. |

### Columns added during additional analysis

| Column | Description |
|---|---|
| leiden | Leiden cluster assignment computed on the (sub-)dataset (resolution auto-selected). |
| hqcr_filtered_out, hqtr_filtered_out, hqpr_i_filtered_out | Binary label if the cell's belief score for that modality is below the 0.45 filtering threshold. |
| hqr_filtered_out | Binary label if the cell was filtered out by any of hqcr, hqtr or hqpr (OR of the columns above). |

## Anndata: rna_cluster
`report/analsis` holds a file named `rna_cluster.h5ad`. This anndata is from the subcluster analysis and holds the same columns as `rna_qc_annotated.h5ad`, but only for the cells of the analysed cluster. In addition, `anndata.uns` holds `spoqc_celltype_colors` and `spoqc_leiden_colors`, the label-to-color mappings used for the analysis plots.

## Spatialdata metadata
`rna_qc_annotated.h5ad` also holde the metadata for the HQCRs, HQPRs and HQTRs which is stored under `anndata.uns`. 

The `hqcr` is a json formatted pandas dataframe of the format:
```
hqcr_df = pd.DataFrame({
    'islands': sdata['table'].obs['island_index'], 
    'cell_region': sdata['table'].obs['cell_region']
})
```
,where `island_index` is the cell island (cell group) and `cell_region` is the category (e.g., hqcr) that spoQC has classified.


The `hqpr` and `hqtr` are a numpy array of the form:
```
[[10525.0, 10561.0, 10808.0, 11007.0], [11272.0, 10587.0, 11579.0, 10915.0], [12578.0, 10784.0, 13000.0, 11465.0]]
```
, where each entry `hqpr[i]` or `hqtr[i]` is a HQPR or HQTR, repsectively.

Currenlty you can find the individual metadata files also under:
* `report/hqcr_ident/hqcr.json`
* `report/hqpr/hqtr_bounding_box/*/hqpr.txt`
* `report/hqtr/hqtr_bounding_box/hqtr.txt`