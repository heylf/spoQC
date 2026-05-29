# Output reports

* HQCR = High quality cell region
* HQPR = High quality pixel region
* HQTR = High quality transcript region
* LQCR = Low quality cell region
* LQPR = Low quality pixel region
* LQTR = Low quality transcript region

Here it will be explained what the folder `report/` holds.

## Data

### staining_log
`report/` holds a file named `staining_log.txt`.
Currently spoQC analysis all image channels and uses and integer index because channel names can be very arbitrary in their naming.
Therefore, some of the folder and files for the analysis of pixel quality (HQPR) will have integers standing for the individual image channels.
To know which integer belongs to which channel, please look in to `staining_log.txt`.

### Anndata: rna_qc_annotated
`report/analsis` holds a file named `rna_qc_annotated.h5ad`.
SpoQC maps QC metrics and masks to cells with a designed function so you can invstigate spoQC metrics per cell.
The file has the following columnn in `anndata.obs`:

Masks
* hqcr_mask = Binary label if a cell was defined as high quality cell (1) else (0).
* hqcr_beliefs = Mean value of all cell segmentation pixel beliefs.
* hqpr_i_mask = Binary label if a cell had a n > t image pixels identified as high quality pixel (1) else (0) using staining channel i. Currently t = 15. 
* hqpr_i_beliefs = Mean value of all image pixel beliefs using staining channel i.
* hqtr_mask = Binary label if a cell had a n > t transcript density spots identified as high quality spots (1) else (0). Currently t = 15. 
* hqtr_beliefs = Mean value of all transcript density beliefs.

Cell metrics
* doublet = If cell was identified by overlpy being involved in a potential doublet event.
* thinness_score = Bubble score.
* convexhull_outside_trnascripts = Number of transcript in the convex hull of the cell not assigned to a cell (uRNAs).
* convexity_cell
* convexity_mean_nuceli = Mean convexity of all nuclei if at least one is present.
* convexity_min_nuceli = Min convexity of all nuclei if at least one is present.
* multi_nuceli = If the cell has more than one nuclei.
* nucleus_free = If the cell is nucelus free.
* nuceli_count = Number of nuclei in the cell.
* border_cell = If the cell is a border cell.
* island_index = Index of the cell island it belongs to.
* island_score = Island score.
* small_islands = If the cell belongs to a small cell island.
* num_low_qc_transcript = Number of low quality transcripts (qv < 20).

Image metrics
* hqpr_i_intensity = Mean pixel intensity for the cell polygon using staining channel i.
* edge_strength_hqpr_i
* energy_hqpr_i
* relevance_hqpr_i
* entropy_hqpr_i
* homogenity_hqpr_i 
* uniformity_hqpr_i

Transcript metrics
* hqtr_qv_density = Mean QV density for the cell polygon.
* hqtr_ac_density = Mean AC density for the cell polygon.
* hqtr_intensity = Mean transcript density for the cell polygon.
* edge_strength_hqtr
* energy_hqtr
* relevance_hqtr
* entropy_hqtr
* homogenity_hqtr
* uniformity_hqtr

### Spatialdata metadata
`rna_qc_annotated.h5ad` also holde the metadata for the HQCRs, HQPRs and HQTRs which is stored under `anndata.uns`. The `hqcr` is a json formatted pandas dataframe of the format:

```
hqcr_df = pd.DataFrame({
    'islands': sdata['table'].obs['island_index'], 
    'cell_region': sdata['table'].obs['cell_region']
})
```

,where `island_index` is the cell island (cell group) and `cell_region` is the category (e.g., hqcr) that spoQC has classified.

---------------

## combine_masks/*/

This folder holds plots for the combination of all binary masks for HQCRs, HQPRs and HQTRs.
The integers in the folder and files stand for the image channel.
Please have a look at [Staining Log](#staining-log) for more explanation.

### imageplot_final

<img src="./_static/figures/tutorial/combine_masks/0/imageplot_final.png" width="800">

Spatial overlap of all binary masks.
So far only the raw masks are used, without taking celltype information info account.

Categories:
* no mask: Pixel is covered by 0 masks.
* 1 mask: Pixel is covered by only 1 mask.
* 2 mask: Pixel is covered by 2 masks (does not matter which once).
* all mask: Pixel is covered by all 3 masks.

### imageplot_combined_beliefs

<img src="./_static/figures/tutorial/combine_masks/0/imageplot_combined_beliefs.png" width="800">

Combined belief masks for all three data modalities: HQCR, HQPR, and HQTR. The brighter a pixel is the higher is the probability that the observation (cell, pixel, or transcript density spot) is of good quality.

### imageplot_hqcr, imageplot_hqpr, imageplot_hqtr

<img src="./_static/figures/tutorial/combine_masks/0/imageplot_hqcr.png" width="800">

Binary mask of HQCR, HQPR, or HQTR.

Categories:
* black: Pixel is not part of mask.
* white: Pixel is part of mask.

### imageplot_hqcr_beliefs, imageplot_hqpr_*_beliefs, imageplot_hqtr_beliefs

<img src="./_static/figures/tutorial/combine_masks/0/imageplot_hqcr_beliefs.png" width="800">

Belief mask of HQCR, HQPR, or HQTR. The brighter a pixel is the higher is the probability that the observation (cell, pixel, or transcript density spot) is of good quality.

### venn_combined_masks

<img src="./_static/figures/tutorial/combine_masks/0/venn_combined_masks.png" width="400">

Venndiagram describing the pixel overlaps for HQCR, HQPR, and HQTR.
The title also states the \% of area that is not covered by any mask.

---------------

## analysis/

This folder holds plots that investigates the quality more deeply. You can use these plots to get an initial understanding. We recommend to use the [QC annotated anndata] (#anndata-rna-qc-annotated) so you have full control and can generate your own plots. Feel free to dig around spoQC`s code base to get some of the plots we use and adapt them to your personal need.

### category/

Here we investigate spoQC's metrics on specific artefacts events. Currently we cover:
* border cells
* cells associated with doublet events
* nulceus free cells

### cluster/

In the default mode spoQC's picks the cell type cluster with the highest number of cells. You can provide with `--cluster_celltype` a cell type name to directly analysis a specific cluster or just use the [QC annotated anndata](#anndata-rna-qc-annotated), then you have directly control.

### overview/

Here we spoQC's metrics for the whole dataset.

---------------

## hqcr/

### hqcr_ident/
This folder holds plots for the identification of HQCRs.

#### markov_random_field_calculations

<img src="./_static/figures/tutorial/hqcr/hqcr_ident/markov_random_field_calculations.png" width="600">

Markov refinement of the probability that a pixel is belonging to a HQCR.
First plot on the left shows the probabilites (prior) before is is applied to the Markov refinement.
Second plot (middle) shows a simplified binary version as a comparison for the Markov refinement.
Third plot on the right shows the binary mask as a result from the Markov refinement.

#### hqcr/

<img src="./_static/figures/tutorial/hqcr/hqcr_ident/hqcr/scatterplot_hqcr_1.png" width="400">

Spatial plots of HQCRs.
Each point is a cell.
Many cells with high number of 1's in the binary mask for HQCR.
The folder also holds plots where the region is cropped (`zoomed`).

#### lqcr/

<img src="./_static/figures/tutorial/hqcr/hqcr_ident/lqcr/scatterplot_lqcr_1029.png" width="400">

Spatial plots of LQCRs.
Each point is a cell.
Too many cells with high number of 0's in the binary mask for LQCR.
The folder also holds plots where the region is cropped (`zoomed`).

#### scatterplot_refined_qc_class

<img src="./_static/figures/tutorial/hqcr/hqcr_ident/scatterplot_refined_qc_class.png" width="400">

Spatial plot of all HQCRs.
Each point is a cell.

Categories.
* red: Cell belongs to a HQCR.
* blue: Cell does **not** belong to a HQCR.

#### hqcr_cell_region.html

Several plots to investigate the metrices for cells belonging to HQCRs, LQCR and small HQCR (<= 50 cells).
Individual plots you can find as pngs in the folder as well.

#### hqcr_data.html
Several plots to investigate the metrices for all cells (whole data).
Individual plots you can find as pngs in the folder as well.

---------------

### hqcr_celltype/
This folder holds plots for the celltype investigation in connection to HQCRs.

#### hqcr_celltype.html
Several plots to investigate the HQCR metrices for all cells and all celltypes.
Individual plots you can find as pngs in the folder as well.

#### celltype_qc_analysis.html
Several plots to investigate the HQCR metrices for all cells and all celltypes including the indetification of doublet and nucleus free cells.
Some plots have vertical lines displaying the left and right threshold (celltype refined thresholds) one could apply to filter out bad quality cells.
These thresholds are further optimized with the distribution of doublet and nucleus free cells, if these distribtions could have been estimated (enough cells).
Individual plots you can find as pngs in the folder as well.

#### markov_random_field_calculations

<img src="./_static/figures/tutorial/hqcr/hqcr_celltype/markov_random_field_calculations.png" width="600">

This is similar to the plot as in the folder `hqcr` but this time the celltype refined thresholds were used to further refine the prior used by the Markov refinment process.

#### threshold_log.txt
This file states if the celltype thresholds were adjusted by doublet and/or nucelus free cells.

---------------

## hqpr/
The integers in the folder and files stand for the image channel.
Please have a look at [Staining Log](#staining-log) for more explanation.

### hqpr_bounding_box/
This folder holds plots for cropped HQPRs.

#### imageplot_marked_merged_subfigures

<img src="./_static/figures/tutorial/hqpr/0/hqpr_bounding_box/imageplot_marked_merged_subfigures.png" width="600">

Bounding box (red) of merge subfigures (overlapping) bounding boxes of HQPRs.

#### imageplot_marked_subfigures

<img src="./_static/figures/tutorial/hqpr/0/hqpr_bounding_box/imageplot_marked_subfigures.png" width="600">

Bounding box (red) of identified HQPRs.

#### subfigures/

<img src="./_static/figures/tutorial/hqpr/0/hqpr_bounding_box/subfigures/imageplot_subfigure1.png" width="600">

Images of the cropped HQPRs.

---------------

### hqpr_celltype/
This folder holds plots for celltype investigations of the used HQPR metrices.

#### celltype_qc_analysis.html

Several plots to investigate the HQPR metrices for all cells and all celltypes including the indetification of doublet and nucleus free cells.

---------------

### metrics/

This folder holds individual images of the HQPR metrics.

---------------

### hqpr_refinement/
This folder holds plots for the refinement of HQPRs.

#### markov_random_field_calculations

<img src="./_static/figures/tutorial/hqpr/0/hqpr_refinement/markov_random_field_calculations
.png" width="600">

Markov refinement of the probability that a pixel is belonging to a HQPR.
First plot on the left shows the probabilites (prior) before is is applied to the Markov refinement.
Second plot (middle) shows a simplified binary version as a comparison for the Markov refinement.
Third plot on the right shows the binary mask as a result from the Markov refinement.

---------------

## hqtr/

### hqtr_ac/
This folder holds plots for the investigation of ambient RNA for HQTRs.

#### histogram_transcript_ac

<img src="./_static/figures/tutorial/hqtr/hqtr_ac/histogram_transcript_ac.png" width="400">

This plot is useful to adjust the threshold for the ambient RNA consideration.

#### imageplot_norm_p_ac_density

<img src="./_static/figures/tutorial/hqtr/hqtr_ac/imageplot_norm_p_ac_density.png" width="600">

Spatial plot of the normalized prability that a pixel is **not** part of ambient RNA, i.e., a value of 1.0 means that the pixel is with high probability **not** ambient RNA.

#### imageplot_transcript_global_autocorrelation_density

<img src="./_static/figures/tutorial/hqtr/hqtr_ac/imageplot_transcript_global_autocorrelation_density.png" width="600">

Spatial plot of the global Moran's I density.

#### imageplot_transcript_local_autocorrelation_density

<img src="./_static/figures/tutorial/hqtr/hqtr_ac/imageplot_transcript_local_autocorrelation_density.png" width="600">

Spatial plot of the global Moran's I density.

#### imageplot_transcript_autocorrelation_density

<img src="./_static/figures/tutorial/hqtr/hqtr_ac/imageplot_transcript_autocorrelation_density.png" width="600">

Spatial plot of the combined Moran's I density.

---------------

### hqtr_bounding_box/
This folder holds plots for cropped HQTRs.

#### imageplot_marked_merged_subfigures

<img src="./_static/figures/tutorial/hqtr/hqtr_bounding_box
/imageplot_marked_merged_subfigures.png" width="600">

Bounding box (red) of merge subfigures (overlapping) bounding boxes of HQTRs.

#### imageplot_marked_subfigures

<img src="./_static/figures/tutorial/hqtr/hqtr_bounding_box
/imageplot_marked_subfigures.png" width="600">

Bounding box (red) of identified HQTRs.

#### subdocs/source/_static/figures/

<img src="./_static/figures/tutorial/hqtr/hqtr_bounding_box/subfigures
/imageplot_subfigure1.png" width="600">

Images of the cropped HQTRs.

---------------

### hqtr_celltype/

This folder holds plots for celltype investigations of the used HQTR metrices.

#### celltype_qc_analysis.html
Several plots to investigate the HQTR metrices for all cells and all celltypes including the indetification of doublet and nucleus free cells.

---------------

###  hqtr_clustering/
This folder holds plots for the clustering approach of the HQTRs.

#### imageplot_norm_p_informative_pixel

<img src="./_static/figures/tutorial/hqtr/hqtr_clustering
/imageplot_norm_p_informative_pixel.png" width="600">

Spatial plot of the normalized prability that a pixel is of good quality based on the pixel metrices, i.e., a value of 1.0 means that the pixel contributes a lot of information.
The probability is not refined by the Markov refinement.

#### imageplot_norm_p_informative_pixel_hqtr

<img src="./_static/figures/tutorial/hqtr/hqtr_clustering
/imageplot_norm_p_informative_pixel_hqtr.png" width="600">

Spatial plot of the normalized prability that a pixel is of good quality based on the pixel metrices + QV + AC consideration, i.e., a value of 1.0 means that the pixel contributes a lot of information.
The probability is not refined by the Markov refinement.
Comparing `imageplot_norm_p_informative_pixel` with `imageplot_norm_p_informative_pixel_hqtr` gives you an idea how much the consideration of transcript quality (qv) and detection of ambient RNA (ac) would change the informational content of the pixels.

---------------

### metrics/

This folder holds individual images of the HQPR metrics.

---------------

### hqtr_qv/
This folder holds plots for the investigation of the transcript quality density of HQTRs.
The transcript quality (qv) is a Phred-score based measurement done by the vendor.

#### histogram_transcript_qv

<img src="./_static/figures/tutorial/hqtr/hqtr_qv
/histogram_transcript_qv.png" width="600">

This plot is useful to adjust the threshold for the ambient RNA consideration.

#### imageplot_norm_p_qv_density

<img src="./_static/figures/tutorial/hqtr/hqtr_qv
/imageplot_norm_p_qv_density.png" width="600">

Spatial plot of the normalized prability that a pixel is of good quality, i.e., a value of 1.0 means that the pixel is with high probability of good quality.

---------------

### hqtr_refinement/
This folder holds plots for the refinement of HQTRs.

#### markov_random_field_calculations

<img src="./_static/figures/tutorial/hqtr/hqtr_refinement
/markov_random_field_calculations.png" width="600">

Markov refinement of the probability that a pixel is belonging to a HQTR.
First plot on the left shows the probabilites (prior) before is is applied to the Markov refinement.
Second plot (middle) shows a simplified binary version as a comparison for the Markov refinement.
Third plot on the right shows the binary mask as a result from the Markov refinement.

---------------

## ambientqc/
This folder holds plots for the inspection of ambient RNA.

### contamination_global_morans_I

<img src="./_static/figures/tutorial/ambientqc/contamination_global_morans_I.png" width="400">

Genes sorted by Morans'I.
As ambient RNA is expected to be relatively uniformly distributed, it should produce random spatial patterns that yield low Moran’s I values.

---------------

## cellcycleqc/

This folder holds plot for the investigation of the cellcycle if those genes are present in the features.

### scatterplot_densityplot_phase_1

<img src="./_static/figures/tutorial/cellcycleqc/scatterplot_densityplot_phase_1.png" width="700">

Spatial plot of cell density based on the different cellcycle phases.

### barplot_sample_cellcycle_fractions

<img src="./_static/figures/tutorial/cellcycleqc/barplot_sample_cellcycle_fractions.png" width="400">

Fraction of cells in the different cellcycle phase.

---------------

## cellqc/

### scatterplot_border_cell

<img src="./_static/figures/tutorial/cellqc/scatterplot_border_cell.png" width="400">

Spatial plot showing potential border cells.
A border cell is a cell that lies on the border of the slide or (sub-)tissue.

Cell categories:
- False (blue): cell in the inner tissue.
- True (red): border cell.

### scatterplot_densityplot_num_low_qc_transcript

<img src="./_static/figures/tutorial/cellqc/scatterplot_densityplot_num_low_qc_transcript.png" width="400">

Spatial-density plot of the cells, where the density is weighted by the number of low quality (qc < 20) transcripts.

### scatterplot_island

<img src="./_static/figures/tutorial/cellqc/scatterplot_island.png" width="400">

Spatial plot showing cell forming larger (blue) or smaller cell island (red).

### scatterplot_densityplot_convexity_cell_convexity_metric_cell

<img src="./_static/figures/tutorial/cellqc/scatterplot_densityplot_convexity_cell_convexity_metric_cell.png" width="400">

Spatial-density plot of the cells, where the density is weighted by the convexity of the cell.

Cell categories:
- False (black): cell is not convex (convexity <=0.5) and might have a weird shape.
- True (blue): cell is convex (convexity > 0.5).

### scatterplot_densityplot_convexity_nuclei_convexity_mean_nuceli

<img src="./_static/figures/tutorial/cellqc/scatterplot_densityplot_convexity_nuclei_convexity_mean_nuceli.png" width="400">

Spatial-density plot of the cells, where the density is weighted by the mean convexity of the nuclei.
The mean is taken because a cell might have more than 1 nucleus.

Cell categories:
- False (black): nuclei in the cell are not convex (convexity <=0.5) and might have a weird shapes.
- True (blue): nuclei are convex (convexity > 0.5).

### scatterplot_densityplot_convexity_nuclei_convexity_metric_cell

<img src="./_static/figures/tutorial/cellqc/scatterplot_densityplot_convexity_nuclei_convexity_metric_cell.png" width="400">

Spatial-density plot of the cells, where the density is weighted by the convexity of the cell.
However, the color of the cell is defined by the mean convexity of the nuclei.

Cell categories:
- False (black): nuclei in the cell are not convex (convexity <=0.5) and might have a weird shapes.
- True (blue): nuclei are convex (convexity > 0.5).

### scatterplot_densityplot_nucleus_free

<img src="./_static/figures/tutorial/cellqc/scatterplot_densityplot_nucleus_free.png" width="400">

Spatial-density plot of nucleus-free cells. 

---------------

## doubletqc/
This folder holds plots for the doublet detection with ovrlpy.

### histogram_signal_integrity_and_signal

<img src="./_static/figures/tutorial/doubletqc/histogram_signal_integrity_and_signal.png" width="400">

To fine-tune ovrlpy.
Running ovrlpy requires `integrity_sigma` and `signal_threshold`.

### scatterplot_densityplot_doublet

<img src="./_static/figures/tutorial/doubletqc/scatterplot_densityplot_doublet.png" width="400">

Density map of doublet events in the data.
Each dot is a cell.

Categories:
* False: Cell is **not** associated with a doublet region.
* True: Cell is associated with a doublet region.

### scatter_signal_integrity_3d

<img src="./_static/figures/tutorial/doubletqc/scatter_signal_integrity_3d.png" width="400">

This plot displays each transcript by their x, y, and z coordinates. 
Each z coordinate is highlighted with a different color.

### spatial_signal_integrity_map

<img src="./_static/figures/tutorial/doubletqc/spatial_signal_integrity_map.png" width="800">

The ovrlpy tool works by comparing the transcript landscape between the top and bottom z-levels.
A high VSI value (close to 1.0) indicates strong similarity between these layers, while a lower value (brightes spots), below a defined threshold, suggests potential doublet events, as the top and bottom layers display different transcriptional patterns.

### doublet_case_* and doublet_case_*_zoomed

<img src="./_static/figures/tutorial/doubletqc/doublet_case_0.png" width="800">

The first and second figures show the same doublet region, with the second figure providing a zoomed-in view. 
Each point in each plot (except the signal integrity plot) represents an individual transcript molecule.

Each figure begins with a **UMAP**, colored by the expected cell types from our annotation.
The **celltype map** projects these annotated cell types back into spatial coordinates.
The **signal integrity** plot follows the same format as seen earlier, but is focused on the identified doublet region.
Finally, the **ROI plots** show the transcript landscape at the top and bottom z-levels, as well as slices along the x- and y-axes.

---------------

## generalqc/
This folder holds plots for general overview of the quality of the data.

### rna_qc_sample_mqc.html

Several plots that give you a general overview of the quality of the data based on cell metrices.

### scatterplot_densityplot_invalid_cell_geometry

<img src="./_static/figures/tutorial/generalqc/scatterplot_densityplot_invalid_cell_geometry.png" width="400">

Density map of invalid cell geometries in the data. Each dot is a polygon.

### scatterplot_densityplot_invalid_nucleus_geometry

<img src="./_static/figures/tutorial/generalqc/scatterplot_densityplot_invalid_nucleus_geometry.png" width="400">

Density map of invalid nucleus geometries in the data. Each dot is a polygon.

### invalid_cell_geomtry_*

<img src="./_static/figures/tutorial/generalqc/invalid_cell_geomtry_0.png" width="200">

Example of invalid cell geometries.

### invalid_nucleus_geomtry_*

<img src="./_static/figures/tutorial/generalqc/invalid_nucleus_geomtry_0.png" width="200">

Example of invalid nucleus geometries.

---------------

## voidqc/
This folder holds plots for the inspection of regions outside cells, which we call *void*.
A lot of plots uses Delauny triangulation and thus display spatial plots with triangles, where each triangle holds information.
A *void* is a triangle cluster, i.e., triangles that form a bigger area.

### spatial_traingle_all_clsuters_log10_transcripts_counts_outside_cell

<img src="./_static/figures/tutorial/voidqc/spatial_traingle_all_clsuters_log10_transcripts_counts_outside_cell.png" width="400">

The darker the area the more transcript a triangle cluster (void) holds.
This helps you to figure out if some voids are still lots of information that could be used.