# Run
Executing spoQC pipeline via python (sequential):

SpoQC needs an HPC infrastructure to perform all task on a full SRT datset with full resolution. You might be able to perform spoQC locally with a lower resolution or with subsetting your data.

## Annotation

Optional step if you do not have a cell type annotation yet, then spoQC can do an analysis using an unsupervised (Leiden) clustering.

```
python3 -m spoqc -s "annotation" -i [input_spatial_data_bundle] -o [output_folder] -t [spoqc_tmp_folder] -n [n_cores]
```

This will generate you an annotation file in spoQC format `[spoqc_tmp_folder]/report/annotation/unsupervised_cell_annotation.tsv`

## Execute everything in spoQC

You can execute spoQC completly with:

```
python3 -m spoqc -s all -i [input_spatial_data_bundle] -o [output_folder] -t [spoqc_tmp_folder] -n [n_cores] -a [annotation_file]
```


## Individual step execution

You can execute spoQC for each step individually with:

```
python3 -m spoqc -s [step] -i [input_spatial_data_bundle] -o [output_folder] -t [spoqc_tmp_folder] -n [n_cores] -a [annotation_file]
```

with [step] in the following order (if you do not follow this order things will break):

* generalqc
* bubbleqc
* doubletqc
* voidqc
* cellqc
* ambientqc
* hqcr_ident
* hqcr_celltype
* hqpr_metrices
* hqpr_clustering
* hqpr_clustering
* hqpr_refinement
* hqpr_bounding_box
* hqpr_celltype
* hqtr_metrices
* hqtr_ac
* hqtr_qv
* hqtr_clustering
* hqtr_refinement
* hqtr_bounding_box
* hqtr_celltype
* combine_masks
* transcriptqc
* modelqc
* cellcycleqc
* analysis_overview
* analysis_cluster
* analysis_category

For example for the first step you execute the command:

```
python3 -m spoqc -s generalqc -i [input_spatial_data_bundle] -o [output_folder] -t [spoqc_tmp_folder] -n [n_cores] -a [annotation_file]
```