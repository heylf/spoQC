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
[spoqc_tmp_folder]/report/annotation/unsupervised_cell_annotation.tsv
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

## Step 3: Run Individual Pipeline Steps

Advanced users can execute individual spoQC steps separately.

Run:

```bash
python3 -m spoqc -s [step] -i [input_spatial_data_bundle] -o [output_folder] -t [spoqc_tmp_folder] -n [n_cores] -a [annotation_file]
```

Replace `[step]` with one of the following pipeline stages.

> **Important:** These steps must be executed in the exact order shown below. Running steps out of order will cause downstream analyses to fail.

1. generalqc
2. bubbleqc
3. doubletqc
4. voidqc
5. cellqc
6. ambientqc
7. hqcr_ident
8. hqcr_celltype
9. hqpr_metrices (has to be run for each staining)
10. hqpr_clustering (has to be run for each staining)
11. hqpr_clustering (has to be run for each staining)
12. hqpr_refinement (has to be run for each staining)
13. hqpr_bounding_box (has to be run for each staining)
14. hqpr_celltype (has to be run for each staining)
15. hqtr_metrices
16. hqtr_ac
17. hqtr_qv
18. hqtr_clustering
19. hqtr_refinement
20. hqtr_bounding_box
21. hqtr_celltype
22. combine_masks (has to be run for each staining)
23. transcriptqc
24. modelqc
25. cellcycleqc
26. analysis_overview
27. analysis_cluster
28. analysis_category

### Example

To run the first pipeline step (`generalqc`), execute:

```bash
python3 -m spoqc -s generalqc -i [input_spatial_data_bundle] -o [output_folder] -t [spoqc_tmp_folder] -n [n_cores] -a [annotation_file]
```

Wait until the step has completed successfully before continuing with the next step in the list.