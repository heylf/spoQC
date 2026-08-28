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

## (Run Individual Pipeline Steps)

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
11. hqpr_refinement (has to be run for each staining)
12. hqpr_bounding_box (has to be run for each staining)
13. hqpr_celltype (has to be run for each staining)
14. hqtr_metrices
15. hqtr_ac
16. hqtr_qv
17. hqtr_clustering
18. hqtr_refinement
19. hqtr_bounding_box
20. hqtr_celltype
21. combine_masks (has to be run for each staining)
22. transcriptqc
23. modelqc
24. cellcycleqc
25. analysis_overview
26. analysis_cluster
27. analysis_category
28. final_report

### Example

To run the first pipeline step (`generalqc`), execute:

```bash
python3 -m spoqc -s generalqc -i [input_spatial_data_bundle] -o [output_folder] -t [spoqc_tmp_folder] -n [n_cores] -a [annotation_file]
```

Wait until the step has completed successfully before continuing with the next step in the list.