# Additioanl reports

* HQCR = High quality cell region
* HQPR = High quality pixel region
* HQTR = High quality transcript region
* LQCR = Low quality cell region
* LQPR = Low quality pixel region
* LQTR = Low quality transcript region

The `report/` folder holds `report.html`, spoQC's interactive, navigable
summary of a run: a single self-contained page with a sidebar to browse the
overview, subcluster analysis, spatial plots, HQR/HQPR/HQTR filters, and
per-modality metrics.

The following report pages are part of the final report and can be used to further inspect the quality and celltype dependent quality.

## General single cell metrics
```{raw} html
:file: ./_static/figures/tutorial/rna_qc_sample_mqc.html
```

## HQCR celltype analysis
```{raw} html
:file: ./_static/figures/tutorial/hqcr_celltype_qc_analysis.html
```

## HQCR cell region analysis
```{raw} html
:file: ./_static/figures/tutorial/hqcr_cell_region.html
```

## HQPR celltype analysis
```{raw} html
:file: ./_static/figures/tutorial/hqpr_celltype_qc_analysis.html
```

## HQTR celltype analysis
```{raw} html
:file: ./_static/figures/tutorial/hqtr_celltype_qc_analysis.html
```