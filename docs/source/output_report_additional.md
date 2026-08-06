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

Below is an example report, generated on a subsample of the [10x Xenium breast cancer data Rep1](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast).

## General single cell metrics
```{raw} html
<iframe
    class="scrollable-report-frame"
    src="./_static/figures/tutorial/rna_qc_sample_mqc.html"
    loading="lazy"
    style="width:100%; height:800px; border:none;">
</iframe>
```

## HQCR celltype analysis
```{raw} html
<iframe
    class="scrollable-report-frame"
    src="./_static/figures/tutorial/hqcr_celltype_qc_analysis.html"
    loading="lazy"
    style="width:100%; height:800px; border:none;">
</iframe>
```

## HQCR cell region analysis
```{raw} html
<iframe
    class="scrollable-report-frame"
    src="./_static/figures/tutorial/hqcr_cell_region.html"
    loading="lazy"
    style="width:100%; height:800px; border:none;">
</iframe>
```

## HQPR celltype analysis
```{raw} html
<iframe
    class="scrollable-report-frame"
    src="./_static/figures/tutorial/hqpr_celltype_qc_analysis.html"
    loading="lazy"
    style="width:100%; height:800px; border:none;">
</iframe>
```

## HQTR celltype analysis
```{raw} html
<iframe
    class="scrollable-report-frame"
    src="./_static/figures/tutorial/hqtr_celltype_qc_analysis.html"
    loading="lazy"
    style="width:100%; height:800px; border:none;">
</iframe>
```