# Output reports

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

Below is an example report, generated on the full [10x Xenium breast cancer data Rep1](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast). The slide includes a DAPI staining labeled as "0". SpoQC is capable of analyzing all stainings in your dataset.

## Overview
```{raw} html
:file: ./_static/figures/tutorial/report_p0.html
```

## Subcluster analysis
```{raw} html
:file: ./_static/figures/tutorial/report_p1.html
```

## Spatial plots Leiden clusters
```{raw} html
:file: ./_static/figures/tutorial/report_p2.html
```

## Spatial plots annotation clusters
```{raw} html
:file: ./_static/figures/tutorial/report_p3.html
```

## High quality regions (HQRs)
```{raw} html
:file: ./_static/figures/tutorial/report_p4.html
```

## Individual HQR filters
```{raw} html
:file: ./_static/figures/tutorial/report_p5.html
```

## All HQCR metrics
```{raw} html
:file: ./_static/figures/tutorial/report_p6.html
```

## All HQPR metrics
```{raw} html
:file: ./_static/figures/tutorial/report_p7.html
```

## All HQTR metrics
```{raw} html
:file: ./_static/figures/tutorial/report_p8.html
```

## All HQTR metrics
```{raw} html
:file: ./_static/figures/tutorial/report_p8.html
```