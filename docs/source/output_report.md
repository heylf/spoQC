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
per-modality metrics. `report/` also holds a `report_lowres.html`, a
lightweight version of the same report with downscaled images, more suitable
for quickly sharing or viewing a report.

Below is an example report, generated on the example dataset used throughout
this documentation.

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