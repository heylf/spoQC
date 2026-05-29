# Contribute

SpoQC has four important pillars.

- metrics
- priors
- subworkflows
- standard pre- and postprocessing scripts

> [!NOTE]
> We are working to standardize these pillars to be able to provide templates to make contribution easier.

## spoqc/metrics/

Metrics are currently split into image, segmentation and transcript density relevant. We want to stress out, that some metrics might overlap with other modalities. We are currently still optimizing the layers of spoQC.

Examples:
- segmentation metrics example: `spoqc/metrics/segmentation/overlap_area.py`. The metric has to be linked back to the cell and it needs to be saved in the SpatialData object (in the anndata).
- image metrics example: `spoqc/metrics/image/edge_strength.py`. The metric should be saved as a 1D matrix.
- transcript density metrics example: `spoqc/metrics/image/transcript_density_image.py`. The metric should be saved as a 1D matrix.

## spoqc/priors/

Prior code is used in order to estimate an initial prior for the defined metric for bad (good) spatial observations (e.g., pixel). Priors are combined in the script `spoqc/priors/combine_priors.py`.

## spoqc/subworkflows/

SpoQC has predefined subworkflows for various tasks. Some workflows, such as `qc_doublets.py`, are used to start the data processing and plotting for metrics.

## standard pre- and postprocessing scripts

SpoQC has also scripts for various data pre- and postprocessing steps, such as normalizations.

## How to provide a new metric?

1. First identify into which layer the metric falls.
2. Write an individual script for the calculation of the metric and place it in the `spoqc/metrics`.
3. Write a prior estimation for the individual metric and place it in the `spoqc/priors` folder.
4. Add the prior to the `spoqc/priors/combine_priors.py` script. Each prior has to represent the prbability of the good quality of the spatial observation (e.g., cell or pixel).



