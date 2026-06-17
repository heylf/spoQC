# Contribute

There are several ways to contribute to spoQC. The project is built around four main pillars:

- metrics
- priors
- subworkflows
- standard pre- and postprocessing scripts

> [!NOTE]
> We are currently working on standardizing these components and providing templates to make contributions easier and more consistent.

## `spoqc/metrics/`

Metrics are used to quantify different aspects of spatial transcriptomics data quality. Currently, metrics are organized into three categories:

- image metrics
- segmentation metrics
- transcript density metrics

Some metrics may be relevant to multiple categories. The organization of these layers is still being refined as spoQC evolves.

### Examples

**Segmentation metric**

`spoqc/metrics/segmentation/overlap_area.py`

Segmentation metrics must be linked back to individual cells and stored in the SpatialData object (within the associated AnnData table).

**Image metric**

`spoqc/metrics/image/edge_strength.py`

Image metrics should be saved as a one-dimensional (1D) matrix.

**Transcript density metric**

`spoqc/metrics/image/transcript_density_image.py`

Transcript density metrics should also be saved as a one-dimensional (1D) matrix.

## `spoqc/priors/`

Priors are used to estimate the initial probability that a spatial observation (for example, a cell or pixel) is of high or low quality based on a specific metric.

All priors are combined in:

```
spoqc/priors/combine_priors.py
```

Each prior contributes evidence about the quality of a spatial observation and is integrated into the overall quality assessment.

## `spoqc/subworkflows/`

SpoQC contains several predefined subworkflows that automate common analysis tasks.

Some workflows, such as:

```
qc_doublets.py
```

serve as entry points for metric calculation, quality assessment, visualization, and reporting.

Subworkflows are a good place to contribute additional analysis pipelines or improve existing workflows.

## Standard Pre- and Postprocessing Scripts

SpoQC also includes scripts for common preprocessing and postprocessing operations.

Examples include:

- normalization methods
- data transformations
- filtering procedures
- result aggregation and reporting

Contributions that improve interoperability with new data formats or analysis workflows are particularly welcome.

## How to Add a New Metric

To contribute a new metric, follow these steps:

1. Identify which metric category the new metric belongs to (image, segmentation, transcript density, or another relevant layer).
2. Implement the metric calculation and place the script in the appropriate folder under:

   ```
   spoqc/metrics/
   ```

3. Create a corresponding prior estimation method and place it in:

   ```
   spoqc/priors/
   ```

4. Register the new prior in:

   ```
   spoqc/priors/combine_priors.py
   ```

5. Ensure that the prior returns the probability that a spatial observation (for example, a cell or pixel) is of **high quality**.

### Checklist for New Metrics

- [ ] Metric implementation added to `spoqc/metrics/`
- [ ] Metric output stored in the expected format
- [ ] Prior implementation added to `spoqc/priors/`
- [ ] Prior registered in `spoqc/priors/combine_priors.py`
- [ ] Prior represents the probability of high-quality observations
- [ ] Documentation and examples added where appropriate


