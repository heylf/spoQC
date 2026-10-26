# Contribute

There are several ways to contribute to spoQC. Since version 0.1.0 spoQC is object oriented, so that new metrics and priors can be added without touching the rest of the pipeline. The project is built around four main pillars:

- metrics (`spoqc/metrics/`)
- priors (`spoqc/priors/`)
- missions (`spoqc/missions/`, formerly called `subworkflows`)
- the core framework and standard pre- and postprocessing scripts (`spoqc/core/`, `spoqc/general/`, `spoqc/helperfuncs.py`)

## Code structure

### `spoqc/core/`

The core classes that connect all parts of spoQC:

| File | Content |
|---|---|
| `starship.py` | `Enterpise`, the central object of a run. It holds the parsed arguments (`enterprise.args`), the data (`enterprise.cargo`), the metric sets, the prior sets and the HQR sets. It is passed to every metric, prior and mission. |
| `_config.py` | `Args`, the parsed command line options and fixed parameters (e.g., seed, image type, resolution). |
| `_data.py` | `CargoSpatialData`, which loads the SpatialData object (`enterprise.cargo.sdata`) and performs the standard data processing. |
| `metric.py` | `Metric` and `MetricSet`. |
| `prior.py` | `Prior` and `PriorSet`, including the prior combination methods. |
| `hqr.py` | `HqcrSet`, `HqprSet` and `HqtrSet`, which collect the metrics used for the HQCR, HQPR and HQTR analysis. |
| `_output_structure.py` | Creates the output folder structure for each step. |

### `spoqc/metrics/`

Metrics quantify different aspects of spatial transcriptomics data quality. They are organized by the high quality region (HQR) they belong to:

- `spoqc/metrics/hqcr/`: segmentation (cell) metrics, e.g., `convexity.py`, `doublet_score.py`
- `spoqc/metrics/hqpr/`: image (staining) metrics, e.g., `edge_strength.py`
- `spoqc/metrics/hqtr/`: transcript (density) metrics, e.g., `qv_image.py`, `ac_image.py`

### `spoqc/priors/`

Priors estimate the probability that a spatial observation (a cell or a pixel) is of **high quality** based on one or several metrics. They are organized in the same way as the metrics:

- `spoqc/priors/hqcr/`, e.g., `negative_probe_counts.py`, `doublet_distance.py`
- `spoqc/priors/hqpr/`, e.g., `pixel_score.py`
- `spoqc/priors/hqtr/`, e.g., `pixel_score.py`, `qv_density.py`, `ac_density.py`

All priors of a modality are combined in `spoqc/missions/combine_priors.py`:

- HQCR: asymmetric evidence aggregation and the cell traffic light system (see [Segmentation (cell) metrics](metrics_cell_layer.md)).
- HQPR/HQTR: the mean of all `prob_*` columns returned by the priors.

### `spoqc/missions/`

Missions (formerly `subworkflows`) are the entry points of the individual pipeline steps, e.g., `hqcr.py`, `hqpr.py`, `qc_cellcycle.py` or `final_report.py`. Each mission checks `enterprise.args.step` and runs only for the steps it belongs to. Missions are a good place to contribute additional analyses or to improve existing workflows.

### Standard pre- and postprocessing scripts

SpoQC also includes scripts for common preprocessing and postprocessing operations (`spoqc/general/`, `spoqc/helperfuncs.py`, `spoqc/additional_analysis/`), such as:

- normalization methods
- data transformations
- filtering procedures
- result aggregation and reporting

Contributions that improve interoperability with new data formats or analysis workflows are particularly welcome.

## How metrics and priors are loaded

You do **not** have to register a new metric or prior anywhere. When spoQC starts, `Enterpise.load_metric_sets()` and `Enterpise.load_prior_sets()` import every module in `spoqc/metrics/{hqcr,hqpr,hqtr}/` and `spoqc/priors/{hqcr,hqpr,hqtr}/`:

- Each metric module must provide a function `init_metric(enterprise)` that returns a `core.metric.Metric`.
- Each prior module must provide a function `init_prior(enterprise)` that returns a `core.prior.Prior`.

Modules without these functions are skipped with a warning. Consequently:

- The HQCR clustering uses all `submetrics` of all metrics in `metrics/hqcr/`.
- The HQPR/HQTR analysis uses all metrics in `metrics/hqpr/` and `metrics/hqtr/`.
- The prior combination uses all priors in the respective `priors/` folder.

### `Metric`

| Argument | Description |
|---|---|
| `calc_func` | The function that calculates the metric. It is called as `calc_func(*args, **kwargs)`. |
| `name` | Unique name of the metric within its modality. Priors refer to this name in `needs_metrics`. |
| `submetrics` | Names of the values the metric produces (e.g., `["convexity_metric_cell", "convexity_min_nuceli"]`). For HQCR these are columns in `sdata['table'].obs` and are used for the HQCR clustering. |
| `modality` | `hqcr`, `hqpr` or `hqtr`. |
| `needs_metrics` | Names of metrics that have to be calculated before this metric. `MetricSet.calculate_metrics` resolves these dependencies and stops with an error on circular dependencies. |
| `step_when_it_is_calculated` | Steps (`-s`) in which the metric is calculated, e.g., `["cellqc", "all"]`. |
| `should_be_calculated` | Set to `False` to switch the metric off (default: `True`). |
| `loaded_for_analysis` | Whether the metric is loaded for the additional analysis. |
| `loaded_for_visualization` | Submetrics that are visualized in the report. |
| `args`, `kwargs` | Arguments passed to `calc_func`. |

### `Prior`

| Argument | Description |
|---|---|
| `calc_func` | The function that calculates the prior. It is called as `calc_func(*args, **kwargs)`. |
| `name` | Unique name of the prior within its modality. |
| `modality` | `hqcr`, `hqpr` or `hqtr`. |
| `needs_metrics` | Names of the metrics the prior is based on. spoQC stops with an error if one of these metrics does not exist. |
| `tmp_path` | Optional path to temporary data. |
| `args`, `kwargs` | Arguments passed to `calc_func`. |

### Expected output

- **HQCR metrics** write their results as columns into `sdata['table'].obs`. After each step, new columns are saved to `[tmp_dir]/{step}_output_hqcr.parquet` and loaded again in later steps.
- **HQPR/HQTR metrics** write a one-dimensional (1D) array (one value per pixel) to `[tmp_dir]/metrices/{modality}/[{staining}/]{name}_output_{modality}[_{staining}].parquet`.
- **HQCR priors** return one probability per cell (array-like), where higher values mean higher quality.
- **HQPR/HQTR priors** return a dask dataframe with one row per pixel that contains a column `prob_{name}` with the probability of high quality.

## How to add a new metric

1. Identify the modality of the new metric (`hqcr`, `hqpr` or `hqtr`).
2. Add a new module to `spoqc/metrics/{modality}/` with the calculation function and an `init_metric` function. Template (based on `spoqc/metrics/hqcr/convexity.py`):

   ```python
   from ... import core

   def _calc_my_metric(sdata, figure_path):
       sdata['table'].obs['my_metric'] = ...

   def init_metric(enterprise):

       # These have to be defined.
       name = "my_metric"
       submetrics = ["my_metric"]
       modality = "hqcr"
       needs_metrics = []
       step_when_it_is_calculated = ["cellqc", "all"]
       loaded_for_analysis = True
       loaded_for_visualization = ["my_metric"]

       # These are given to your metric calc function.
       args = [enterprise.cargo.sdata, f"{enterprise.args.output_dir}/cellqc/"]
       kwargs = None

       metric = core.metric.Metric(
           _calc_my_metric,
           name,
           submetrics,
           modality,
           needs_metrics = needs_metrics,
           step_when_it_is_calculated = step_when_it_is_calculated,
           loaded_for_analysis = loaded_for_analysis,
           loaded_for_visualization = loaded_for_visualization,
           args = args,
           kwargs = kwargs,
       )

       return metric
   ```

   `spoqc/metrics/hqcr/valid_geometries.py` is a minimal example of a metric whose values are calculated elsewhere.

## How to add a new prior

1. Add a new module to `spoqc/priors/{modality}/` with the calculation function and an `init_prior` function. Template (based on `spoqc/priors/hqcr/negative_probe_counts.py`):

   ```python
   from ... import core

   def _calc_probs(df, figure_path):
       # Return the probability that each cell is of HIGH quality.
       return ...

   def init_prior(enterprise):

       # These have to be defined.
       name = "my_metric_prior"
       modality = "hqcr"
       tmp_path = None
       needs_metrics = ["my_metric"]

       # These are given to your prior calc function.
       args = [enterprise.hqcr_set.cell_clustering_df, f'{enterprise.args.output_dir}/hqcr/hqcr_ident/']
       kwargs = None

       prior = core.prior.Prior(
           _calc_probs,
           name,
           modality,
           needs_metrics = needs_metrics,
           tmp_path = tmp_path,
           args = args,
           kwargs = kwargs,
       )

       return prior
   ```

2. Make sure the prior returns the probability that a spatial observation (a cell or pixel) is of **high quality**.

```{note}
Every HQCR prior is automatically part of the asymmetric evidence aggregation and of the cell traffic light system.
```

### Checklist

- [ ] Metric module with `init_metric` added to `spoqc/metrics/{modality}/`
- [ ] Metric output stored in the expected format
- [ ] Prior module with `init_prior` added to `spoqc/priors/{modality}/`
- [ ] `needs_metrics` of the prior refers to existing metric names
- [ ] Prior represents the probability of high-quality observations
- [ ] Entry added to `CHANGELOG.md`
- [ ] Documentation and examples added where appropriate

## Versioning

Use `bump_version.py` in the repository root for a quick version bump of the tool, and document your changes in `CHANGELOG.md`:

```bash
python3 bump_version.py -v [new_version]
```
