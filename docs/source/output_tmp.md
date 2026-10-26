# Output temporary data files

* HQCR = High quality cell region
* HQPR = High quality pixel region
* HQTR = High quality transcript region

The folder `spoQC_tmp/` (the path passed via `-t`/`--tmp_dir`) contains several
raw, per-cell and per-pixel data files and folders that are useful to filter and inspect
the quality of the spatialdata. The pixel-level files are the building blocks
that spoQC later aggregates to the cell level in
`report/analysis/rna_qc_annotated.h5ad` (see [Output Data](output_data.md)).

For each modality, spoQC writes a **raw** mask/belief file and a
**smoothed** counterpart: the smoothed version runs the raw belief image
through Markov random field (loopy belief propagation) smoothing, the same
smoothing that produces the `*_smoothed` columns documented in
[Output Data](output_data.md).

```{note}
The temporary files were renamed in version 0.0.2 (e.g., `hqcr_output_mask_raw.parquet` is now `mask_raw_output_hqcr.parquet`). See `CHANGELOG.md` for the full list.
```

## HQCR

### Cell metrics: {step}_output_hqcr.parquet

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import pandas as pd
  cellqc_df = pd.read_parquet('spoQC_tmp/cellqc_output_hqcr.parquet')
  ```

  Structure:

  * index: cell_id
  * columns: all HQCR metric columns added to `sdata['table'].obs` in that step

  Explanation:

  After each step that calculates HQCR metrics (e.g., `generalqc`, `doubletqc`, `voidqc`, `cellqc`), spoQC writes the newly added cell columns to `{step}_output_hqcr.parquet`. These files are read in again by later steps. The columns are described in [Output Data](output_data.md).

</details>

### Traffic light: traffic_light_output_hqcr.parquet

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import pandas as pd
  traffic_light = pd.read_parquet('spoQC_tmp/traffic_light_output_hqcr.parquet')
  ```

  Structure:

  * index: cell_id
  * columns: hqcr_traffic_light

  Explanation:

  * hqcr_traffic_light: Traffic light QC level of the cell (`green`, `yellow` or `red`), see [Segmentation (cell) metrics](metrics_cell_layer.md).

</details>

### Raw: mask_raw_output_hqcr.parquet

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import pandas as pd
  hqcr_mask = pd.read_parquet('spoQC_tmp/mask_raw_output_hqcr.parquet')
  ```

  Structure:

  * index: pixel_id
  * columns: hqcr_beliefs, hqcr_mask

  Explanation:

  * hqcr_beliefs: Belief mask for each pixel of the HQCR image, i.e., the combined HQCR prior probability of the cell the pixel belongs to.
  * hqcr_mask: Binary mask that the pixel belongs to a HQCR (1) or not (0), derived by thresholding hqcr_beliefs at 0.5.

</details>

### Smoothed: mask_smoothed_raw_output_hqcr.parquet

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import pandas as pd
  hqcr_mask_smoothed = pd.read_parquet('spoQC_tmp/mask_smoothed_raw_output_hqcr.parquet')
  ```

  Structure:

  * index: pixel_id
  * columns: hqcr_beliefs_smoothed, hqcr_mask_smoothed

  Explanation:

  * hqcr_beliefs_smoothed: Markov-smoothed belief mask for each pixel of the HQCR image.
  * hqcr_mask_smoothed: Binary mask that the pixel belongs to a HQCR (1) or not (0), derived from the smoothed beliefs.

</details>

### HQCR celltype refined: mask_celltype_refined_output_hqcr.parquet

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import pandas as pd
  hqcr_mask = pd.read_parquet('spoQC_tmp/mask_celltype_refined_output_hqcr.parquet')
  ```

  Structure:

  * index: pixel_id
  * columns: hqcr_beliefs, hqcr_mask

  Explanation:

  * hqcr_beliefs: Belief mask for each pixel of the HQCR image.
  * hqcr_mask: Binary mask that the pixel belongs to a HQCR (1) or not (0). **This mask was refined with celltype information.**

</details>

### HQCR celltype refined smoothed: mask_smoothed_celltype_refined_output_hqcr.parquet

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import pandas as pd
  hqcr_mask_smoothed = pd.read_parquet('spoQC_tmp/mask_smoothed_celltype_refined_output_hqcr.parquet')
  ```

  Structure:

  * index: pixel_id
  * columns: hqcr_beliefs_smoothed, hqcr_mask_smoothed

  Explanation:

  * hqcr_beliefs_smoothed: Markov-smoothed belief mask for each pixel of the HQCR image, **refined with celltype information**.
  * hqcr_mask_smoothed: Binary mask that the pixel belongs to a HQCR (1) or not (0), derived from the smoothed, celltype-refined beliefs.

</details>

## HQPR
`report/` holds a file named `staining_log.txt`.
Currently spoQC analyses all image channels and uses an integer index because channel names can be very arbitrary in their naming.
Therefore, some of the folders and files for the analysis of pixel quality (HQPR) will have integers standing for the individual image channels.
To know which integer belongs to which channel, please look into `staining_log.txt`.

### Image metrics: metrices/hqpr/{channel}/

The individual image metrics of a channel are stored as 1D arrays (one value per pixel) in `spoQC_tmp/metrices/hqpr/{channel}/{metric}_output_hqpr_{channel}.parquet`, e.g., `edge_strength_output_hqpr_0.parquet`.

### Raw: mask_raw_output_hqpr_{channel}

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import dask.dataframe as dd
  channel = 0
  hqpr_mask_ddf = dd.read_parquet(f'spoQC_tmp/mask_raw_output_hqpr_{channel}', engine="pyarrow")
  # If you want to convert it into a pandas data frame use line below.
  # Be aware, doing the line below takes time and consumes lots of memory because you read in all the data at once.
  # hqpr_mask_df = hqpr_mask_ddf.compute()
  ```

  Structure:

  * index: pixel_id
  * columns: cluster, s_score, as_score, intensity, p_informative_pixel, prob_pixel_score, mask_pixel_score, hqpr_*_beliefs, hqpr_*_mask

  Explanation:

  * cluster: The pixel cluster the pixel belonged to.
  * s_score: Structure score defining if the pixel contributes to biological important structures. The higher the score the more information the pixel holds for the structure.
  * as_score: Antistructure score defining if the pixel contributes to biological unimportant structures (e.g., bubbles). The higher the score the less information the pixel holds for biological important structures.
  * intensity: Raw pixel intensity of the staining channel.
  * p_informative_pixel: Probability density of a pixel being of good quality, based on the pixel's cluster.
  * prob_pixel_score: Min-max normalized version of p_informative_pixel (pixel score prior).
  * mask_pixel_score: Binary mask derived by thresholding prob_pixel_score at 0.5.
  * hqpr_*_beliefs: Mean of all `prob_*` columns of the HQPR priors. A value of 1.0 means 100% certainty that the pixel is of good quality.
  * hqpr_*_mask: Binary mask that the pixel belongs to a HQPR (1) or not (0), derived by thresholding hqpr_*_beliefs at 0.5.

</details>

### Smoothed: mask_smoothed_raw_output_hqpr_{channel}

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import dask.dataframe as dd
  channel = 0
  hqpr_mask_smoothed_ddf = dd.read_parquet(f'spoQC_tmp/mask_smoothed_raw_output_hqpr_{channel}', engine="pyarrow")
  # If you want to convert it into a pandas data frame use line below.
  # Be aware, doing the line below takes time and consumes lots of memory because you read in all the data at once.
  # hqpr_mask_smoothed_df = hqpr_mask_smoothed_ddf.compute()
  ```

  Structure:

  * index: pixel_id
  * columns: hqpr_*_beliefs, hqpr_*_beliefs_smoothed, hqpr_*_mask_smoothed

  Explanation:

  * hqpr_*_beliefs: The raw (non-smoothed) belief, kept for reference.
  * hqpr_*_beliefs_smoothed: Markov-smoothed belief mask for each pixel of the HQPR image.
  * hqpr_*_mask_smoothed: Binary mask that the pixel belongs to a HQPR (1) or not (0), derived from the smoothed beliefs.

</details>

## HQTR

### Transcript metrics: metrices/hqtr/

The transcript-density image (`transcript_density_output_hqtr.parquet`) and its image metrics are stored as 1D arrays (one value per pixel) in `spoQC_tmp/metrices/hqtr/`, e.g., `edge_strength_output_hqtr.parquet`.

### Raw: mask_raw_output_hqtr

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import dask.dataframe as dd
  hqtr_mask_ddf = dd.read_parquet('spoQC_tmp/mask_raw_output_hqtr', engine="pyarrow")
  # If you want to convert it into a pandas data frame use line below.
  # Be aware, doing the line below takes time and consumes lots of memory because you read in all the data at once.
  # hqtr_mask_df = hqtr_mask_ddf.compute()
  ```

  Structure:

  * index: pixel_id
  * columns: cluster, s_score, as_score, intensity, p_informative_pixel, prob_pixel_score, mask_pixel_score, qv_density, prob_qv_density, ac_density, d_ac_density, prob_ac_density, hqtr_beliefs, hqtr_mask

  Explanation:

  * cluster: The pixel cluster the pixel belonged to.
  * s_score: Structure score defining if the pixel contributes to biological important structures. The higher the score the more information the pixel holds for the structure.
  * as_score: Antistructure score defining if the pixel contributes to biological unimportant structures (e.g., bubbles). The higher the score the less information the pixel holds for biological important structures.
  * intensity: Raw transcript-density intensity of the pixel.
  * p_informative_pixel: Probability density of a pixel being of good quality, based on the pixel's cluster.
  * prob_pixel_score: Min-max normalized version of p_informative_pixel (pixel score prior).
  * mask_pixel_score: Binary mask derived by thresholding prob_pixel_score at 0.5.
  * qv_density, prob_qv_density: Transformed transcript QV density and its min-max normalized probability (QV prior).
  * ac_density, d_ac_density, prob_ac_density: Ambient RNA (AC) density, its transformed density and its min-max normalized probability (AC prior).
  * hqtr_beliefs: Mean of prob_pixel_score, prob_qv_density and prob_ac_density. A value of 1.0 means 100% certainty that the pixel is of good quality.
  * hqtr_mask: Binary mask that the pixel belongs to a HQTR (1) or not (0), derived by thresholding hqtr_beliefs at 0.5.

</details>

### Smoothed: mask_smoothed_raw_output_hqtr

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import dask.dataframe as dd
  hqtr_mask_smoothed_ddf = dd.read_parquet('spoQC_tmp/mask_smoothed_raw_output_hqtr', engine="pyarrow")
  # If you want to convert it into a pandas data frame use line below.
  # Be aware, doing the line below takes time and consumes lots of memory because you read in all the data at once.
  # hqtr_mask_smoothed_df = hqtr_mask_smoothed_ddf.compute()
  ```

  Structure:

  * index: pixel_id
  * columns: hqtr_beliefs, hqtr_beliefs_smoothed, hqtr_mask_smoothed

  Explanation:

  * hqtr_beliefs: The raw (non-smoothed) belief, kept for reference.
  * hqtr_beliefs_smoothed: Markov-smoothed belief mask for each pixel of the HQTR image.
  * hqtr_mask_smoothed: Binary mask that the pixel belongs to a HQTR (1) or not (0), derived from the smoothed beliefs.

</details>

### Transcript QV density: qv_density_output_hqtr

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import dask.dataframe as dd
  hqtr_qv_ddf = dd.read_parquet('spoQC_tmp/qv_density_output_hqtr', engine="pyarrow")
  ```

  Structure:

  * index: pixel_id
  * columns: qv_density

  Explanation:

  * qv_density: Transcript quality value (QV) density of the pixel (transcript-quality image, see [Transcript (density) metrics](metrics_transcript_layer.md)). The QV prior (`prob_qv_density`) is stored in `mask_raw_output_hqtr`.

</details>

### Ambient RNA density: ac_density_output_hqtr

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import dask.dataframe as dd
  hqtr_ac_ddf = dd.read_parquet('spoQC_tmp/ac_density_output_hqtr', engine="pyarrow")
  ```

  Structure:

  * index: pixel_id
  * columns: ac_density

  Explanation:

  * ac_density: Combined autocorrelation (AC) density of the pixel, used to flag potential ambient/spillover RNA (see [Transcript (density) metrics](metrics_transcript_layer.md)). The AC prior (`prob_ac_density`) is stored in `mask_raw_output_hqtr`.

</details>
