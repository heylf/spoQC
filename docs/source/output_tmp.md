# Output temporary data files

* HQCR = High quality cell region
* HQPR = High quality pixel region
* HQTR = High quality transcript region

The folder `spoQC_tmp/` (the path passed via `-t`/`--tmp`) contains several
raw, per-pixel data files and folders that are useful to filter and inspect
the quality of the spatialdata. These are the pixel-level building blocks
that spoQC later aggregates to the cell level in
`report/analysis/rna_qc_annotated.h5ad` (see [Output Data](output_data.md)).

For each modality, spoQC writes a **raw** mask/belief file and a
**smoothed** counterpart: the smoothed version runs the raw belief image
through Markov random field (loopy belief propagation) smoothing, the same
smoothing that produces the `*_smoothed` columns documented in
[Output Data](output_data.md).

## HQCR

### Raw: hqcr_output_mask_raw.parquet

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import pandas as pd
  hqcr_mask = pd.read_parquet('spoQC_tmp/hqcr_output_mask_raw.parquet')
  ```

  Structure:

  * index: pixel_id
  * columns: hqcr_mask, hqcr_beliefs

  Explanation:

  * hqcr_mask: Binary mask that the pixel belongs to a HQCR (1) or not (0).
  * hqcr_beliefs: Belief mask for each pixel of the HQCR image.

</details>

### Smoothed: hqcr_output_mask_smoothed_raw.parquet

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import pandas as pd
  hqcr_mask_smoothed = pd.read_parquet('spoQC_tmp/hqcr_output_mask_smoothed_raw.parquet')
  ```

  Structure:

  * index: pixel_id
  * columns: hqcr_beliefs_smoothed, hqcr_mask_smoothed

  Explanation:

  * hqcr_beliefs_smoothed: Markov-smoothed belief mask for each pixel of the HQCR image.
  * hqcr_mask_smoothed: Binary mask that the pixel belongs to a HQCR (1) or not (0), derived from the smoothed beliefs.

</details>

### HQCR celltype refined: hqcr_output_mask_celltype_refined.parquet

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import pandas as pd
  hqcr_mask = pd.read_parquet('spoQC_tmp/hqcr_output_mask_celltype_refined.parquet')
  ```

  Structure:

  * index: pixel_id
  * columns: hqcr_mask, hqcr_beliefs

  Explanation:

  * hqcr_mask: Binary mask that the pixel belongs to a HQCR (1) or not (0). **This mask was refined with celltype information.**
  * hqcr_beliefs: Belief mask for each pixel of the HQCR image.

</details>

### HQCR celltype refined smoothed: hqcr_output_mask_smoothed_celltype_refined.parquet

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import pandas as pd
  hqcr_mask_smoothed = pd.read_parquet('spoQC_tmp/hqcr_output_mask_smoothed_celltype_refined.parquet')
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

### Raw: hqpr_*_output_mask_raw

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import dask.dataframe as dd
  channel = 0
  hqpr_mask_ddf = dd.read_parquet(f'spoQC_tmp/hqpr_{channel}_output_mask_raw', engine="pyarrow")
  # If you want to convert it into a pandas data frame use line below.
  # Be aware, doing the line below takes time and consumes lots of memory because you read in all the data at once.
  # hqpr_mask_df = hqpr_mask_ddf.compute()
  ```

  Structure:

  * index: pixel_id
  * columns: cluster, s_score, as_score, intensity, p_informative_pixel, hqpr_*_beliefs, hqpr_*_mask

  Explanation:

  * cluster: The pixel cluster the pixel belonged to.
  * s_score: Structure score defining if the pixel contributes to biological important structures. The higher the score the more information the pixel holds for the structure.
  * as_score: Antistructure score defining if the pixel contributes to biological unimportant structures (e.g., bubbles). The higher the score the less information the pixel holds for biological important structures.
  * intensity: Raw pixel intensity of the staining channel.
  * p_informative_pixel: Probability of a pixel beeing of good quality, based on the pixel's cluster.
  * hqpr_*_beliefs: Min-max normalized version of p_informative_pixel. A value of 1.0 means 100% certainty that the pixel is of good quality.
  * hqpr_*_mask: Binary mask that the pixel belongs to a HQPR (1) or not (0), derived by thresholding hqpr_*_beliefs at 0.5.

</details>

### Smoothed: hqpr_*_output_mask_smoothed_raw

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import dask.dataframe as dd
  channel = 0
  hqpr_mask_smoothed_ddf = dd.read_parquet(f'spoQC_tmp/hqpr_{channel}_output_mask_smoothed_raw', engine="pyarrow")
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

### Raw: hqtr_output_mask_raw

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import dask.dataframe as dd
  hqtr_mask_ddf = dd.read_parquet('spoQC_tmp/hqtr_output_mask_raw', engine="pyarrow")
  # If you want to convert it into a pandas data frame use line below.
  # Be aware, doing the line below takes time and consumes lots of memory because you read in all the data at once.
  # hqtr_mask_df = hqtr_mask_ddf.compute()
  ```

  Structure:

  * index: pixel_id
  * columns: cluster, s_score, as_score, intensity, p_informative_pixel, hqtr_beliefs, hqtr_mask

  Explanation:

  * cluster: The pixel cluster the pixel belonged to.
  * s_score: Structure score defining if the pixel contributes to biological important structures. The higher the score the more information the pixel holds for the structure.
  * as_score: Antistructure score defining if the pixel contributes to biological unimportant structures (e.g., bubbles). The higher the score the less information the pixel holds for biological important structures.
  * intensity: Raw transcript-density intensity of the pixel.
  * p_informative_pixel: Probability of a pixel beeing of good quality, based on the pixel's cluster.
  * hqtr_beliefs: Min-max normalized combination of p_informative_pixel with the transcript QV and ambient-RNA (AC) probabilities below (see hqtr_output_qv_prob and hqtr_output_ac_prob). A value of 1.0 means 100% certainty that the pixel is of good quality.
  * hqtr_mask: Binary mask that the pixel belongs to a HQTR (1) or not (0), derived by thresholding hqtr_beliefs at 0.5.

</details>

### Smoothed: hqtr_output_mask_smoothed_raw

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import dask.dataframe as dd
  hqtr_mask_smoothed_ddf = dd.read_parquet('spoQC_tmp/hqtr_output_mask_smoothed_raw', engine="pyarrow")
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

### Transcript QV probability: hqtr_output_qv_prob

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import dask.dataframe as dd
  hqtr_qv_ddf = dd.read_parquet('spoQC_tmp/hqtr_output_qv_prob', engine="pyarrow")
  ```

  Structure:

  * index: pixel_id
  * columns: qv_density, p_qv_density, norm_p_qv_density

  Explanation:

  * qv_density: Transcript quality value (QV) density of the pixel.
  * p_qv_density: Probability that the pixel's QV density indicates good quality (10x Genomics uses a QV threshold of 20).
  * norm_p_qv_density: Min-max normalized version of p_qv_density. This is one of the components summed into hqtr_beliefs above.

</details>

### Ambient RNA probability: hqtr_output_ac_prob

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import dask.dataframe as dd
  hqtr_ac_ddf = dd.read_parquet('spoQC_tmp/hqtr_output_ac_prob', engine="pyarrow")
  ```

  Structure:

  * index: pixel_id
  * columns: ac_density, p_ac_density, norm_p_ac_density

  Explanation:

  * ac_density: Density of maximum gene autocorrelation at the pixel, used to flag potential ambient/spillover RNA.
  * p_ac_density: Probability that the pixel's autocorrelation density indicates good (non-ambient) quality.
  * norm_p_ac_density: Min-max normalized version of p_ac_density. This is one of the components summed into hqtr_beliefs above.

</details>
