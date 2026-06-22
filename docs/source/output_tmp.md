# Output data files

* HQCR = High quality cell region
* HQPR = High quality pixel region
* HQTR = High quality transcript region

The folder `spoQC_tmp/` containt several data files and folders that are useful to filter and inspect the quality of the spatialdata.

## Binary and belief masks

### HQCR raw: hqcr_output_mask_raw.parquet

<details>
  <!-- <summary>Click to expand</summary><br>^ -->
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

### HQPR raw: hqpr_*_output_mask_raw
`report/` holds a file names `staining_log.txt`.
Currently spoQC analysis all image channels and uses an integer index because channel names can be very arbitrary in their naming.
Therefore, some of the folder and files for the analysis of pixel quality (HQPR) will have integers standing for the individual image channels.
To know which integer belongs to which channel, please look in to `staining_log.txt`.

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
  * columns: norm_p_informative_pixel, hqpr_mask, hqpr_beliefs

  Explanation:

  * norm_p_informative_pixel: Normalized probability of a pixel beeing of good quality. A value of 1.0 means 100% certainty that the pixel is of good quality.
  * hqpr_mask: Binary mask that the pixel belongs to a HQPR (1) or not (0).
  * hqpr_beliefs: Belief mask for each pixel of the HQPR image.

</details>

### HQTR raw: hqtr_output_mask_raw

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
  * columns: norm_p_informative_pixel, hqtr_mask, hqtr_beliefs

  Explanation:

  * norm_p_informative_pixel: Normalized probability of a pixel beeing of good quality. A value of 1.0 means 100% certainty that the pixel is of good quality.
  * hqtr_mask: Binary mask that the pixel belongs to a HQTR (1) or not (0).
  * hqtr_beliefs: Belief mask for each pixel of the HQTR image.

</details>

## Scores

### HQPR (anti)structure scores: hqpr_*_output_mask_raw
`report/` holds a file names `staining_log.txt`.
Currently spoQC analysis all image channels and uses an integer index because channel names can be very arbitrary in their naming.
Therefore, some of the folder and files for the analysis of pixel quality (HQPR) will have integers standing for the individual image channels.
To know which integer belongs to which channel, please look in to `staining_log.txt`.

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import dask.dataframe as dd
  channel = 0
  hqpr_prob_ddf = dd.read_parquet(f'spoQC_tmp/hqpr_{channel}_output_mask_raw', engine="pyarrow")
  # If you want to convert it into a pandas data frame use line below.
  # Be aware, doing the line below takes time and consumes lots of memory because you read in all the data at once.
  # hqpr_prob_df = hqpr_prob_ddf.compute()
  ```

  Structure:

  * index: pixel_id
  * columns: cluster, s_score, as_score, intensity, p_informative_pixel, norm_p_informative_pixel

  Explanation:

  * cluster: The pixel cluster the pixel belonged to.
  * s_score: Structure score defining if the pixel contributes to biological important structures. The higher the score the more information the pixel holds for the structure.
  * as_score: Antistructure score defining if the pixel contributes to biological unimportant structures (e.g., bubbles). The higher the score the less information the pixel holds for biological important structures.
  * p_informative_pixel: Probability of a pixel beeing of good quality.
  * norm_p_informative_pixel: Normalized prability of a pixel beeing of good quality. A value of 1.0 means 100% certainty that the pixel is of good quality.
</details>


### HQTR (anti)structure scores: hqtr_output_mask_raw

<details>
  <summary>Click to expand</summary><br>
  Read in with:

  ```
  import dask.dataframe as dd
  hqtr_prob_ddf = dd.read_parquet('spoQC_tmp/hqtr_output_mask_raw', engine="pyarrow")
  # If you want to convert it into a pandas data frame use line below.
  # Be aware, doing the line below takes time and consumes lots of memory because you read in all the data at once.
  # hqtr_prob_df = hqtr_prob_ddf.compute()
  ```

  Structure:

  * index: pixel_id
  * columns: cluster, s_score, as_score, intensity, p_informative_pixel, norm_p_informative_pixel

  Explanation:

  * cluster: The pixel cluster the pixel belonged to.
  * s_score: Structure score defining if the pixel contributes to biological important structures. The higher the score the more information the pixel holds for the structure.
  * as_score: Antistructure score defining if the pixel contributes to biological unimportant structures (e.g., bubbles). The higher the score the less information the pixel holds for biological important structures.
  * p_informative_pixel: Probability of a pixel beeing of good quality.
  * norm_p_informative_pixel: Normalized prability of a pixel beeing of good quality. A value of 1.0 means 100% certainty that the pixel is of good quality.
</details>