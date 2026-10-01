
# In[]
from .. import hqr
from ..core import parquet, raster, threads

# In[]
def start_pixel_mask_refinement(
        figure_path,
        spoqc_tmp_folder,
        modality,
        dim_x,
        dim_y,
        beta,
        max_iter,
        *,
        staining=None,
        beliefs_raw=None,
    ):
    """beliefs_raw: start_pixel_qc's per-pixel beliefs when it ran in this process; None reads them from mask_raw."""

# # In[]

# import dask.dataframe as dd
# import pandas as pd
# import numpy as np

# from spoqc import helperfuncs
# from spoqc import hqr


# figure_path = CONST.FIGURE_PATH
# spoqc_tmp_folder = CONST.TMP_PATH
# modality = 'hqpr'
# beta = 1.5
# max_iter = 15
# staining=CONST.STAINING





# In[]

    prefix = modality
    suffix = 'raw'
    if ( staining ):
        figure_path = f'{figure_path}/{modality}/{modality}_refinement/{staining}/'
        prefix = f'{modality}_{staining}'
    else:
        figure_path = f'{figure_path}/{modality}/{modality}_refinement/'

    if ( beliefs_raw is None ):
        beliefs_raw = raster.read_pixel_columns(
            f'{spoqc_tmp_folder}/{prefix}_output_mask_raw', [f"{prefix}_beliefs"], dim_x * dim_y, threads.budget()
        )[f"{prefix}_beliefs"]

    # Start the refinement of the proability for the pixel score.
    beliefs, labels = hqr.markov_random_field_zarr_parallel.first_version_loopy_belief_propagation_parallel(
        beliefs_raw.reshape((dim_x, dim_y)),
        beta=beta,
        max_iter=max_iter,
        normalize='total'
    )

    hqr.markov_random_field_zarr_parallel.visualize_markov_calculation(
        beliefs_raw.reshape((dim_x, dim_y)),
        labels[:],
        figure_path
    )

    #  Write out masks.
    print(f"[NOTE] Write out {prefix} masks")

    # Build the output from fully in-memory arrays and hand it to dask via
    # from_pandas so the resulting ddf has known, sorted divisions. Pairing a
    # freshly chunked dask.array against image_ddf.index (unknown divisions,
    # from a parquet read) preserves index-to-value association but not the
    # physical row order returned by .compute()/round-tripped through parquet.
    columns = {
        f"{prefix}_beliefs": beliefs_raw,
        f"{prefix}_beliefs_smoothed": beliefs.ravel(),
        f"{prefix}_mask_smoothed": labels.ravel(),
    }
    n_rows = len(beliefs_raw)
    parquet.write_parts(f"{spoqc_tmp_folder}/{prefix}_output_mask_smoothed_raw", n_rows,
                        parquet.columns_of(columns), range(0, n_rows, parquet.PART_ROWS), threads.budget())


# # In[]
# direct = beliefs[:].flatten()

# back = dd.read_parquet(
#     f"{spoqc_tmp_folder}/{prefix}_output_mask_smoothed_raw", 
#     columns=[f"{prefix}_beliefs_smoothed"], 
#     engine="pyarrow"
# )
# back = back[f"{prefix}_beliefs_smoothed"].compute().to_numpy()

# print(np.nanmax(np.abs(direct - back)))

# %%
