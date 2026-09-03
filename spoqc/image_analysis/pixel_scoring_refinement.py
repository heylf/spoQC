
# In[]
import dask.dataframe as dd
import pandas as pd

from .. import helperfuncs
from .. import hqr

# In[]
def start_pixel_mask_refinement(
        figure_path,
        spoqc_tmp_folder,
        modality,
        dim_x,
        dim_y,
        *,
        beta=1.5,
        max_iter=15,
        chunk_size=10000,
        staining=None
    ):

    prefix = modality
    suffix = 'raw'
    if ( staining ):
        figure_path = f'{figure_path}/{modality}/{modality}_refinement/{staining}/'
        prefix = f'{modality}_{staining}'
    else:
        figure_path = f'{figure_path}/{modality}/{modality}_refinement/'

    image_ddf = dd.read_parquet(
        f'{spoqc_tmp_folder}/{prefix}_output_mask_raw',
        columns=[f"{prefix}_beliefs"],
        engine="pyarrow"
    )
    
    beliefs_raw = image_ddf[f"{prefix}_beliefs"].compute().to_numpy()

    # Start the refinement of the proability for the pixel score.
    beliefs, labels = hqr.markov_random_field_zarr_parallel.first_version_loopy_belief_propagation_parallel(
        beliefs_raw.reshape((dim_x, dim_y)),
        spoqc_tmp_folder,
        modality,
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

    # Build the output from fully in-memory arrays and hand it to dask via from_pandas so the resulting ddf has known,
    # sorted divisions. Pairing a freshly chunked dask.array against image_ddf.index (unknown divisions, from a parquet
    # read) preserves index-to-value association but not the physical row order returned by .compute()/round-tripped 
    # through parquet.
    out_df = pd.DataFrame({
        f"{prefix}_beliefs": beliefs_raw,
        f"{prefix}_beliefs_smoothed": beliefs[:].flatten(),
        f"{prefix}_mask_smoothed": labels[:].flatten(),
    })
    n_partitions = max(1, -(-len(out_df) // chunk_size))
    image_ddf = dd.from_pandas(out_df, npartitions=n_partitions)

    helperfuncs.ddf_to_parquet(image_ddf, prefix, spoqc_tmp_folder, [], 'mask_smoothed_raw')

