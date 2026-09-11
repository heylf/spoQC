
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
        chunk_size,
        *,
        beta=1.5,
        max_iter=15,
        staining=None
    ):

    suffix = modality
    prefix = 'raw'
    if ( staining ):
        figure_path = f'{figure_path}/{modality}/{modality}_refinement/{staining}/'
        suffix = f'{modality}_{staining}'
    else:
        figure_path = f'{figure_path}/{modality}/{modality}_refinement/'

    image_ddf = dd.read_parquet(
        f'{spoqc_tmp_folder}/mask_{prefix}_output_{suffix}',
        columns=[f"{suffix}_beliefs"],
        engine="pyarrow"
    )
    
    beliefs_raw = image_ddf[f"{suffix}_beliefs"].compute().to_numpy()

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
    print(f"[NOTE] Write out {suffix} masks")

    # Build the output from fully in-memory arrays and hand it to dask via from_pandas so the resulting ddf has known,
    # sorted divisions. Pairing a freshly chunked dask.array against image_ddf.index (unknown divisions, from a parquet
    # read) preserves index-to-value association but not the physical row order returned by .compute()/round-tripped 
    # through parquet.
    out_df = pd.DataFrame({
        f"{suffix}_beliefs": beliefs_raw,
        f"{suffix}_beliefs_smoothed": beliefs[:].flatten(),
        f"{suffix}_mask_smoothed": labels[:].flatten(),
    })
    n_partitions = max(1, -(-len(out_df) // chunk_size))
    image_ddf = dd.from_pandas(out_df, npartitions=n_partitions)

    helperfuncs.ddf_to_parquet(image_ddf, f'mask_smoothed_{prefix}', spoqc_tmp_folder, [], suffix)

