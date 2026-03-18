
# In[]
import dask.dataframe as dd
import dask.array as da

from .. import helperfuncs
from .. import hqr

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
        chunk_size=10000,
        staining=None
    ):

    prefix = modality
    if ( staining ):
        figure_path = f'{figure_path}/{modality}/{staining}/{modality}_refinement/'
        prefix = f'{modality}_{staining}'
    else:
        figure_path = f'{figure_path}/{modality}/{modality}_refinement/'

    image_ddf = dd.read_parquet(f'{spoqc_tmp_folder}/{prefix}_output_mask_prob',
                                columns=["norm_p_informative_pixel"], engine="pyarrow")

    # Start the refinement of the proability for the pixel score.
    beliefs, labels = hqr.markov_random_field_zarr_parallel.first_version_loopy_belief_propagation_parallel(
        image_ddf["norm_p_informative_pixel"].compute().to_numpy().reshape((dim_x, dim_y)),
        spoqc_tmp_folder,
        modality,
        beta=beta,
        max_iter=max_iter,
        normalize='total'
    )

    hqr.markov_random_field_zarr_parallel.visualize_markov_calculation(
        image_ddf["norm_p_informative_pixel"].compute().to_numpy().reshape((dim_x, dim_y)),
        labels[:],
        figure_path
    )
    
    #  Write out binary mask.
    print(f"[NOTE] Write out binary {prefix} mask")

    # Get index as a dask.array with matching partitioning
    idx_da = image_ddf.index
    beliefs_darr = da.from_array(beliefs[:].flatten(), chunks=chunk_size)
    beliefs_dser = dd.from_dask_array(beliefs_darr, index=idx_da).rename(f'{prefix}_beliefs')
    labels_darr = da.from_array(labels[:].flatten(), chunks=chunk_size)
    labels_dser = dd.from_dask_array(labels_darr, index=idx_da).rename(f'{prefix}_mask')

    image_ddf = image_ddf.assign(
        **{f'{prefix}_beliefs': beliefs_dser},
        **{f'{prefix}_mask': labels_dser}
    )

    helperfuncs.ddf_to_parquet(image_ddf, prefix, spoqc_tmp_folder, [], 'mask_raw')


# %%
