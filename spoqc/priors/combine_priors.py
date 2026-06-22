import dask.dataframe as dd
from dask_ml.preprocessing import MinMaxScaler

# We will combine the pixel scorep prior with more priors

def combine_priors_hqpr(spoqc_tmp_folder, image_ddf, belief_name, mask_name):
    image_ddf = image_ddf.rename(
        columns={
            "norm_p_pixel_score": belief_name,
            "pixel_score_mask": mask_name,
        }
    )
    return image_ddf


def combine_priors_hqtr(spoqc_tmp_folder, image_ddf, belief_name, mask_name):
    scaler = MinMaxScaler()

    qv_ddf = dd.read_parquet(f'{spoqc_tmp_folder}/hqtr_output_qv_prob', columns=["norm_p_qv_density"],
                                engine="pyarrow")
    ac_ddf = dd.read_parquet(f'{spoqc_tmp_folder}/hqtr_output_ac_prob', columns=["norm_p_ac_density"],
                                engine="pyarrow")

    # Informative pixel probability for hqtr = p(structure) + p(good qv) + p(not ambient)
    a = image_ddf['norm_p_pixel_score'].to_dask_array(lengths=True)
    b = qv_ddf['norm_p_qv_density'].to_dask_array(lengths=True)
    c = ac_ddf['norm_p_ac_density'].to_dask_array(lengths=True)
    series = a + b + c 
    image_ddf = image_ddf.assign(**{belief_name: series})
    scaled_ddf = scaler.fit_transform(image_ddf[[belief_name]])
    image_ddf = image_ddf.assign(
        **{
            belief_name: scaled_ddf.iloc[:,0],
            mask_name: (scaled_ddf.iloc[:, 0] > 0.5).astype(int),
        }
    )
    return image_ddf

# %%
