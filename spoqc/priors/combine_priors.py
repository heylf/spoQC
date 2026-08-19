import pandas as pd
import dask.dataframe as dd
from dask_ml.preprocessing import MinMaxScaler

from .. import priors
from .. import helperfuncs

# We will combine the pixel scorep prior with more priors
def combine_priors_hqcr(sdata, figure_path, cell_df, qc_domains_adata, counts):

    prior_transcript_counts, cell_df = priors.hqcr.transcript_and_gene_counts.calc_counts_probs(
        sdata, 
        figure_path,
        cell_df,
        qc_domains_adata,
        counts,
        1.0,
    )
    prior_gene_counts, cell_df = priors.hqcr.transcript_and_gene_counts.calc_counts_probs(
        sdata, 
        figure_path,
        cell_df,
        qc_domains_adata,
        "n_genes_by_counts",
        0.5,
    )
    prior_doublet_distance = priors.hqcr.doublet_distance.calc_probs_doublet_distance(sdata, figure_path, nstds=100)
    prior_negative_probe_counts = priors.hqcr.negative_probe_counts.calc_probs(
        cell_df,
        figure_path
    )
    prior_invalid_cell_geometry = priors.hqcr.invalid_geometry.calc_probs(sdata, figure_path, 'cell')
    prior_invalid_nucelus_geometry = priors.hqcr.invalid_geometry.calc_probs(sdata, figure_path, 'nucleus')

    final_prior = \
        prior_transcript_counts + \
        prior_gene_counts + \
        prior_doublet_distance + \
        prior_negative_probe_counts + \
        prior_invalid_cell_geometry + \
        prior_invalid_nucelus_geometry
    final_prior = helperfuncs.min_max_normalize(final_prior)

    sdata['table'].obs['good_quality_probabilities'] = final_prior


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

    qv_ddf = dd.read_parquet(
        f'{spoqc_tmp_folder}/hqtr_output_qv_prob',
        columns=["norm_p_qv_density"],
        engine="pyarrow",
    )
    ac_ddf = dd.read_parquet(
        f'{spoqc_tmp_folder}/hqtr_output_ac_prob',
        columns=["norm_p_ac_density"],
        engine="pyarrow",
    )

    # Informative pixel probability for hqtr = p(structure) + p(good qv) + p(not ambient)
    a = image_ddf['norm_p_pixel_score'].compute().values
    b = qv_ddf['norm_p_qv_density'].compute().values
    c = ac_ddf['norm_p_ac_density'].compute().values
    series_np = a + b + c

    chunk_size = image_ddf.divisions[1] - image_ddf.divisions[0]
    series_ddf = dd.from_pandas(pd.Series(series_np), chunksize=chunk_size)
    image_ddf = image_ddf.assign(**{belief_name: series_ddf})
    scaled_ddf = scaler.fit_transform(image_ddf[[belief_name]])
    image_ddf = image_ddf.assign(
        **{
            belief_name: scaled_ddf.iloc[:,0],
            mask_name: (scaled_ddf.iloc[:, 0] > 0.5).astype(int),
        }
    )
    return image_ddf

# %%
