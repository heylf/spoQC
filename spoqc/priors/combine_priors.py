import dask.dataframe as dd
import numpy as np

from .. import priors

def traffic_light(priors, bad_threshold=0.3, warning_threshold=0.6):
    n_bad = sum(p < bad_threshold for p in priors)
    n_warning = sum(
        bad_threshold <= p < warning_threshold
        for p in priors
    )

    if n_bad >= 2:
        return "red"

    if n_bad == 1 or n_warning >= 2:
        return "yellow"

    return "green"


# Asymetric evidence aggregation will put a penalty on priors that are extremely bad.
# Example A: [.90,.90,.90,.90,.90,.90], result = 0.900
# Example B: [.99,.99,.99,.99,.99,.10], result = 0.391
def asymmetric_evidence_aggregation(priors, gamma=2.0, axis=-1):
    priors = np.asarray(priors, dtype=float)
    priors = np.clip(priors, 1e-12, 1.0)
    surprise = -np.log(priors)
    weighted_surprise = np.mean(surprise ** gamma, axis=axis) ** (1 / gamma)
    return np.exp(-weighted_surprise)


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
    prior_doublet_distance = priors.hqcr.doublet_distance.calc_probs_doublet_distance(sdata, figure_path, nstds=10)
    prior_negative_probe_counts = priors.hqcr.negative_probe_counts.calc_probs(
        cell_df,
        figure_path
    )
    prior_invalid_cell_geometry = priors.hqcr.invalid_geometry.calc_probs(sdata, figure_path, 'cell')
    prior_invalid_nucelus_geometry = priors.hqcr.invalid_geometry.calc_probs(sdata, figure_path, 'nucleus')

    stacked_priors = np.stack(
        [
            prior_transcript_counts,
            prior_gene_counts,
            prior_doublet_distance,
            prior_negative_probe_counts,
            prior_invalid_cell_geometry,
            prior_invalid_nucelus_geometry,
        ],
        axis=1,
    )
    final_prior = asymmetric_evidence_aggregation(stacked_priors, axis=1)

    sdata['table'].obs['good_quality_probabilities'] = final_prior

    traffic_lights = [
        traffic_light(cell_priors)
        for cell_priors in zip(
            prior_transcript_counts,
            prior_gene_counts,
            prior_doublet_distance,
            prior_negative_probe_counts,
            prior_invalid_cell_geometry,
            prior_invalid_nucelus_geometry,
        )
    ]
    sdata['table'].obs['hqcr_traffic_light'] = traffic_lights


def combine_priors_hqpr(spoqc_tmp_folder, image_ddf, belief_name, mask_name):
    image_ddf = image_ddf.rename(
        columns={
            "norm_p_pixel_score": belief_name,
            "pixel_score_mask": mask_name,
        }
    )
    return image_ddf


def combine_priors_hqtr(spoqc_tmp_folder, image_ddf, belief_name, mask_name):
    qv_ddf = dd.read_parquet(
        f"{spoqc_tmp_folder}/hqtr_output_qv_prob",
        columns=["norm_p_qv_density"],
        engine="pyarrow",
        calculate_divisions=True,
    )

    ac_ddf = dd.read_parquet(
        f"{spoqc_tmp_folder}/hqtr_output_ac_prob",
        columns=["norm_p_ac_density"],
        engine="pyarrow",
        calculate_divisions=True,
    )

    # Keep everything lazy / partitioned
    belief = (
        image_ddf["norm_p_pixel_score"]
        + qv_ddf["norm_p_qv_density"]
        + ac_ddf["norm_p_ac_density"]
    )

    image_ddf = image_ddf.assign(**{belief_name: belief})
    num_priors = 3.0
    scaled = image_ddf[belief_name] / num_priors

    return image_ddf.assign(
        **{
            belief_name: scaled,
            mask_name: (scaled > 0.5).astype("int8"),
        }
    )

# %%
