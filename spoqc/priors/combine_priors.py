import dask.dataframe as dd
import numpy as np
import pandas as pd

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
def combine_priors_hqcr(sdata, figure_path, cell_df, qc_domains_adata, counts, doublet_prior_std, threads, seed, gmm_n_init):

    prior_transcript_counts, cell_df = priors.hqcr.transcript_and_gene_counts.calc_counts_probs(
        sdata, 
        figure_path,
        cell_df,
        qc_domains_adata,
        counts,
        1.0,
        threads,
    )
    prior_gene_counts, cell_df = priors.hqcr.transcript_and_gene_counts.calc_counts_probs(
        sdata, 
        figure_path,
        cell_df,
        qc_domains_adata,
        "n_genes_by_counts",
        0.5,
        threads,
    )
    prior_doublet_distance = priors.hqcr.doublet_distance.calc_probs_doublet_distance(sdata, figure_path, doublet_prior_std)
    prior_negative_probe_counts = priors.hqcr.negative_probe_counts.calc_probs(
        cell_df,
        figure_path,
        seed=seed,
        n_init=gmm_n_init,
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


def combine_priors_hqpr(norm_p_pixel_score, pixel_score_mask, belief_name, mask_name):
    return {belief_name: norm_p_pixel_score, mask_name: pixel_score_mask}


def read_pixel_prior(path, column, n_rows):
    # Read a per-pixel prior written by core.parquet.write_parts; its index must be the pixel order 0..n-1.
    series = dd.read_parquet(path, columns=[column], engine="pyarrow")[column].compute()
    if not series.index.equals(pd.RangeIndex(n_rows)):
        raise ValueError(f"{path} is not indexed by pixel 0..{n_rows - 1}")
    return series.to_numpy()


def combine_priors_hqtr(spoqc_tmp_folder, norm_p_pixel_score, pixel_score_mask, belief_name, mask_name):
    n_rows = len(norm_p_pixel_score)
    qv = read_pixel_prior(f"{spoqc_tmp_folder}/hqtr_output_qv_prob", "norm_p_qv_density", n_rows)
    ac = read_pixel_prior(f"{spoqc_tmp_folder}/hqtr_output_ac_prob", "norm_p_ac_density", n_rows)
    num_priors = 3.0
    scaled = (norm_p_pixel_score + qv + ac) / num_priors

    return {
        "norm_p_pixel_score": norm_p_pixel_score,
        "pixel_score_mask": pixel_score_mask,
        belief_name: scaled,
        mask_name: (scaled > 0.5).astype("int8"),
    }

# %%
