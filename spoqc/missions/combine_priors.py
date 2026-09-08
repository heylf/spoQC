import pkgutil
import importlib
import dask.dataframe as dd
import numpy as np

from .. import priors
from .. import core

def _generate_prior_set(name_priorset):
    priorset_list = []
    for module_info in pkgutil.iter_modules(priors.segmentation.__path__):
        module_name = module_info.name
        full_name = f"{priors.segmentation.__name__}.{module_name}"
        module = importlib.import_module(full_name)

        if hasattr(module, "init_prior"):
            priorset_list.append(module.init_prior())
            print(f"Loaded prior: {module_name}")
        else:
            print(f"WARNING: {module_name} has no init_prior() function")

    priorset = core.prior.PriorSet(name_priorset, priorset_list)
    priorset.calculate_metrics()
    return priorset


# We will combine the pixel scorep prior with more priors
def combine_priors_hqcr(sdata, figure_path, cell_df, qc_domains_adata, counts, doublet_prior_std):

    priorset = _generate_prior_set('hqcr')
    final_prior = priorset.combine_prior_asymmetric_evidence_aggregation()
    traffic_lights = priorset.combine_prior_traffic_light_system()
    sdata['table'].obs['good_quality_probabilities'] = final_prior
    sdata['table'].obs['hqcr_traffic_light'] = traffic_lights
    




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
    prior_doublet_distance = priors.hqcr.doublet_distance.calc_probs_doublet_distance(sdata, figure_path, doublet_prior_std)
    prior_negative_probe_counts = priors.hqcr.negative_probe_counts.calc_probs(
        cell_df,
        figure_path
    )
    prior_invalid_cell_geometry = priors.hqcr.invalid_geometry.calc_probs(sdata, figure_path, 'cell')
    prior_invalid_nucelus_geometry = priors.hqcr.invalid_geometry.calc_probs(sdata, figure_path, 'nucleus')



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
