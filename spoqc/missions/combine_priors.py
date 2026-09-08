import pkgutil
import importlib
import dask.dataframe as dd
import numpy as np

from .. import priors
from .. import core


# We will combine the pixel scorep prior with more priors
def combine_priors_hqcr(enterprise, figure_path):

    enterprise.hqcr_priorset.calculate_metrics()
    final_prior = enterprise.hqcr_priorset.combine_prior_asymmetric_evidence_aggregation()
    traffic_lights = enterprise.hqcr_priorset.combine_prior_traffic_light_system()
    enterprise.cargo.sdata['table'].obs['good_quality_probabilities'] = final_prior
    enterprise.cargo.sdata['table'].obs['hqcr_traffic_light'] = traffic_lights


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
