import pkgutil
import importlib
import dask.dataframe as dd
import numpy as np

from .. import priors
from .. import core
from .. import helperfuncs

# We will combine the pixel scorep prior with more priors
def combine_priors_hqcr(enterprise):
    enterprise.hqcr_priorset.calculate_priors_df()
    final_prior = enterprise.hqcr_priorset.combine_prior_asymmetric_evidence_aggregation()
    traffic_lights = enterprise.hqcr_priorset.combine_prior_traffic_light_system()
    enterprise.cargo.sdata['table'].obs['good_quality_probabilities'] = final_prior
    enterprise.cargo.sdata['table'].obs['hqcr_traffic_light'] = traffic_lights


def combine_priors_hqpr(enterprise):
    enterprise.hqpr_priorset.calculate_priors_ddf()
    row_means_series = enterprise.hqpr_priorset.combine_prior_mean_ddf()

    modality = "hqpr"
    belief_name = f"{modality}_{enterprise.args.staining}_beliefs"
    mask_name = f"{modality}_{enterprise.args.staining}_mask"
    tmp_suffix = f"{modality}_{enterprise.args.staining}"
    figure_path = f'{enterprise.args.output_dir}/{modality}/{modality}_clustering/{enterprise.args.staining}/'

    # Preserves chunk size.
    enterprise.hqpr_priorset.prior_ddf = enterprise.hqpr_priorset.prior_ddf.assign(**{
        belief_name: row_means_series,
        mask_name: (row_means_series > 0.5).astype("int8"),
    }).persist()

    helperfuncs.plot_pixels(
        figure_path,
        enterprise.hqpr_priorset.prior_ddf[belief_name].compute().to_numpy().reshape(
            enterprise.cargo.dim_x, enterprise.cargo.dim_y
        ),
        enterprise.cargo.imagedim,
        'beliefs',
        'Combined probability (beliefs)', 
        'hot',
        False,
        False
    )

    print("[NOTE] Writing out data")
    timer = helperfuncs.Timer()
    timer.start()
    helperfuncs.ddf_to_parquet(enterprise.hqpr_priorset.prior_ddf, 'mask_raw', enterprise.args.tmp_dir, [], tmp_suffix)
    timer.stop()

    return enterprise.hqpr_priorset.prior_ddf


def combine_priors_hqtr(enterprise):
    enterprise.hqtr_priorset.calculate_priors_ddf()
    row_means_series = enterprise.hqtr_priorset.combine_prior_mean_ddf()

    modality = "hqtr"
    belief_name = f"{modality}_beliefs"
    mask_name = f"{modality}_mask"
    tmp_suffix = modality
    figure_path = f'{enterprise.args.output_dir}/{modality}/{modality}_clustering/'

    # Preserves chunk size.
    enterprise.hqtr_priorset.prior_ddf = enterprise.hqtr_priorset.prior_ddf.assign(**{
        belief_name: row_means_series,
        mask_name: (row_means_series > 0.5).astype("int8"),
    }).persist()

    helperfuncs.plot_pixels(
        figure_path,
        enterprise.hqtr_priorset.prior_ddf[belief_name].compute().to_numpy().reshape(
            enterprise.cargo.dim_x, enterprise.cargo.dim_y
        ),
        enterprise.cargo.imagedim,
        'beliefs',
        'Combined probability (beliefs)', 
        'hot',
        False,
        False
    )

    print("[NOTE] Writing out data")
    timer = helperfuncs.Timer()
    timer.start()
    helperfuncs.ddf_to_parquet(enterprise.hqtr_priorset.prior_ddf, 'mask_raw', enterprise.args.tmp_dir, [], tmp_suffix)
    timer.stop()

    return enterprise.hqtr_priorset.prior_ddf
# %%
