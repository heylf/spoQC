import numpy as np

from ... import helperfuncs
from ... import core

def _calc_probs(sdata, figure_path, obj_type):

    data = np.array(sdata['table'].obs[f'wvalid_{obj_type}_geometry'])

    helperfuncs.plot_histogram_for_array(
        data,
        2,
        figure_path,
        f"Invalid {obj_type} geometries: t=0.0 with 0.0 x {np.round(0.0, 3)} std",
        f"invalid_{obj_type}_gemotry_prior",
        t=0.0
    )

    return 1 - np.array(sdata['table'].obs[f'wvalid_{obj_type}_geometry'])


def init_prior(enterprise):

    # These have to be defined.
    name = "invalid_cell_geometry"
    modality = "hqcr"
    tmp_path = None
    needs_metrics = ["valid_geometries"]

    # These are given by your prior calc function.
    args = [enterprise.cargo.sdata, f"{enterprise.args.output_dir}/hqcr/hqcr_ident/", "cell"]
    kwargs = None

    prior = core.prior.Prior(
        _calc_probs, 
        name,
        modality,
        needs_metrics = needs_metrics,
        tmp_path = tmp_path,
        args = args,
        kwargs = kwargs,
    )    
    
    return prior