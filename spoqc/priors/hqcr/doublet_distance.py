
# In[]
import pandas as pd
import numpy as np

from scipy.stats import norm

from ... import helperfuncs
from ... import core

def _calc_probs_doublet_distance(sdata, figure_path, nstds = 1.0):
    distances = sdata['table'].obs['doublet_distance']
    max_std = 1.0
    prob_densities = norm.pdf(distances, loc=0.0, scale=nstds*max_std)
    probs = np.array([0.0] * len(prob_densities))

    # If you have no doublets then min_max normalization does not matter.
    if ( len(distances[distances == 100_000]) != len(distances) ):
        print("[NOTE] Doublets are in data, thus normalize probs.")
        probs = helperfuncs.min_max_normalize(prob_densities)

    helperfuncs.plot_histogram_for_array(
        distances,
        100,
        figure_path,
        f"Doublet distance: t=0.0 with {nstds} x {np.round(max_std, 3)} std",
        "doublet_distance_prior",
        t=0.0,
        std=max_std,
        nstds=nstds,
    )

    probs_good_quality = 1 - probs
    return probs_good_quality


def init_prior(enterprise):

    # These have to be defined.
    name = "doublet_prior"
    tmp_path = None
    needs_metrics = ["doublet_score"]

    # These are given by your prior calc function.
    args = [enterprise.cargo.sdata, f'{enterprise.args.output_dir}/hqcr/hqcr_ident/']
    kwargs = {"nstds": enterprise.args.doublet_prior_std}

    prior = core.prior.Prior(
        _calc_probs_doublet_distance, 
        name,
        needs_metrics = needs_metrics,
        tmp_path = tmp_path,
        args = args,
        kwargs = kwargs,
    )    
    
    return prior
