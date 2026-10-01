# In[]
import numpy as np

from ... import helperfuncs
from .. import gaussian


def calc_probs_doublet_distance(sdata, figure_path, nstds):
    distances = sdata['table'].obs['doublet_distance']
    max_std = 1.0
    prob_densities = gaussian.gaussian_density(np.asarray(distances), 0.0, nstds * max_std)
    # Without doublets (every distance 100,000) all densities are equal and scale to 0.
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
