
# In[]
import pandas as pd
import numpy as np

from scipy.stats import norm
from ... import helperfuncs

def calc_probs_doublet_distance(sdata, figure_path, nstds=3):
    distances = sdata['table'].obs['doublet_distance']
    max_std = 1.0
    prob_densities = norm.pdf(distances, loc=0.0, scale=nstds*max_std)
    probs = np.array([0.0] * len(prob_densities))

    # If you have no doublets then min_max normalization does not matter.
    if ( len(distances[distances == 100_000]) != len(distances) ):
        probs = helperfuncs.min_max_normalize(prob_densities)

    helperfuncs.plot_histogram_for_array(
        distances,
        20,
        figure_path,
        f"Doublet distance: t=0.0 with {nstds} x {np.round(max_std, 3)} std",
        "doublet_distance_prior",
        t=0.0
    )

    probs_good_quality = 1 - probs
    return probs_good_quality