
# In[]
import pandas as pd
import numpy as np

from scipy.stats import norm
from ... import helperfuncs

def calc_probs_doublet_distance(sdata, nstds=None):
    distances = sdata.table.obs['doublet_distance']
    max_std = 1.0
    set_nstds = 3.0
    if ( nstds ):
        set_nstds = nstds
    prob_densities = norm.pdf(distances, loc=0.0, scale=set_nstds*max_std)
    probs = np.array([0.0] * len(prob_densities))

    # If you have no doublets then min_max normalization does not matter.
    if ( len(distances[distances == 100_000]) != len(distances) ):
        probs = helperfuncs.min_max_normalize(prob_densities)

    probs_good_quality = 1 - probs
    return probs_good_quality