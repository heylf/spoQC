
# In[]
import pandas as pd
import numpy as np

from scipy.stats import norm
from ... import helperfuncs
from sklearn.mixture import GaussianMixture

def calc_probs_doublet_distance(sdata, figure_path, nstds):
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

# ddd = density divided by distance (relative density)
# The closer ddd is to 0 the better the quality.
# The bigger -log10(ddd) is the better the quality.
def calc_probs_ddd(sdata, figure_path, nstds, max_std = 1.0, tail = "right", mean=None):
    ddds = -np.log10(np.array(sdata['table'].obs['doublet_ddd']) + 1e-10)

    # There will be 2 peaks.
    # One peak is the added cosntant for the log10 (1e-10), which is basically cells without a doublet event.
    # The actual peak we are looking for is thus the min of all peaks.
    mix = GaussianMixture(n_components=2, tol=1e-8, max_iter=int(1e4))
    mix.fit(ddds.reshape(-1, 1))
    means = mix.means_
    cov = mix.covariances_

    if mean == None:
        mean = np.min(means)

    pdf = norm.pdf(ddds, loc=mean, scale=nstds*max_std)
    probs = np.array([0.0] * len(pdf))

    # Just a trick, if values are bigger or smaller based on tail then set those values to t and thus get the highest 
    # density for all those values.
    # Here we do not inverse, i.e., values < max_mean or values > max_mean will get the best possible probability.
    if tail == "left":
        pdf = np.where(ddds < mean, np.max(pdf), pdf)
    elif tail == "right":
        pdf = np.where(ddds > mean, np.max(pdf), pdf)

    # If you have no doublets then min_max normalization does not matter.
    if ( len(ddds[ddds == 100_000]) != len(ddds) ):
        print("[NOTE] Doublets are in data, thus normalize probs.")
        probs = helperfuncs.min_max_normalize(pdf)

    helperfuncs.plot_histogram_for_array(
        ddds,
        100,
        figure_path,
        f"""
        Log10+1 Doublet density divided by distance: t={np.round(mean, 3)} with {nstds} x {np.round(max_std, 3)} std 
        & {tail} tail filtering
        """,
        "doublet_ddd_prior",
        t=mean,
        std=max_std,
        nstds=nstds,
    )

    probs_good_quality = probs
    return probs_good_quality