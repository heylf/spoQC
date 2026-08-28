import pandas as pd
import numpy as np

from ... import helperfuncs

from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from dask_ml.preprocessing import MinMaxScaler

def calc_probs_pixel_score(pixel_scores, figure_path, gmm_mod=3, nstds=1, t=None, std=None):
    mix = GaussianMixture(n_components=gmm_mod, tol=1e-8, max_iter=int(1e4))
    mix.fit(pixel_scores.reshape(-1, 1))
    means = mix.means_
    cov = mix.covariances_
    stds = [ np.sqrt(np.trace(cov[i])) for i in range(0, gmm_mod) ]
    max_std = stds[np.argmax(means)]

    max_mean = -1
    if ( t ):
        max_mean = t
        max_std = 1.0  # Since mean is hard picked, we will use unit variance.
    else:
        max_mean = np.max(means)

    if ( std ):
        max_std = std

    print(f'Using std {max_std} and mean {max_mean} for pixel prior')

    helperfuncs.plot_histogram_for_array(
        pixel_scores,
        100,
        figure_path,
        f"Pixel scores: t={np.round(max_mean, 3)} with {nstds} x {np.round(max_std, 3)} std",
        "pixel_scores_prior",
        t=max_mean,
        std=max_std,
        nstds=nstds,
    )

    # Calculate the probability density at x for each pixel clusters.
    return norm.pdf(pixel_scores, loc=max_mean, scale=nstds*max_std)