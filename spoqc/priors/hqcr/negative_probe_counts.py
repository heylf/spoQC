import pandas as pd
import numpy as np

from ... import helperfuncs

from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from dask_ml.preprocessing import MinMaxScaler

def calc_probs(df, figure_path, gmm_mod=1, nstds=1, t=1, std=1, tail="right"):
    values = np.array(df["control_probe_counts"])
    mix = GaussianMixture(n_components=gmm_mod, tol=1e-8, max_iter=int(1e4))
    mix.fit(values.reshape(-1, 1))
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

    print(f'Using std {max_std} and mean {max_mean} for pixel prior and tail {tail}')

    helperfuncs.plot_histogram_for_array(
        values,
        20,
        figure_path,
        f"Negative probes: t={np.round(max_mean, 3)} with {nstds} x {np.round(max_std, 3)} std & {tail} tail filtering",
        "negative_probes_prior",
        t=max_mean,
        std=max_std,
        nstds=nstds,
    )

    pdf = norm.pdf(values, loc=max_mean, scale=nstds*max_std)

    # Just a trick, if values are bigger or smaller based on tail then set those values to t and thus get the highest 
    # density for all those values.
    # Later we will inverse, i.e., values < max_mean or values > max_mean will get the worst possible probability.
    if tail == "left":
        pdf = np.where(values < max_mean, np.max(pdf), pdf)
    elif tail == "right":
        pdf = np.where(values > max_mean, np.max(pdf), pdf)

    out = np.max(pdf) - pdf

    # Calculate the probability at x for each pixel clusters.
    return helperfuncs.min_max_normalize(out)