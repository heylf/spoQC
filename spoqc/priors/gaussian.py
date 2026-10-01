"""spoQC's Gaussian prior: the one implementation behind every prior that scores values by a normal density.

Callers: hqcr negative probe counts and doublet distance (per cell), hqpr/hqtr pixel scores (per
pixel cluster), and the hqtr qv/ac densities (per pixel). Each takes the density of its values
under N(mean, scale) (`gaussian_density`) and min-max scales it (`helperfuncs.min_max_normalize`).

The formulation, chosen as the most correct of the four copies it replaces:
- the density is scipy.stats.norm.pdf (hqtr's copy spelled out the formula by hand);
- a tail is set to the peak, norm.pdf(mean), the density's true maximum (the negative-probe copy
  used np.max over its values, which is below the peak when no value sits exactly at the mean);
- min-max scaling is (x - min) / (max - min), exactly 0 and 1 at the extremes (hqpr and hqtr
  used dask_ml MinMaxScaler's x * (1 / range) + (0 - min / range), three roundings).
"""

import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


def gmm_parameters(values, n_components, t=None, std=None, *, seed, n_init):
    """
    (mean, std) of the prior: the mean and std of the highest-mean component of a Gaussian
    mixture fitted to `values`; `t` replaces the mean (and sets the std to 1.0), `std` the std.

    The mixture is always fitted, as it was, with random_state=seed (the run's seed): the fit
    neither reads nor advances numpy's global random state, so it gives the same result in any
    step order and when its step runs alone. n_init (--gmm_n_init, default 1) k-means starts are
    drawn from that seed and the most likely fit is kept; a 3-component fit can have several
    optima, and a single start reaches one of them.
    """
    mix = GaussianMixture(
        n_components=n_components, tol=1e-8, max_iter=int(1e4), n_init=n_init, random_state=seed
    )
    mix.fit(np.asarray(values).reshape(-1, 1))
    means = mix.means_
    cov = mix.covariances_
    stds = [np.sqrt(np.trace(cov[i])) for i in range(0, n_components)]
    max_std = stds[np.argmax(means)]

    if t:
        max_mean = t
        max_std = 1.0  # Since mean is hard picked, we will use unit variance.
    else:
        max_mean = np.max(means)

    if std:
        max_std = std
    return max_mean, max_std


def gaussian_density(values, mean, scale, tail=None, invert=False):
    """
    norm.pdf(values, mean, scale), elementwise (so any slicing of `values` gives the same numbers).

    tail "left" / "right": values below / above the mean get the peak density norm.pdf(mean), i.e.
    count as good as the mean itself. invert: return peak - density instead (0 at the mean, rising
    away from it), for priors where closeness to the threshold is bad.
    """
    density = norm.pdf(values, loc=mean, scale=scale)
    if tail is None and not invert:
        return density
    peak = norm.pdf(mean, loc=mean, scale=scale)
    if tail == "left":
        density = np.where(values < mean, peak, density)
    elif tail == "right":
        density = np.where(values > mean, peak, density)
    elif tail is not None:
        raise ValueError(f"tail must be 'left', 'right' or None, not {tail!r}")
    return peak - density if invert else density
