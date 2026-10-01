import numpy as np

from ... import helperfuncs
from .. import gaussian


def calc_probs_pixel_score(pixel_scores, figure_path, gmm_mod=3, nstds=1, t=None, std=None, *, seed, n_init):
    max_mean, max_std = gaussian.gmm_parameters(pixel_scores, gmm_mod, t, std, seed=seed, n_init=n_init)

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
    return gaussian.gaussian_density(pixel_scores, max_mean, nstds * max_std)
