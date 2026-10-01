import numpy as np

from ... import helperfuncs
from .. import gaussian


def calc_probs(df, figure_path, gmm_mod=1, nstds=1, t=1, std=1, tail="right", *, seed, n_init):
    values = np.array(df["control_probe_counts"])
    max_mean, max_std = gaussian.gmm_parameters(values, gmm_mod, t, std, seed=seed, n_init=n_init)

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

    # Values beyond the mean on the tail side count as good as the mean; the prior falls with
    # the density, from the peak down.
    out = gaussian.gaussian_density(values, max_mean, nstds * max_std, tail=tail, invert=True)
    return helperfuncs.min_max_normalize(out)
