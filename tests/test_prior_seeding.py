"""The GaussianMixture priors are seeded from the run's seed, independent of numpy's global RNG.

origin/dev fitted them with random_state=None, i.e. on numpy's global random state, so each fit
depended on everything that drew from it before (step order, running a step alone). Seeded, a
fit neither reads nor advances the global state: its result is the same whatever ran before it,
and it leaves the state for later consumers untouched.
"""

import argparse

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from spoqc import helperfuncs
from spoqc.cli_args import build_parser, positive_int
from spoqc.priors.hqcr import negative_probe_counts
from spoqc.priors.hqpr import pixel_score

SEED = 123  # cli.py's run seed
PRIOR_DRAWS = [0, 1, 7, 1000]  # np.random.rand() calls made before the fit


@pytest.fixture(autouse=True)
def no_histograms(monkeypatch):
    monkeypatch.setattr(
        helperfuncs, "plot_histogram_for_array", lambda *args, **kwargs: None
    )


def pixel_scores():
    """100 cluster scores (production fits one score per pixel cluster, 100 clusters) in three
    overlapping groups, so the 3-component fit's k-means initialisation decides the optimum."""
    rng = np.random.default_rng(3)
    return np.concatenate(
        [rng.gamma(2.0, 1.0, 60), rng.normal(4.0, 1.5, 30), rng.normal(7.0, 2.0, 10)]
    )


def negative_probe_frame():
    counts = np.random.default_rng(5).poisson(0.8, 2_000)
    return pd.DataFrame({"control_probe_counts": counts})


def after_draws(draws, fit):
    np.random.seed(0)
    np.random.rand(draws)
    return fit()


PRIORS = {
    "pixel_score": lambda: pixel_score.calc_probs_pixel_score(
        pixel_scores(), None, 3, 6, seed=SEED, n_init=1
    ),
    "negative_probes": lambda: negative_probe_counts.calc_probs(
        negative_probe_frame(), None, seed=SEED, n_init=1
    ),
    # t=None: the fitted mean is used, so the result depends on the fit
    "negative_probes_fitted_mean": lambda: negative_probe_counts.calc_probs(
        negative_probe_frame(), None, 3, 1, None, None, seed=SEED, n_init=1
    ),
    # --gmm_n_init > 1: every start is drawn from the seed, none from the global state
    "pixel_score_n_init_10": lambda: pixel_score.calc_probs_pixel_score(
        pixel_scores(), None, 3, 6, seed=SEED, n_init=10
    ),
}


@pytest.mark.parametrize("name", PRIORS)
def test_prior_does_not_depend_on_earlier_global_draws(name):
    expected = after_draws(0, PRIORS[name])
    for draws in PRIOR_DRAWS[1:]:
        got = after_draws(draws, PRIORS[name])
        assert np.array_equal(got, expected, equal_nan=True), (
            f"{name} changed after {draws} global draws"
        )


@pytest.mark.parametrize("name", PRIORS)
def test_prior_leaves_the_global_state_untouched(name):
    np.random.seed(0)
    np.random.rand(7)
    before = np.random.get_state()
    PRIORS[name]()
    after = np.random.get_state()
    assert (
        after[0] == before[0]
        and np.array_equal(after[1], before[1])
        and after[2:] == before[2:]
    ), f"{name} advanced numpy's global random state"


def test_prior_follows_the_seed():
    """The seed is used: another seed gives another 3-component fit on these scores."""
    scores = pixel_scores()
    same = pixel_score.calc_probs_pixel_score(scores, None, 3, 6, seed=SEED, n_init=1)
    assert np.array_equal(
        pixel_score.calc_probs_pixel_score(scores, None, 3, 6, seed=SEED, n_init=1), same
    )
    others = [
        pixel_score.calc_probs_pixel_score(scores, None, 3, 6, seed=s, n_init=1)
        for s in range(20)
    ]
    assert any(not np.array_equal(o, same) for o in others), (
        "no seed changes the fit: the test data is not init-sensitive"
    )


# --gmm_n_init: the number of k-means starts per GMM prior fit


def multi_optimum_scores():
    """Synthetic cluster scores on which the 3-component fit has three local optima: about half
    of the single k-means starts (seeds 0..29) miss the most likely one."""
    rng = np.random.default_rng(13)
    k = rng.integers(4, 7)
    centers = np.sort(rng.uniform(0, 30, k))
    sizes = rng.integers(5, 40, k)
    return np.concatenate([rng.normal(c, rng.uniform(0.5, 2.5), n) for c, n in zip(centers, sizes)])


STARTS = range(30)
MULTI_PRIORS = {
    "pixel_score": lambda x, seed, n_init: pixel_score.calc_probs_pixel_score(
        x, None, 3, 6, seed=seed, n_init=n_init
    ),
    "negative_probes_fitted_mean": lambda x, seed, n_init: negative_probe_counts.calc_probs(
        pd.DataFrame({"control_probe_counts": x}), None, 3, 1, None, None, seed=seed, n_init=n_init
    ),
}


def single_start_likelihoods(x):
    """The lower bound (mean log-likelihood) of the prior's GMM with one start from each seed."""
    return [
        GaussianMixture(n_components=3, tol=1e-8, max_iter=int(1e4), random_state=s).fit(x.reshape(-1, 1)).lower_bound_
        for s in STARTS
    ]


def origin_pixel_score_prior(scores, nstds, seed):
    """origin/dev db00d98 calc_probs_pixel_score (random_state=None), the global state seeded first."""
    np.random.seed(seed)
    mix = GaussianMixture(n_components=3, tol=1e-8, max_iter=int(1e4))
    mix.fit(scores.reshape(-1, 1))
    stds = [np.sqrt(np.trace(mix.covariances_[i])) for i in range(0, 3)]
    return norm.pdf(scores, loc=np.max(mix.means_), scale=nstds * stds[np.argmax(mix.means_)])


@pytest.mark.parametrize("scores", [pixel_scores, multi_optimum_scores])
def test_default_n_init_is_origin_devs_fit(scores):
    """--gmm_n_init 1 (the default) is origin/dev's single start, seeded like the global state."""
    x = scores()
    got = pixel_score.calc_probs_pixel_score(x, None, 3, 6, seed=SEED, n_init=1)
    assert np.array_equal(got, origin_pixel_score_prior(x, 6, SEED))


@pytest.mark.parametrize("name", MULTI_PRIORS)
def test_single_starts_reach_different_optima(name):
    """The test data is init-sensitive: single starts land on different fits."""
    x = multi_optimum_scores()
    results = [MULTI_PRIORS[name](x, s, 1) for s in STARTS]
    distinct = [r for i, r in enumerate(results) if not any(np.allclose(r, q, rtol=1e-6) for q in results[:i])]
    assert len(distinct) >= 2, f"{name}: every start reaches the same fit"


@pytest.mark.parametrize("name", MULTI_PRIORS)
def test_n_init_keeps_the_most_likely_start(name):
    """With 10 starts, every seed gives the prior of the most likely single-start optimum."""
    x = multi_optimum_scores()
    likelihoods = single_start_likelihoods(x)
    best = MULTI_PRIORS[name](x, int(np.argmax(likelihoods)), 1)
    seeds = range(10)
    assert any(not np.allclose(MULTI_PRIORS[name](x, s, 1), best, rtol=1e-6) for s in seeds), (
        "no single start misses the best optimum: the check below would not discriminate"
    )
    for s in seeds:
        assert np.allclose(MULTI_PRIORS[name](x, s, 10), best, rtol=1e-6), f"{name}: seed {s}, 10 starts"


REQUIRED = ["-i", "in.zarr", "-o", "out", "-t", "tmp"]


def test_cli_gmm_n_init_defaults_to_one():
    assert build_parser().parse_args(REQUIRED).gmm_n_init == 1
    assert build_parser().parse_args(REQUIRED + ["--gmm_n_init", "10"]).gmm_n_init == 10


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "abc", "", "03"])
def test_cli_gmm_n_init_rejects_non_positive_integers(value):
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        positive_int(value)
    with pytest.raises(SystemExit):
        build_parser().parse_args(REQUIRED + ["--gmm_n_init", value])
