"""priors.gaussian, the one Gaussian prior, against the four origin/dev copies it replaces.

The copies are in tests/legacy/ (negative_probe_counts, doublet_distance, pixel_score_prior,
ac_or_qv). Where the merged formulation equals a copy, the test asserts bit identity; where it
differs (docs/perf/hqtr_ambient.md), the test pins the size and place of the difference.
"""

import types

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from conftest import assert_same_array, load_legacy

from spoqc import helperfuncs
from spoqc.priors import gaussian
from spoqc.priors.hqcr import doublet_distance, negative_probe_counts
from spoqc.priors.hqpr import pixel_score
from spoqc.priors.hqtr import ac_or_qv

EPS = np.finfo(np.float64).eps


def origin_min_max_normalize(array):
    """helperfuncs.min_max_normalize at origin/dev."""
    array = np.array(array)
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def legacy(name, package):
    module = load_legacy(name, package)
    module.helperfuncs = types.SimpleNamespace(
        min_max_normalize=origin_min_max_normalize,
        plot_histogram_for_array=lambda *args, **kwargs: None,
    )
    return module


@pytest.fixture(autouse=True)
def no_histograms(monkeypatch):
    monkeypatch.setattr(
        helperfuncs, "plot_histogram_for_array", lambda *args, **kwargs: None
    )


class TestMinMax:
    def test_is_x_minus_min_over_range(self):
        values = np.random.default_rng(0).gamma(2.0, 3.0, 10_001)
        assert_same_array(
            helperfuncs.min_max_normalize(values),
            origin_min_max_normalize(values),
            "min-max",
        )

    @pytest.mark.parametrize("workers", [1, 2, 5])
    def test_same_bits_for_any_split(self, workers):
        values = np.random.default_rng(1).normal(0, 1, 700_001)
        assert_same_array(
            helperfuncs.min_max_normalize(values, workers),
            origin_min_max_normalize(values),
            "split",
        )

    def test_constant_scales_to_zero_and_nan_stays(self):
        values = np.array([2.5, np.nan, 2.5])
        got = helperfuncs.min_max_normalize(values)
        assert got[0] == 0 and got[2] == 0 and np.isnan(got[1])

    def test_in_place(self):
        values = np.arange(5.0)
        assert helperfuncs.min_max_normalize(values, out=values) is values
        assert values.tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]


class TestDensity:
    def test_is_scipy_norm_pdf(self):
        from scipy.stats import norm

        values = np.linspace(-5, 30, 1_001)
        assert_same_array(
            gaussian.gaussian_density(values, 20.0, 3.0),
            norm.pdf(values, loc=20.0, scale=3.0),
            "pdf",
        )

    @pytest.mark.parametrize("tail", ["left", "right", None])
    def test_same_numbers_on_slices(self, tail):
        values = np.random.default_rng(2).gamma(2.0, 10.0, 9_999)
        whole = gaussian.gaussian_density(values, 20.0, 3.0, tail=tail, invert=True)
        parts = np.concatenate(
            [
                gaussian.gaussian_density(
                    values[i : i + 1_000], 20.0, 3.0, tail=tail, invert=True
                )
                for i in range(0, len(values), 1_000)
            ]
        )
        assert_same_array(parts, whole, "slices")

    def test_tail_side_is_the_peak(self):
        got = gaussian.gaussian_density(
            np.array([-1.0, 0.5, 3.0]), 1.0, 1.0, tail="right", invert=True
        )
        assert got[2] == 0.0 and got[1] < got[0]

    def test_unknown_tail_raises(self):
        with pytest.raises(ValueError, match="tail"):
            gaussian.gaussian_density(np.zeros(3), 0.0, 1.0, tail="both")


class TestAgainstOrigin:
    def test_pixel_score_density_is_bit_identical(self):
        """hqpr/hqtr: with origin/dev's global random state seeded to the prior's seed, the GMM
        parameters and the density are unchanged; the seeded prior leaves the global state alone."""
        scores = np.random.default_rng(3).gamma(2.0, 1.0, 100)
        np.random.seed(5)
        expected = legacy("pixel_score_prior", "spoqc.priors.hqpr").calc_probs_pixel_score(scores, None, 3, 6)
        np.random.seed(0)
        untouched = np.random.random()
        np.random.seed(0)
        got = pixel_score.calc_probs_pixel_score(scores, None, 3, 6, seed=5, n_init=1)
        assert_same_array(got, expected, "pixel score density")
        assert np.random.random() == untouched  # the seeded fit draws nothing from the global state

    @pytest.mark.parametrize("all_singlets", [False, True])
    def test_doublet_distance_is_bit_identical(self, all_singlets):
        distances = np.random.default_rng(4).uniform(0, 60, 500)
        distances[::3] = 100_000
        if all_singlets:
            distances[:] = 100_000
        sdata = {"table": types.SimpleNamespace(obs=pd.DataFrame({"doublet_distance": distances}))}
        expected = legacy("doublet_distance", "spoqc.priors.hqcr").calc_probs_doublet_distance(sdata, None, 100)
        got = doublet_distance.calc_probs_doublet_distance(sdata, None, 100)
        assert_same_array(got, np.asarray(expected, dtype=np.float64), "doublet prior")

    def test_negative_probes_bit_identical_when_a_cell_sits_at_the_mean(self):
        counts = np.random.default_rng(5).poisson(0.8, 2_000)
        assert (counts == 1).any()
        df = pd.DataFrame({"control_probe_counts": counts})
        np.random.seed(6)
        expected = legacy("negative_probe_counts", "spoqc.priors.hqcr").calc_probs(df, None)
        assert_same_array(negative_probe_counts.calc_probs(df, None, seed=6, n_init=1), expected, "negative probes")

    def test_negative_probes_without_a_cell_at_the_mean(self):
        """origin/dev took the peak as the largest density among the cells. With no cell at t = 1
        probe, the tail cells (> 1) and the cells at 0 (as far from 1) all got that value, every
        prior was 0 and min-max gave 0 / 0 = NaN. The true peak keeps the cells at 0 below it."""
        counts = np.array([0, 0, 2, 3, 5, 0, 7])
        df = pd.DataFrame({"control_probe_counts": counts})
        np.random.seed(6)
        with np.errstate(invalid="ignore"):
            expected = legacy("negative_probe_counts", "spoqc.priors.hqcr").calc_probs(df, None)
        got = negative_probe_counts.calc_probs(df, None, seed=6, n_init=1)
        assert np.isnan(expected).all()
        assert got.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    @pytest.mark.parametrize("thresh,std,tail", [(20.0, 3, "left"), (0.4, 1, "left"), (0.4, 1, "right")])
    def test_qv_ac_prior_within_a_few_ulp_of_origin(self, tmp_path, thresh, std, tail):
        rng = np.random.default_rng(7)
        values = np.where(rng.random(50_000) < 0.4, 0.0, rng.gamma(2.0, thresh, 50_000))
        old = legacy("ac_or_qv", "spoqc.priors.hqtr")
        image_ddf = dd.from_dask_array(da.from_array(values, chunks=10_000), columns=["x"])
        expected = old.calc_prob_pixel_stuff_v2(image_ddf, str(tmp_path), thresh, std, tail, "x")["norm_p_x"].compute().to_numpy()
        got, _ = ac_or_qv.calc_prob_pixel_stuff_v2(values, str(tmp_path), thresh, std, tail, "x", 3)
        # The density formula (scipy vs by hand) and the min-max formula differ in the last bits,
        # scaled up by 1 / (range of the densities); measured here: at most 1.9e-15.
        assert np.max(np.abs(got - expected)) <= 16 * EPS
        assert got.min() == 0.0 and got.max() == 1.0
