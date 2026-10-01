"""Golden-output tests for the loopy belief propagation MRF.

The expected SHA-256 digests were produced by the tiled zarr/memmap version
(origin/dev db00d98, kept verbatim in tests/reference_mrf_db00d98.py) that
this in-RAM version replaces, so they pin the beliefs and labels bit for bit.
(1030, 1100) spans the original's 1024 x 1024 tile boundary in both axes, so
its digests come from the original's multi-tile path.
"""
import hashlib

import numpy as np
import pytest

from spoqc.hqr.markov_random_field_zarr_parallel import (
    first_version_loopy_belief_propagation_parallel as lbp,
)

PARAMS = {
    "total": dict(beta=1.5, max_iter=15),
    "min": dict(beta=1.0, max_iter=20),
}

# (shape, normalize): (beliefs sha256, labels sha256, number of label-1 pixels)
GOLDEN = {
    ((67, 131), 'total'): ('7af9f2fb3278cd52684308786aae96080234eba27e163fdc615d41efc63b34c2', '677d9e07da9d94428dfc21f50347b00a4b801b1ba065a388af22479c978acfbb', 4411),
    ((67, 131), 'min'): ('7e1e4fc7b11385ff365bc03a0eb411b8943f33dc971f45feb47248a0ebe8f61d', '23618efc681d83f1df40039f416bc74c256b97bfb2239ef56c1ace3e54cb754d', 4403),
    ((1, 257), 'total'): ('afe8e89102d9a8cdb40c93d63cecf3fab3094bd7787f0fc8de698293ae677213', '671ac8d430bdf0e9afdd574e1fa83d7fede72672790681a832fd91ed6f4200eb', 135),
    ((1, 257), 'min'): ('13621bbb51b406dd758ecf73c5983d64f7576c4e54a5c9cf3c3b2027d10659b9', 'b11a2ee9a5d1497c5f8db5a7fe335a8ea9ad69f707dc56c06fcad58a9dafda57', 134),
    ((257, 1), 'total'): ('d196e1f76c86ad0a98c4c5100dc796f5ade78a1f60a4cb1df51a99884d4d1784', '4cbeffafa686c877701e8d6b6d40bdc19e433b63bcb64e7a3fab5af3d1e5506a', 131),
    ((257, 1), 'min'): ('8ecbf5c6b3493d268ad766fa81453a024c1bac1d8d827f28d7ad34136334efaa', '8bfcc70f6fc4d65cca7d824ab8fc3aa835c02d877aa90658923dde3bdc2f1c89', 132),
    ((1030, 1100), 'total'): ('772078009ba33a9fbb36a4f88f562682d74c54e3bee970e8e9de8d745863e7a4', '744c84210f038e3b93ce17649022cea156f3a8a884d517df5c81d9c721249b6d', 566685),
    ((1030, 1100), 'min'): ('610fa7c790473151641f6720c5ab25d8e09f39f093287203926ae40b7b1f1503', '40182a74732521831e3b2a13cca066c97282c099c40fa2cf93e86b91c85f9dfb', 566865),
}


def make_prob_map(shape, seed=20260927):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    noisy = 0.5 + 0.45 * np.sin(yy / 5.0) * np.cos(xx / 3.0) + 0.1 * rng.standard_normal(shape)
    return np.clip(noisy, 0, 1)


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


@pytest.mark.parametrize("numba_threads", [1, 2], indirect=True)
@pytest.mark.parametrize("shape, normalize", sorted(GOLDEN))
def test_lbp_matches_golden_output_bit_for_bit(shape, normalize, numba_threads):
    beliefs_sha, labels_sha, n_label1 = GOLDEN[(shape, normalize)]
    prob_map = make_prob_map(shape)
    beliefs, labels = lbp(prob_map, normalize=normalize, **PARAMS[normalize])
    assert beliefs.dtype == np.float32 and beliefs.shape == shape
    assert labels.dtype == np.int8 and labels.shape == shape
    assert int(labels.sum()) == n_label1
    assert _sha(beliefs) == beliefs_sha
    assert _sha(labels) == labels_sha


def test_lbp_float32_and_float64_inputs_agree():
    prob_map = make_prob_map((67, 131))
    b64, l64 = lbp(prob_map, beta=1.5, max_iter=5, normalize="total")
    b32, l32 = lbp(prob_map.astype(np.float32), beta=1.5, max_iter=5, normalize="total")
    assert np.array_equal(b64, b32) and np.array_equal(l64, l32)


def test_lbp_rejects_unknown_normalization():
    with pytest.raises(SystemExit):
        lbp(make_prob_map((7, 5)), normalize="median")
