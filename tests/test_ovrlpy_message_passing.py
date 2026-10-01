"""spoqc._ovrlpy_fast._message_passing_parallel vs ovrlpy 1.2.0's _message_passing, verbatim below.

Compared byte for byte (tobytes), so signed zeros and NaN placement count. The inputs hold what the
elevation map holds (about half NaN: pixels without transcripts) plus the values that separate a
faithful kernel from a near miss: -0.0 next to NaN, subnormals, and 1-wide and prime-sized grids
where np.roll's wrap-around meets itself. Each mutant below is a plausible slip and must fail.
"""

from __future__ import annotations

import warnings
from functools import reduce
from operator import add

import numpy as np
import polars as pl
import pytest
from numba import njit, prange

ovrlpy = pytest.importorskip("ovrlpy")

from ovrlpy import _subslicing  # noqa: E402

from spoqc import _ovrlpy_fast as fast  # noqa: E402


def reference(x, /, n_iter):
    """ovrlpy 1.2.0 ovrlpy/_subslicing.py _message_passing, verbatim."""
    with warnings.catch_warnings():
        # ignore 'mean of empty slice' warning if all values are nan
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for _ in range(n_iter):
            x = reduce(
                add,
                (
                    np.nanmean([x, np.roll(x, shift, axis=axis)], axis=0)  # type: ignore
                    for axis in (0, 1)
                    for shift in (1, -1)
                ),
            )
            x /= 4
    return x


def elevation_map(shape, seed, nan_fraction=0.5):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 3.0, shape).astype(np.float32)
    x[rng.random(shape) < nan_fraction] = np.nan
    flat = x.reshape(-1)
    special = rng.choice(flat.size, size=max(1, flat.size // 20), replace=False)
    flat[special[0::3]] = -0.0
    flat[special[1::3]] = np.float32(1e-45) * rng.integers(
        1, 5, len(special[1::3])
    )  # subnormal
    flat[special[2::3]] = -np.float32(1e-45) * rng.integers(1, 5, len(special[2::3]))
    return x


SHAPES = [(1, 1), (1, 7), (7, 1), (2, 2), (3, 5), (31, 17), (64, 97)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("n_iter", [1, 2, 20])
@pytest.mark.parametrize("nan_fraction", [0.0, 0.5, 0.95])
def test_bit_identical_to_ovrlpy(shape, n_iter, nan_fraction):
    x = elevation_map(shape, seed=sum(shape) + n_iter, nan_fraction=nan_fraction)
    before = x.tobytes()
    expected = reference(x.copy(), n_iter)
    actual = fast._message_passing_parallel(x, n_iter)
    assert actual.dtype == expected.dtype == np.float32
    assert actual.tobytes() == expected.tobytes()
    assert x.tobytes() == before, "the input map must not change, as in ovrlpy"


@pytest.mark.parametrize("row", [[np.nan, -0.0], [-0.0, -0.0], [-0.0]])
def test_negative_zero_becomes_positive_zero_as_in_numpy(row):
    """np.sum starts from its identity +0.0, so nanmean(-0.0, -0.0) and nanmean(-0.0, NaN) are +0.0."""
    x = np.array([row], dtype=np.float32)
    expected = reference(x.copy(), 1)
    assert not np.signbit(expected[~np.isnan(expected)]).any()
    assert fast._message_passing_parallel(x, 1).tobytes() == expected.tobytes()


def test_all_nan_map_stays_nan():
    x = np.full((4, 6), np.nan, dtype=np.float32)
    assert (
        fast._message_passing_parallel(x, 3).tobytes()
        == reference(x.copy(), 3).tobytes()
    )


def test_zero_iterations_return_the_input_itself_like_ovrlpy():
    x = elevation_map((5, 5), seed=1)
    assert fast._message_passing_parallel(x, 0) is x
    assert reference(x, 0) is x


@pytest.mark.parametrize("dtype", [np.float64, np.float16, np.int32])
def test_other_dtypes_are_refused(dtype):
    with pytest.raises(TypeError, match="float32 maps only"):
        fast._message_passing_parallel(np.zeros((3, 3), dtype=dtype), 1)


def test_process_coordinates_is_bit_identical_after_install(monkeypatch):
    """The whole ovrlpy entry point: z_center of every transcript, stock vs installed shim."""
    rng = np.random.default_rng(7)
    n = 20_000
    frame = pl.DataFrame(
        {
            "x": rng.uniform(3.0, 180.0, n).astype(np.float32),
            "y": rng.uniform(5.0, 140.0, n).astype(np.float32),
            "z": rng.normal(10.0, 2.0, n).astype(np.float32),
        }
    )
    stock = ovrlpy.process_coordinates(frame, n_iter=20)
    monkeypatch.setattr(_subslicing, "_message_passing", _subslicing._message_passing)
    from ovrlpy import _ovrlp, _utils

    monkeypatch.setattr(_utils, "_calculate_embedding", _utils._calculate_embedding)
    monkeypatch.setattr(_ovrlp, "_calculate_embedding", _ovrlp._calculate_embedding)
    fast.install()
    assert _subslicing._message_passing is fast._message_passing_parallel
    patched = ovrlpy.process_coordinates(frame, n_iter=20)
    assert patched.equals(stock)
    assert (
        patched["z_center"].to_numpy().tobytes()
        == stock["z_center"].to_numpy().tobytes()
    )


# ---------------------------------------------------------------------------
# Mutants: each replaces _message_passing_step and must be caught.
# ---------------------------------------------------------------------------
@njit(fastmath=False, error_model="numpy")
def _nanmean_direct(a, b):
    """Returns the other value when one is NaN, skipping numpy's NaN -> 0.0 replacement."""
    if a != a:
        return b
    if b != b:
        return a
    return np.float32(np.float64(a + b) / 2)


@njit(parallel=True, fastmath=False, error_model="numpy")
def _step_direct_nan(x, out):
    n_rows, n_cols = x.shape
    for i in prange(n_rows):
        up = i - 1 if i > 0 else n_rows - 1
        down = i + 1 if i < n_rows - 1 else 0
        for j in range(n_cols):
            left = j - 1 if j > 0 else n_cols - 1
            right = j + 1 if j < n_cols - 1 else 0
            c = x[i, j]
            t = _nanmean_direct(c, x[up, j]) + _nanmean_direct(c, x[down, j])
            t = t + _nanmean_direct(c, x[i, left])
            t = t + _nanmean_direct(c, x[i, right])
            out[i, j] = t / np.float32(4)


@njit(parallel=True, fastmath=False, error_model="numpy")
def _step_fold_order(x, out):
    """Sums the column neighbours first."""
    n_rows, n_cols = x.shape
    for i in prange(n_rows):
        up = i - 1 if i > 0 else n_rows - 1
        down = i + 1 if i < n_rows - 1 else 0
        for j in range(n_cols):
            left = j - 1 if j > 0 else n_cols - 1
            right = j + 1 if j < n_cols - 1 else 0
            c = x[i, j]
            t = fast._nanmean2(c, x[i, left]) + fast._nanmean2(c, x[i, right])
            t = t + fast._nanmean2(c, x[up, j])
            t = t + fast._nanmean2(c, x[down, j])
            out[i, j] = t / np.float32(4)


@njit(parallel=True, fastmath=False, error_model="numpy")
def _step_no_wrap(x, out):
    """Clamps at the border instead of np.roll's wrap-around."""
    n_rows, n_cols = x.shape
    for i in prange(n_rows):
        up = max(i - 1, 0)
        down = min(i + 1, n_rows - 1)
        for j in range(n_cols):
            left = max(j - 1, 0)
            right = min(j + 1, n_cols - 1)
            c = x[i, j]
            t = fast._nanmean2(c, x[up, j]) + fast._nanmean2(c, x[down, j])
            t = t + fast._nanmean2(c, x[i, left])
            t = t + fast._nanmean2(c, x[i, right])
            out[i, j] = t / np.float32(4)


@njit(parallel=True, fastmath=False, error_model="numpy")
def _step_float64_sum(x, out):
    """Accumulates the four means in float64 and rounds once."""
    n_rows, n_cols = x.shape
    for i in prange(n_rows):
        up = i - 1 if i > 0 else n_rows - 1
        down = i + 1 if i < n_rows - 1 else 0
        for j in range(n_cols):
            left = j - 1 if j > 0 else n_cols - 1
            right = j + 1 if j < n_cols - 1 else 0
            c = x[i, j]
            t = np.float64(fast._nanmean2(c, x[up, j])) + np.float64(
                fast._nanmean2(c, x[down, j])
            )
            t = t + np.float64(fast._nanmean2(c, x[i, left]))
            t = t + np.float64(fast._nanmean2(c, x[i, right]))
            out[i, j] = np.float32(t / 4)


@njit(parallel=True, fastmath=False, error_model="numpy")
def _step_in_place(x, out):
    """Updates the map in place (Gauss-Seidel), so later rows see this iteration's values."""
    n_rows, n_cols = x.shape
    for i in range(n_rows):
        up = i - 1 if i > 0 else n_rows - 1
        down = i + 1 if i < n_rows - 1 else 0
        for j in range(n_cols):
            left = j - 1 if j > 0 else n_cols - 1
            right = j + 1 if j < n_cols - 1 else 0
            c = x[i, j]
            t = fast._nanmean2(c, x[up, j]) + fast._nanmean2(c, x[down, j])
            t = t + fast._nanmean2(c, x[i, left])
            t = t + fast._nanmean2(c, x[i, right])
            x[i, j] = t / np.float32(4)
            out[i, j] = x[i, j]


_step_fastmath = njit(parallel=True, fastmath=True, error_model="numpy")(
    fast._message_passing_step.py_func
)

@njit(fastmath=False, error_model="numpy")
def _nanmean_no_identity(a, b):
    """numpy's NaN -> 0.0 replacement, but a' + b' without np.sum's +0.0 start."""
    count = 0
    if a != a:
        a = np.float32(0)
    else:
        count += 1
    if b != b:
        b = np.float32(0)
    else:
        count += 1
    return np.float32(np.float64(a + b) / count)


@njit(parallel=True, fastmath=False, error_model="numpy")
def _step_no_identity(x, out):
    n_rows, n_cols = x.shape
    for i in prange(n_rows):
        up = i - 1 if i > 0 else n_rows - 1
        down = i + 1 if i < n_rows - 1 else 0
        for j in range(n_cols):
            left = j - 1 if j > 0 else n_cols - 1
            right = j + 1 if j < n_cols - 1 else 0
            c = x[i, j]
            t = _nanmean_no_identity(c, x[up, j]) + _nanmean_no_identity(c, x[down, j])
            t = t + _nanmean_no_identity(c, x[i, left])
            t = t + _nanmean_no_identity(c, x[i, right])
            out[i, j] = t / np.float32(4)


MUTANTS = {
    "sum_without_identity": _step_no_identity,
    "nan_returns_other_value": _step_direct_nan,
    "fold_order": _step_fold_order,
    "no_wrap": _step_no_wrap,
    "float64_accumulation": _step_float64_sum,
    "in_place_update": _step_in_place,
    "fastmath": _step_fastmath,
}


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_mutant_is_caught(monkeypatch, name):
    monkeypatch.setattr(fast, "_message_passing_step", MUTANTS[name])
    maps = [
        elevation_map(shape, seed=seed, nan_fraction=nan_fraction)
        for seed, shape in enumerate([(31, 17), (64, 97), (1, 7), (2, 2)])
        for nan_fraction in (0.0, 0.5, 0.95)
    ]
    maps.append(np.full((3, 4), -0.0, dtype=np.float32))  # signed zeros meeting signed zeros
    caught = [
        fast._message_passing_parallel(x.copy(), n_iter).tobytes() != reference(x.copy(), n_iter).tobytes()
        for x in maps
        for n_iter in (1, 20)
    ]
    assert any(caught), f"mutant {name!r} survived"
