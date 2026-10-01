"""spoqc.core.groupreduce vs explicit per-group loops."""

import numpy as np
import pytest

from spoqc.core import groupreduce


@pytest.fixture
def ragged():
    rng = np.random.default_rng(0)
    sizes = rng.integers(0, 40, 300)
    sizes[:5] = [0, 1, 8, 9, 129]
    group_idx = np.repeat(np.arange(len(sizes)), sizes)
    return rng.uniform(0, 1, len(group_idx)), group_idx, sizes


def test_group_offsets_bound_each_group(ragged):
    _, group_idx, sizes = ragged
    offsets = groupreduce.group_offsets(group_idx, len(sizes))
    np.testing.assert_array_equal(np.diff(offsets), sizes)


def test_cyclic_shift_wraps_within_group():
    offsets = np.array([0, 3, 3, 7])
    np.testing.assert_array_equal(groupreduce.cyclic_shift(offsets, 1), [1, 2, 0, 4, 5, 6, 3])
    np.testing.assert_array_equal(groupreduce.cyclic_shift(offsets, 2), [2, 0, 1, 5, 6, 3, 4])


def test_group_count_counts_masked_elements(ragged):
    values, group_idx, sizes = ragged
    counts = groupreduce.group_count(group_idx, values > 0.5, len(sizes))
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    assert counts.tolist() == [int((values[a:b] > 0.5).sum()) for a, b in zip(offsets[:-1], offsets[1:])]


@pytest.mark.parametrize('reducer', [np.mean, np.min, np.max, np.sum])
def test_ragged_reduce_is_bit_identical_to_per_group_reduction(ragged, reducer):
    values, group_idx, sizes = ragged
    offsets = groupreduce.group_offsets(group_idx, len(sizes))
    actual = groupreduce.ragged_reduce(values, offsets, reducer, np.full(len(sizes), -1.0))
    expected = np.array([reducer(list(values[a:b])) if b > a else -1.0 for a, b in zip(offsets[:-1], offsets[1:])])
    assert actual.tobytes() == expected.tobytes()


def test_ragged_lists_splits_per_group():
    assert groupreduce.ragged_lists(['a', 'b', 'c'], np.array([0, 0, 2, 3])) == [[], ['a', 'b'], ['c']]
