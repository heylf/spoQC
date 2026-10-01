"""Differential test: the in-RAM MRF against the verbatim tiled original.

tests/reference_mrf_db00d98.py is origin/dev db00d98's
markov_random_field_zarr_parallel.py (only its helperfuncs import is made
absolute). Each case must match bit for bit: beliefs, labels, the full
messages array, dtypes, shapes and the printed iteration/convergence trace.
Run with: pytest -m slow tests/test_markov_random_field_differential.py
"""
import contextlib
import io

import numpy as np
import pytest

import reference_mrf_db00d98 as reference
from spoqc.hqr import markov_random_field_zarr_parallel as mrf

pytestmark = pytest.mark.slow

SHAPES = [(1, 1), (1, 9), (9, 1), (7, 5), (64, 64), (1024, 1024), (1024, 1025),
          (1025, 1024), (1030, 2049), (2049, 3)]
DTYPES = [np.float32, np.float64]
NORMALIZE = ["min", "total"]
KINDS = ["uniform", "blobs", "binary"]
PARAM_SETS = [
    dict(beta=1.5, max_iter=15),
    dict(beta=1.0, max_iter=40, tolerance=1e-3),  # converges on the change tolerance
    dict(beta=0.7, alpha=0.5, max_iter=25, flip_tolerance=1e-2, flip_check=3),  # flip heuristic
]
BIG = 2_000_000  # pixels; larger grids run only the first parameter set


def make_input(shape, dtype, kind, rng):
    if kind == "uniform":
        p = rng.random(shape)
    elif kind == "binary":
        p = (rng.random(shape) > 0.5).astype(float)  # exact 0 and 1 hit the eps path
    else:
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        p = 0.5 + 0.45 * np.sin(yy / 13.0) * np.cos(xx / 7.0) + 0.05 * rng.standard_normal(shape)
        p = np.clip(p, 0, 1)
    return p.astype(dtype)


def trace(stdout):
    return [x for x in stdout.splitlines()
            if "converged" in x or x.startswith("flipping") or x.strip().isdigit()]


def run_reference(p, tmp_path, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        b, l = reference.first_version_loopy_belief_propagation_parallel(p, str(tmp_path), "x", **kw)
    messages = np.fromfile(tmp_path / "lbp_messages_x.mmap", dtype=np.float32)
    messages = messages.reshape(4, p.shape[0] + 2, p.shape[1] + 2, 2)
    return np.asarray(b[:]), np.asarray(l[:]), messages, trace(buf.getvalue())


def run_new(p, monkeypatch, **kw):
    captured = {}
    kernel = mrf.beliefs_and_labels_numba

    def capture(u, up, *rest):
        captured["messages"] = up.base  # every incoming view is a view of the one messages array
        return kernel(u, up, *rest)

    monkeypatch.setattr(mrf, "beliefs_and_labels_numba", capture)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        b, l = mrf.first_version_loopy_belief_propagation_parallel(p, **kw)
    monkeypatch.undo()
    return b, l, captured["messages"], trace(buf.getvalue())


def bits_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a.view(np.uint8), b.view(np.uint8))


@pytest.mark.parametrize("numba_threads", [1, 4], indirect=True)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_new_mrf_is_bit_identical_to_reference(shape, numba_threads, tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n_cases = 0
    for dtype in DTYPES:
        for normalize in NORMALIZE:
            for kind in KINDS:
                p = make_input(shape, dtype, kind, rng)
                param_sets = PARAM_SETS[:1] if p.size > BIG else PARAM_SETS
                for kw in param_sets:
                    case = f"{shape} {dtype.__name__} {normalize} {kind} {kw}"
                    bo, lo, mo, to = run_reference(p, tmp_path, normalize=normalize, **kw)
                    bn, ln, mn, tn = run_new(p, monkeypatch, normalize=normalize, **kw)
                    assert bn.dtype == np.float32 and ln.dtype == np.int8, case
                    assert bits_equal(bo, bn), f"beliefs differ: {case}"
                    assert bits_equal(lo, ln), f"labels differ: {case}"
                    assert bits_equal(mo, mn), f"messages differ: {case}"
                    assert to == tn, f"iteration trace differs: {case}"
                    n_cases += 1
    assert n_cases == (len(DTYPES) * len(NORMALIZE) * len(KINDS)
                       * (1 if shape[0] * shape[1] > BIG else len(PARAM_SETS)))
