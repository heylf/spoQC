"""A cheaper accumulation for ovrlpy's per-gene embedding, and a parallel z-centre smoothing.

ovrlpy 1.2.0 builds the top/bottom embeddings one gene at a time:

    signal_top = kde_2d_discrete(...)[mask]              # (n_pixels,)  float32
    signal_top = signal_top[:, None] * factor[None, :]   # (n_pixels, n_components)
    embedding_top += top

A full-scale profile (913 Mpx, 42.6M transcripts, py-spy across all threads) put
ovrlpy at **4261 s of the run's 6739 s of CPU -- 63%**, entered from
`doublet_score.py`, and those two multiply lines alone at 1520 s of it.

The accumulator is (n_pixels x n_components) x 4 B (float32, ovrlpy's dtype) --
30 MB for a 500 px patch at n_components=30, far past any L3 -- and ovrlpy allocates a fresh one of those per
gene as a temporary, then touches all of it. `_calculate_embedding_sparse`
updates only the rows a gene is actually nonzero in, which removes the temporary
and most of the traffic. It is bit-identical; see its docstring.

Applied as a shim rather than an edit to site-packages, because a `pip install`
would silently revert the latter. Pinned to the ovrlpy versions whose internals
it reproduces; on any other version `install()` raises, because the pin in
requirements.txt/pyproject.toml has been broken and the replacement is no longer
known to be equivalent.
"""

from __future__ import annotations

from queue import Empty

import numpy as np
from numba import njit, prange

SUPPORTED_OVRLPY_VERSIONS = ("1.2.0",)

_XY_DEFAULT = ("x_pixel", "y_pixel")


def _calculate_embedding_sparse(genes, mask, components, **kwargs):
    """Drop-in replacement for ovrlpy._utils._calculate_embedding, skipping zero rows.

    The accumulation is bound by memory traffic, not arithmetic. With n_components=30 and
    a 500x500 patch the accumulator is 30 x 250,000 x 4 B = **30 MB**, far beyond any L3,
    and ovrlpy touches all of it once per gene per side. At the measured median of 6,492
    genes per patch that is on the order of a terabyte of DRAM traffic for ONE patch,
    which is why neither threads (ovrlpy's own 16: 0.91x) nor processes (4: 1.08x) help.

    But a gene's blurred signal is mostly zero: `kde_2d_discrete` blurs with bandwidth 2.5
    and truncate 4, so a gene is nonzero only within ~10 px of one of its own transcripts.
    Measured by dilating real transcript positions on a 520x520 patch (600 genes, >= 2
    transcripts each):

        nonzero fraction after blur:  median 6.8%   mean 16.0%   p90 44.5%
        below  5%: 43% of genes    below 10%: 56%    below 25%: 78%

    Adding `0.0 * factor_c` is a no-op, so those rows are skipped and the traffic falls
    with the mean nonzero fraction. Finding them costs one pass over `signal`
    (n_pixels x 4 B = 1 MB) against the 60 MB the update itself moves -- about 1%.

    Equivalence: `signal[rows, None] * factor[None, :]` then `+=` is the SAME pair of
    rounding steps as ovrlpy's `signal[:, None] * factor[None, :]` then `+=`, on the same
    values -- no reassociation, no FMA fusion -- so retained rows are exact to the bit.
    A skipped row would have added `0.0 * factor_c`, which is +-0.0 for finite loadings
    and leaves the accumulator unchanged; the only reachable difference is the SIGN of a
    zero in a pixel that is zero for every gene, which compares equal under `==` and
    `np.array_equal` and cannot change `_cosine_similarity`. Non-finite loadings would
    break that argument (`0.0 * inf` is NaN, which ovrlpy propagates and this would not),
    so they are rejected rather than silently handled.

    The accumulator must keep ovrlpy's dtype. ovrlpy fits its PCA on float32 pseudocells,
    so `components` is float32 and ovrlpy sums in float32; an earlier version of this
    shim upcast to float64; the real integrity_map then differed from stock by up to
    6.6e-07 (measured together with a since-dropped process-parallel loop), and a
    synthetic end-to-end run by 3.0e-07 from the upcast alone. The dtype now follows `np.result_type(signal, factor)`, exactly as ovrlpy's product does.
    """
    from ovrlpy._kde import kde_2d_discrete

    x_col, y_col = _XY_DEFAULT
    n_pixels = int(np.count_nonzero(mask))
    n_components = components.shape[0]

    top_acc = None
    bottom_acc = None

    while True:
        try:
            i, gene = genes.get(block=False)
        except Empty:
            break

        # ovrlpy skips genes with fewer than two transcripts in the patch.
        if len(gene) < 2:
            continue

        factor = components[:, i]
        if not np.isfinite(factor).all():
            raise ValueError(
                f"non-finite PCA loading for gene index {i}: skipping zero-signal rows is "
                "only equivalent to ovrlpy for finite loadings, because 0.0 * inf is NaN"
            )

        above = gene.select(_XY_DEFAULT).filter(gene["z"] > gene["z_center"])
        below = gene.select(_XY_DEFAULT).filter(gene["z"] < gene["z_center"])

        for part, which in ((above, "top"), (below, "bottom")):
            if len(part) == 0:
                continue
            signal = kde_2d_discrete(
                part[x_col].to_numpy(), part[y_col].to_numpy(), size=mask.shape, **kwargs
            )[mask]

            rows = np.flatnonzero(signal)
            if rows.size == 0:
                continue

            # The accumulator takes ovrlpy's dtype: signal x factor, as in
            # `signal[:, None] * factor[None, :]`. ovrlpy's PCA is fitted on float32
            # pseudocells, so in production both are float32 and so is the sum.
            acc_dtype = np.result_type(signal.dtype, factor.dtype)
            if which == "top":
                if top_acc is None:
                    top_acc = np.zeros((n_pixels, n_components), dtype=acc_dtype)
                target = top_acc
            else:
                if bottom_acc is None:
                    bottom_acc = np.zeros((n_pixels, n_components), dtype=acc_dtype)
                target = bottom_acc

            target[rows] += signal[rows][:, None] * factor[None, :]

    return (0 if top_acc is None else top_acc, 0 if bottom_acc is None else bottom_acc)


@njit(fastmath=False, error_model="numpy", inline="always")
def _nanmean2(a, b):
    """np.nanmean([a, b], axis=0) for one float32 pixel, in numpy's own steps: NaNs become 0.0
    (_replace_nan); np.sum adds them onto its identity, (0.0 + a') + b' in float32; then
    _divide_by_count's np.divide(tot, cnt), whose float32 / intp operands resolve to the float64
    loop before the result is cast back to float32. 0 / 0 is NaN, numpy's all-NaN result.
    The zeros are load-bearing for the sign of zero: numpy's nanmean of (-0.0, -0.0) and of
    (-0.0, NaN) is +0.0, because -0.0 + 0.0 is +0.0 (tests/test_ovrlpy_message_passing.py)."""
    zero = np.float32(0)
    count = 0
    if a != a:
        a = zero
    else:
        count += 1
    if b != b:
        b = zero
    else:
        count += 1
    return np.float32(np.float64((zero + a) + b) / count)


@njit(parallel=True, fastmath=False, error_model="numpy")
def _message_passing_step(x, out):
    """One iteration of ovrlpy's _message_passing, row-parallel, into `out`:
    ((m(x[i-1,j]) + m(x[i+1,j])) + m(x[i,j-1])) + m(x[i,j+1])) / 4 with m(v) = nanmean(x[i,j], v),
    indices wrapping as np.roll does; the fold order is reduce(add, ...) over (axis 0, shift 1),
    (axis 0, shift -1), (axis 1, shift 1), (axis 1, shift -1)."""
    n_rows, n_cols = x.shape
    four = np.float32(4)
    for i in prange(n_rows):
        up = i - 1 if i > 0 else n_rows - 1  # np.roll(x, 1, axis=0)[i] == x[i - 1]
        down = i + 1 if i < n_rows - 1 else 0  # np.roll(x, -1, axis=0)[i] == x[i + 1]
        for j in range(n_cols):
            left = j - 1 if j > 0 else n_cols - 1
            right = j + 1 if j < n_cols - 1 else 0
            centre = x[i, j]
            total = _nanmean2(centre, x[up, j]) + _nanmean2(centre, x[down, j])
            total = total + _nanmean2(centre, x[i, left])
            total = total + _nanmean2(centre, x[i, right])
            out[i, j] = total / four


def _message_passing_parallel(x, /, n_iter):
    """Drop-in replacement for ovrlpy._subslicing._message_passing on its float32 elevation map.

    ovrlpy runs each iteration as four np.roll copies and four np.nanmean calls over stacked
    (2, n_rows, n_cols) arrays, all on one thread: 57.6 s of serial numpy on breast2's
    7525 x 5470 map (20 iterations; the ~55 s of _replace_nan/_wrapreduction in the Doublet QC
    profile). This computes each output pixel once from its four wrapped neighbours, rows in
    parallel on the numba pool, in the same float32 operations in the same order (see
    _nanmean2 and _message_passing_step), so the map is bit-identical, signed zeros and NaNs
    included. Like ovrlpy, it leaves `x` unchanged and returns x itself when n_iter is 0.
    Raises for any dtype but float32: ovrlpy's default, the one spoQC runs, and the one the
    float64 division step above is written for.
    """
    if x.dtype != np.float32 or x.ndim != 2:
        raise TypeError(
            f"_message_passing_parallel reproduces ovrlpy on 2-D float32 maps only, got {x.ndim}-D {x.dtype}"
        )
    if n_iter < 1:
        return x
    source = np.empty_like(x)
    target = np.empty_like(x)
    _message_passing_step(x, target)
    for _ in range(n_iter - 1):
        source, target = target, source
        _message_passing_step(source, target)
    return target


def install() -> bool:
    """Patch ovrlpy's embedding accumulation; raise if ovrlpy is not a version it reproduces."""
    import ovrlpy
    from ovrlpy import _ovrlp, _subslicing, _utils

    version = getattr(ovrlpy, "__version__", None)
    if version not in SUPPORTED_OVRLPY_VERSIONS:
        raise RuntimeError(
            f"ovrlpy {version} is installed, but spoqc._ovrlpy_fast reproduces the internals "
            f"of ovrlpy {', '.join(SUPPORTED_OVRLPY_VERSIONS)} only (the version pinned in "
            "requirements.txt and pyproject.toml). Install the pinned ovrlpy, or re-verify "
            "_calculate_embedding_sparse against the new version and add it to "
            "SUPPORTED_OVRLPY_VERSIONS."
        )

    _utils._calculate_embedding = _calculate_embedding_sparse
    # _ovrlp imported the symbol directly, so it needs rebinding too.
    _ovrlp._calculate_embedding = _calculate_embedding_sparse
    # _assign_z_mean_message_passing looks _message_passing up in its own module at call time.
    _subslicing._message_passing = _message_passing_parallel
