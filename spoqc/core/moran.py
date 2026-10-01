"""Global Moran's I with esda's permutation test, bit-identical and multi-threaded.

`esda.moran.Moran(y, w, permutations=k)` runs its test as a serial Python loop:
`k` times `np.random.permutation(z)` on the global legacy RandomState, then a
sparse lag and a numpy sum. `moran` returns the same Moran object with the same
numbers and leaves the global RNG in the same state, but:

1. one serial pass walks the MT19937 stream exactly as the legacy shuffle consumes
   it (`random_interval` rejection sampling on 32-bit draws) and records only the
   generator state at the start of each permutation;
2. a `prange` over the permutations regenerates each one from its start state,
   applies it, and computes the lag (the same per-row order as scipy's
   `csr_matvec`) and the numpy pairwise sum of `z * lag`, so every statistic is
   rounded exactly as esda rounds it.

The pool is numba's (NUMBA_NUM_THREADS, set by spoqc.core.threads.configure).
"""

from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numba
import numpy as np
from esda.moran import Moran
from libpysal.weights import Queen
from numba import njit, prange
from scipy import stats

# MT19937 constants (numpy/random/src/mt19937)
_N = 624
_M = 397
_MATRIX_A = np.uint32(0x9908B0DF)
_UPPER = np.uint32(0x80000000)
_LOWER = np.uint32(0x7FFFFFFF)
# permutations per pipeline block: the serial walk of block b+1 overlaps the
# parallel statistics of block b
_BLOCK = 64
# numpy's pairwise summation: below this many elements it adds 8 running sums
_PW_BLOCKSIZE = 128


class _Queen(Queen):
    """
    Queen weights that keep their cached sparse matrix and s0/s1/s2 across re-transforms.

    esda's Moran sets `w.transform` on every call. libpysal's setter then reassigns the
    same cached weights dict and drops the cache, so the next `w.s1` rebuilds the
    sparse matrix from Python dicts (~0.8 s on 168k cells), identically each time.
    Setting the transform the weights already have is therefore skipped.
    """

    def set_transform(self, value="B"):
        value = value.upper()
        if value == self._transform and self.weights is self.transformations.get(value):
            return
        Queen.set_transform(self, value)

    transform = property(Queen.get_transform, set_transform)


def queen_weights(xy: np.ndarray):
    """
    Queen contiguity weights of cell centroids, as spoQC builds them.

    Parameters:
        xy (np.ndarray): (n, 2) centroid coordinates.

    Returns:
        libpysal.weights.W: weights over the n points, in input order.
    """
    gdf = gpd.GeoDataFrame({'x': xy[:, 0], 'y': xy[:, 1]},
                           geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]))
    w = Queen.from_dataframe(gdf)
    w.__class__ = _Queen
    return w


@njit(nogil=True, cache=True)
def _mix(upper, lower, far):
    y = (upper & _UPPER) | (lower & _LOWER)
    mixed = far ^ (y >> np.uint32(1))
    if y & np.uint32(1):
        mixed ^= _MATRIX_A
    return np.uint32(mixed)


@njit(nogil=True, cache=True)
def _twist(key):
    """mt19937_gen: regenerate all 624 words."""
    for i in range(_N - _M):
        key[i] = _mix(key[i], key[i + 1], key[i + _M])
    for i in range(_N - _M, _N - 1):
        key[i] = _mix(key[i], key[i + 1], key[i + _M - _N])
    key[_N - 1] = _mix(key[_N - 1], key[0], key[_M - 1])


@njit(nogil=True, cache=True)
def _temper(y):
    y = np.uint32(y ^ (y >> np.uint32(11)))
    y = np.uint32(y ^ ((y << np.uint32(7)) & np.uint32(0x9D2C5680)))
    y = np.uint32(y ^ ((y << np.uint32(15)) & np.uint32(0xEFC60000)))
    return np.uint32(y ^ (y >> np.uint32(18)))


@njit(nogil=True, cache=True)
def _smallest_mask(i):
    """Smallest 2**k - 1 >= i, the mask numpy's random_interval draws under."""
    m = np.uint32(i)
    for s in (1, 2, 4, 8, 16):
        m |= m >> np.uint32(s)
    return m


@njit(nogil=True, cache=True)
def _temper_all(key, tempered):
    for k in range(_N):
        tempered[k] = _temper(key[k])


@njit(nogil=True, cache=True)
def _shuffle(key, tempered, pos, out):
    """
    One legacy `shuffle` from state (key, pos): for i = n-1 .. 1 draw 32-bit words under
    the smallest mask >= i until one is <= i, then swap items i and that word.

    `tempered` holds key's tempered words; key and tempered advance in place.
    Branchless per draw: a rejected draw swaps item i with itself.
    Returns the new pos.
    """
    n = out.shape[0]
    i = n - 1
    if i <= 0:
        return pos
    mask = _smallest_mask(i)
    while True:
        if pos == _N:
            _twist(key)
            _temper_all(key, tempered)
            pos = 0
        v = np.int64(tempered[pos] & mask)
        pos += 1
        accept = v <= i
        j = v if accept else i
        t = out[i]
        out[i] = out[j]
        out[j] = t
        i -= np.int64(accept)
        if i == 0:
            return pos
        if i <= (mask >> np.uint32(1)):
            mask >>= np.uint32(1)


@njit(nogil=True, cache=True)
def _skip_shuffle(key, tempered, pos, n):
    """The draws of one legacy `shuffle` of n items, consumed without moving anything."""
    i = n - 1
    if i <= 0:
        return pos
    mask = _smallest_mask(i)
    while True:
        if pos == _N:
            _twist(key)
            _temper_all(key, tempered)
            pos = 0
        v = np.int64(tempered[pos] & mask)
        pos += 1
        i -= np.int64(v <= i)
        if i == 0:
            return pos
        if i <= (mask >> np.uint32(1)):
            mask >>= np.uint32(1)


@njit(nogil=True, cache=True)
def _permutation_starts(key, pos, n, permutations):
    """Generator state at the start of each of `permutations` shuffles; advances key in place."""
    keys = np.empty((permutations, _N), dtype=np.uint32)
    starts = np.empty(permutations, dtype=np.int64)
    tempered = np.empty(_N, dtype=np.uint32)
    _temper_all(key, tempered)
    for p in range(permutations):
        keys[p] = key
        starts[p] = pos
        pos = _skip_shuffle(key, tempered, pos, n)
    return keys, starts, pos


@njit(nogil=True, cache=True)
def _block_sum(a, lo, n):
    """numpy's pairwise_sum leaf (loops_utils.h.src) for n <= _PW_BLOCKSIZE."""
    if n < 8:
        res = 0.0
        for i in range(lo, lo + n):
            res += a[i]
        return res
    r0 = a[lo]
    r1 = a[lo + 1]
    r2 = a[lo + 2]
    r3 = a[lo + 3]
    r4 = a[lo + 4]
    r5 = a[lo + 5]
    r6 = a[lo + 6]
    r7 = a[lo + 7]
    i = 8
    while i < n - (n % 8):
        r0 += a[lo + i]
        r1 += a[lo + i + 1]
        r2 += a[lo + i + 2]
        r3 += a[lo + i + 3]
        r4 += a[lo + i + 4]
        r5 += a[lo + i + 5]
        r6 += a[lo + i + 6]
        r7 += a[lo + i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[lo + i]
        i += 1
    return res


@njit(nogil=True, cache=True)
def _pairwise_sum(a, lo, n):
    """
    numpy's pairwise summation of a[lo:lo + n], term for term.

    numpy recurses: above _PW_BLOCKSIZE it splits at n2 = n // 2 rounded down to a
    multiple of 8 and returns sum(left) + sum(right). This walks the same tree
    depth-first with an explicit stack (numba segfaults loading cached recursive functions).
    """
    frame_lo = np.empty(64, dtype=np.int64)
    frame_n = np.empty(64, dtype=np.int64)
    frame_stage = np.empty(64, dtype=np.int64)
    partial = np.empty(64, dtype=np.float64)
    top = 0
    frame_lo[0] = lo
    frame_n[0] = n
    frame_stage[0] = 0
    done = 0  # finished sums on the partial stack
    while top >= 0:
        flo = frame_lo[top]
        fn = frame_n[top]
        if fn <= _PW_BLOCKSIZE:
            partial[done] = _block_sum(a, flo, fn)
            done += 1
            top -= 1
            continue
        n2 = fn // 2
        n2 -= n2 % 8
        stage = frame_stage[top]
        if stage == 0:  # left half first
            frame_stage[top] = 1
            top += 1
            frame_lo[top] = flo
            frame_n[top] = n2
            frame_stage[top] = 0
        elif stage == 1:  # then the right half
            frame_stage[top] = 2
            top += 1
            frame_lo[top] = flo + n2
            frame_n[top] = fn - n2
            frame_stage[top] = 0
        else:  # left + right
            partial[done - 2] = partial[done - 2] + partial[done - 1]
            done -= 1
            top -= 1
    return partial[0]


@njit(nogil=True, cache=True)
def _numpy_sum(a):
    """`a.sum()` for a contiguous float64 vector: the add identity plus the pairwise sum."""
    return 0.0 + _pairwise_sum(a, 0, a.shape[0])


@njit(parallel=True, nogil=True, cache=True)
def _permuted_cross_products(z, keys, starts, indptr, indices, data):
    """For each start state: shuffle z, lag it through the CSR weights, return sum(z_p * lag)."""
    n = z.shape[0]
    permutations = starts.shape[0]
    inum = np.empty(permutations, dtype=np.float64)
    for p in prange(permutations):
        key = keys[p].copy()
        tempered = np.empty(_N, dtype=np.uint32)
        _temper_all(key, tempered)
        zp = z.copy()
        _shuffle(key, tempered, starts[p], zp)
        prod = np.empty(n, dtype=np.float64)
        for r in range(n):
            s = 0.0  # scipy csr_matvec: y[r] = 0 + sum over the row's entries, in order
            for jj in range(indptr[r], indptr[r + 1]):
                s += data[jj] * zp[indices[jj]]
            prod[r] = zp[r] * s
        inum[p] = _numpy_sum(prod)
    return inum


def moran(y, w, permutations: int) -> Moran:
    """
    `esda.moran.Moran(y, w, permutations=permutations)` with the permutations in parallel.

    Every attribute is bit-identical to esda's, and the global numpy RNG ends in the
    same state, as if esda had drawn `permutations` times from it.

    Parameters:
        y (array-like): values at the n spatial units.
        w (libpysal.weights.W): weights aligned with y; row-standardised in place, as esda does.
        permutations (int): number of random permutations for the pseudo p-value.

    Returns:
        esda.moran.Moran: the statistic, moments and permutation results.
    """
    m = Moran(y, w, permutations=0)  # esda's own I, moments and transform
    y = np.asarray(y).flatten()
    z = y - y.mean()  # esda's z before it rescales by y.std()
    n = m.n
    if not permutations:
        return m

    _, key, pos, has_gauss, cached_gaussian = np.random.get_state()
    key = key.copy()
    state = [int(pos)]

    def walk(count):  # runs in order on one thread; advances key and state
        keys, starts, state[0] = _permutation_starts(key, state[0], n, count)
        return keys, starts

    z = np.asarray(z, dtype=np.float64)
    sparse = w.sparse
    data = np.asarray(sparse.data, dtype=np.float64)
    inum = np.empty(permutations, dtype=np.float64)
    lows = range(0, permutations, _BLOCK)
    threads = numba.get_num_threads()
    numba.set_num_threads(max(1, threads - 1))  # one core walks
    try:
        with ThreadPoolExecutor(1) as walker:
            blocks = [walker.submit(walk, min(_BLOCK, permutations - lo)) for lo in lows]
            for lo, block in zip(lows, blocks):
                keys, starts = block.result()
                inum[lo:lo + len(starts)] = _permuted_cross_products(
                    z, keys, starts, sparse.indptr, sparse.indices, data)
    finally:
        numba.set_num_threads(threads)
    np.random.set_state(("MT19937", key, state[0], has_gauss, cached_gaussian))

    # esda's __calc and permutation summary, unchanged
    s0 = w.s0
    sim = n / s0 * inum / m.z2ss
    m.permutations = permutations
    m.sim = sim
    above = sim >= m.I
    larger = above.sum()
    if (permutations - larger) < larger:
        larger = permutations - larger
    m.p_sim = (larger + 1.0) / (permutations + 1.0)
    m.EI_sim = sim.sum() / permutations
    m.seI_sim = np.array(sim).std()
    m.VI_sim = m.seI_sim**2
    with np.errstate(divide="ignore"):
        m.z_sim = (m.I - m.EI_sim) / m.seI_sim
    if m.z_sim > 0:
        m.p_z_sim = stats.norm.sf(m.z_sim)
    else:
        m.p_z_sim = stats.norm.cdf(m.z_sim)
    return m
