"""Window-histogram texture metrics: entropy, uniformity and homogeneity from one pass.

Each pixel's (2r+1) x (2r+1) window histogram is kept as a sorted buffer of the window's
values, updated incrementally as the window slides along a row (drop one column, insert one),
and all three metrics are derived from it. This is bit-identical to the original per-window
kernels (np.bincount per window, then entropy / kl_divergence_uniform / homogeneity):
- the original sums run over histogram bins in ascending value order; the sorted buffer
  visits the distinct values in the same order, and the bins it skips hold a count of 0,
  whose homogeneity term is +0.0 (adding it changes nothing) and which entropy / KL drop;
- a bin's probability is count / size with count in 1..size, so the entropy and KL terms
  take one of `size` values each, precomputed with the original expressions;
- results are stored as float32, and uniformity is the negated float32 KL, as before.
"""

import numpy as np
from numba import njit, prange

type Array2D[T: np.generic] = np.ndarray[tuple[int, int], np.dtype[T]]


@njit
def _occurrence_terms(size, q):
    """Entropy and KL-against-uniform terms of a bin holding `c` of `size` window values."""
    entropy_term = np.empty(size + 1)
    kl_term = np.empty(size + 1)
    for c in range(1, size + 1):
        p = c / size  # occurrence probability
        entropy_term[c] = p * np.log(p)
        kl_term[c] = -(p * np.log(p / q))
    return entropy_term, kl_term


@njit(parallel=True)
def _texture_windows(
    x_padded: Array2D, r: int, entropy: Array2D, kl: Array2D, homogeneity: Array2D
):
    s = 2 * r + 1
    size = s * s
    m, n = entropy.shape
    # uniform distribution: probability for each element
    # if there are less observations than potential levels truncate
    q = max(1 / size, 1 / (np.iinfo(x_padded.dtype).max + 1))
    entropy_term, kl_term = _occurrence_terms(size, q)
    for i in prange(m):
        window = np.empty(size, dtype=np.int64)  # the window's values, sorted ascending
        filled = 0
        for a in range(s):
            for b in range(s):
                value = np.int64(x_padded[i + a, b])
                t = filled
                while t > 0 and window[t - 1] > value:
                    window[t] = window[t - 1]
                    t -= 1
                window[t] = value
                filled += 1
        for j in range(n):
            if j > 0:  # slide right: drop column j - 1, insert column j + s - 1
                for a in range(s):
                    old = np.int64(x_padded[i + a, j - 1])
                    t = 0
                    while window[t] != old:
                        t += 1
                    while t < size - 1:
                        window[t] = window[t + 1]
                        t += 1
                    value = np.int64(x_padded[i + a, j + s - 1])
                    t = size - 1
                    while t > 0 and window[t - 1] > value:
                        window[t] = window[t - 1]
                        t -= 1
                    window[t] = value
            v = np.int64(x_padded[i + r, j + r])  # central value
            entropy_sum = 0.0
            kl_sum = 0.0
            homogeneity_sum = 0.0
            t = 0
            while t < size:  # one step per distinct value, ascending
                u = t
                while u < size and window[u] == window[t]:
                    u += 1
                count = u - t
                entropy_sum += entropy_term[count]
                kl_sum += kl_term[count]
                # homogeneity: the central pixel is removed from its own bin; a large absolute
                # difference to the central value lowers homogeneity; all values equal gives 1.
                centred = count - 1 if window[t] == v else count
                homogeneity_sum += (centred / (size - 1)) / (np.fabs(window[t] - v) + 1)
                t = u
            entropy[i, j] = -entropy_sum
            kl[i, j] = kl_sum
            homogeneity[i, j] = homogeneity_sum


def texture_metrics(
    img: Array2D, window_size: int, mode: str = "reflect"
) -> tuple[Array2D, Array2D, Array2D]:
    """Pixel entropy, uniformity and homogeneity (float32, shape of `img`) over odd `window_size` windows.

    `img` is a 2D array of non-negative integers; it is padded by `mode` (see numpy.pad).
    Uniformity is the negated KL divergence against a uniform distribution.
    Threads: numba's pool, set from CONST.THREADS in cli.py.
    """
    r = (window_size - 1) // 2
    x_padded = np.pad(img, pad_width=r, mode=mode)
    entropy, kl, homogeneity = (np.empty(img.shape, dtype=np.float32) for _ in range(3))
    _texture_windows(x_padded, r, entropy, kl, homogeneity)
    return entropy, -kl, homogeneity
