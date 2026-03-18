import numpy as np

def fractal_dimension(Z, threshold):
    """Calculate fractal dimension of a 2D binary image using box-counting."""
    assert(len(Z.shape) == 2)

    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        return len(np.where(S > 0)[0])

    # Convert to binary
    Z = (Z > threshold)

    # Minimal dimension of image
    p = min(Z.shape)
    n = 2 ** np.floor(np.log2(p)).astype(int)
    Z = Z[:n, :n]

    sizes = 2 ** np.arange(int(np.log2(n)), 1, -1)
    counts = [boxcount(Z, size) for size in sizes]

    # Fit line to log-log data
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]