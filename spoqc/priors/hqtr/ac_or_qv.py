import numpy as np

from ... import helperfuncs
from ...core.threads import map_slices
from .. import gaussian

# elementwise work runs on slices of this many pixels, one per thread task: 2 MB float64
# temporaries, measured 2x faster than 4M-pixel slices on a 64 Mpx image
ROWS_PER_TASK = 1 << 18
# Rows per part file of the prior parquets. origin/dev wrote 10,000-row parts (91,295 files per
# prior on a 913 Mpx slide); measured on a 16 Mpx crop on NFS: 1,600 files 2.4 s -> 16 files 0.35 s
# to write, 3.0 s -> 0.22 s to read, while a part stays 24 MB in memory.
PART_ROWS = 1_000_000


def calc_prob_pixel_stuff_v2(values, figure_path, thresh, std, tail, col, threads):
    """
    The prior of every pixel value in `values` (a 1-D float64 array): the peak of the Gaussian
    density at `thresh` minus the density (the `tail` side counting as the peak), min-max scaled
    to [0, 1] (priors.gaussian; norm_p_{col}).

    Returns (norm_p, part_columns): the norm_p_{col} array, and part_columns(start, stop), the
    columns {col, norm_p_{col}} (what hqtr clustering and the per-cell analysis read) of pixels
    start..stop-1 for core.parquet.write_parts. Elementwise work runs on slices on `threads`
    threads; every value is computed exactly as it is for the whole array.
    """

    if std <= 0:
        raise ValueError("std must be > 0")

    # Because the tailing sets values to the peak, peak - density is 0 there: the extreme case of
    # the worst probability. These are densities, not probabilities, until they are scaled.
    norm_p = np.empty(len(values), dtype=np.float64)

    def density(s):
        norm_p[s] = gaussian.gaussian_density(values[s], thresh, std, tail=tail, invert=True)

    map_slices(density, len(values), ROWS_PER_TASK, threads)
    helperfuncs.min_max_normalize(norm_p, threads, out=norm_p)

    def part_columns(start, stop):
        return {col: values[start:stop], f"norm_p_{col}": norm_p[start:stop]}

    helperfuncs.plot_histogram_for_array(
        values,
        100,
        figure_path,
        f"{col}: t={np.round(thresh, 3)} with {1} x {np.round(std, 3)} std",
        f"{col}_prior",
        t=thresh,
        std=std,
        nstds=1,
    )

    return norm_p, part_columns
