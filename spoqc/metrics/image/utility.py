import numba
import numpy as np
import plotly.express as px

from ... import helperfuncs
from spoqc.core.figures import save_figure

def plot_intensity_histogram(figure_path, background_intensity, hist, bin_edges):

    timer = helperfuncs.Timer()

    figures = []

    # When you plot a histogram via plotly, it stores all the orginal data in the json file 
    # and makes the bins and counts on the javascript side. 
    # Thus the plot get quite large.
    # Use therefore the precomupted histogram data from numpy.
    bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    print("[NOTE] Barplot")
    timer.start()
    fig = px.bar(x=bins, y=hist, labels={'x':'intensity', 'y':'count'})
    fig.update_layout(
        title=f"Total distribution intensity with backkground intensity {background_intensity}"
    )
    timer.stop()
    helperfuncs.apply_general_plotly_layout(fig, True)
    figures.append(fig)
    save_figure(fig, f"{figure_path}/histogram_intensity.png", f"{figure_path}/histogram_intensity.pdf", scale=3)

    with open(f'{figure_path}/histogram_intensity.html', 'w') as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


UINT16_VALUES = 1 << 16


@numba.njit(parallel=True)
def _uint16_counts(image, n_threads):
    """Exact np.bincount(image.ravel(), minlength=65536) of a 2-D uint16 image, one partial count per thread."""
    n_rows, n_cols = image.shape
    partial = np.zeros((n_threads, UINT16_VALUES), dtype=np.int64)
    for t in numba.prange(n_threads):
        for i in range(t * n_rows // n_threads, (t + 1) * n_rows // n_threads):
            for j in range(n_cols):
                partial[t, image[i, j]] += 1
    counts = np.zeros(UINT16_VALUES, dtype=np.int64)
    for t in range(n_threads):
        counts += partial[t]
    return counts


def estimate_background_intensity(image, nbins=100):
    """Background intensity of a 2-D image: the centre of the most populated of `nbins`
    equal bins over [nanmin, nanmax]. Returns (background, hist, bin_edges).

    The bins use the np.histogram(bins=nbins, range=(min, max)) arithmetic that
    da.histogram applied per chunk, so hist and bin_edges are exactly origin/dev's. A uint16
    image (Xenium morphology) is counted per value in one parallel pass and binned from the
    counts; any other dtype goes through np.histogram directly.
    """
    if image.dtype == np.uint16:
        counts = _uint16_counts(image, numba.get_num_threads())
        values = np.flatnonzero(counts)
        vmin, vmax = values[0], values[-1]
    else:
        vmin, vmax = np.nanmin(image), np.nanmax(image)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            raise ValueError("Non-finite min/max encountered.")
    if vmin == vmax:
        vmax = vmin + 1.0
    range_ = (float(vmin), float(vmax))
    if image.dtype == np.uint16:
        hist = np.histogram(values.astype(np.uint16), bins=nbins, range=range_, weights=counts[values])[0]
    else:
        hist = np.histogram(image, bins=nbins, range=range_)[0]
    bin_edges = np.linspace(range_[0], range_[1], num=nbins + 1)

    max_bin_idx = int(np.argmax(hist))
    # center of the winning bin
    background = np.round((bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) * 0.5, 3)
    return background, hist, bin_edges
