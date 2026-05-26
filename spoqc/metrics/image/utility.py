import numpy as np
import plotly.express as px
import dask.array as da

from ... import helperfuncs

def turn_into_uint8(arr):
    # normalize to 0–1 if needed
    if int(arr.min()) != 0 or int(arr.max()) != 1:
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    # scale to 0–255 and convert to uint8
    uint8_arr = (arr * 255).astype(np.uint8)
    return uint8_arr

def pixel_intensity_qc(figure_path, intensities, background_intensity, hist, bin_edges, dim_x, dim_y, imagedim):

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
    fig.write_image(f"{figure_path}/histogram_intensity.png", scale=3)
    fig.write_image(f"{figure_path}/histogram_intensity.pdf", scale=3)

    with open(f'{figure_path}/histogram_intensity.html', 'w') as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    
    signal_noise_ratio_log2fc = np.log2( (intensities + 1) / background_intensity )

    helperfuncs.plot_pixels(
        figure_path,
        np.array(signal_noise_ratio_log2fc).reshape(dim_x, dim_y),
        imagedim,
        'snr', 
        'Log2 Signal-Noise-Ratio', 
        'hot',
        False,
        False
    )
    
    return signal_noise_ratio_log2fc


def estimate_background_intensity_dask(sdata, image_type, resolution, staining, nbins=100, range_=None):
    """
    nbins: number of histogram bins
    range_: optional (min, max); if None, computed lazily with dask
    """
    intensities = sdata[image_type][resolution].image.data[int(staining)]
    intensities.ravel()

    if not hasattr(intensities, "chunks"):
        raise TypeError("Pass a dask.array for the Dask implementation.")

    # Compute min/max lazily if not supplied (cheap: just scalars)
    if range_ is None:
        vmin = da.nanmin(intensities)
        vmax = da.nanmax(intensities)
        vmin, vmax = da.compute(vmin, vmax)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            raise ValueError("Non-finite min/max encountered.")
        if vmin == vmax:
            vmax = vmin + 1.0
        range_ = (float(vmin), float(vmax))

    # Dask builds the histogram in a reduction; result is tiny (nbins) -> safe to .compute()
    hist, bin_edges = da.histogram(intensities, bins=nbins, range=range_)
    hist, bin_edges = da.compute(hist, bin_edges)

    max_bin_idx = int(np.argmax(hist))
    # center of the winning bin
    background = np.round((bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) * 0.5, 3)
    return background, hist, bin_edges


def estimate_background_intensity(intensities):
    nbins = 100

    hist, bin_edges = np.histogram(intensities, bins=nbins)

    # Find the bin with the highest count
    max_count = np.max(hist)
    max_bin_index = np.argmax(hist)
    max_bin_range = (bin_edges[max_bin_index], bin_edges[max_bin_index + 1])

    background_intensity = np.round(np.mean(max_bin_range), 3)

    return (background_intensity, hist, bin_edges)