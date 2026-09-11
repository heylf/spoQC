import numpy as np
import os

from scipy.ndimage import gaussian_filter

from ... import helperfuncs
from ... import core

def _pixel_energy(figure_path, xy_intensities, imagedim, name, spoqc_tmp_folder, tmp_suffix, window_size=5):
    """
    Calculate the energy in a local neighborhood for each pixel in a grayscale image.

    Args:
        img: Provided image data.
        window_size (int): Size of the sliding window (must be odd).

    Returns:
        energy_image (ndarray): Image of energy values for each pixel.
    """
    
    # Square the pixel intensities
    squared_image = xy_intensities.astype(np.float64) ** 2

    # Compute the local energy using a sliding window (mean of squared values)
    energy_image = gaussian_filter(squared_image, sigma=1, radius=window_size)

    log10_energy_image = np.log10(energy_image + 1)

    helperfuncs.plot_pixels(
        figure_path,
        log10_energy_image,
        imagedim,
        'energy', 
        'Log10 Pixel Energy', 
        'hot',
        False,
        False
    )

    helperfuncs.nparr_to_parquet(log10_energy_image.flatten(), name, spoqc_tmp_folder, tmp_suffix)


def init_metric(enterprise):

    # These have to be defined.
    name = "energy"
    submetrics = ["energy"]
    modality = "hqtr"
    needs_metrics = []
    step_when_it_is_calculated = [f"{modality}_metrices", "all"]
    loaded_for_analysis = True
    loaded_for_visualization = True
    should_be_calculated = True

    # Do not touch.
    tmp_folder = ""
    figure_path = ""
    intensities = None
    xy_intensities = None
    texture_intensities = None
    if modality == "hqpr":
        tmp_folder = f'{enterprise.args.tmp_dir}/metrices/{modality}/{enterprise.args.staining}/'
        figure_path = f'{enterprise.args.output_dir}/{modality}/{modality}_metrices/{enterprise.args.staining}/'
        intensities = enterprise.cargo.intensities_hqpr
        xy_intensities = enterprise.cargo.xy_intensities_hqpr
        texture_intensities = enterprise.cargo.texture_intensities_hqpr
    else:
        tmp_folder = f'{enterprise.args.tmp_dir}/metrices/{modality}/'
        figure_path = figure_path = f'{enterprise.args.output_dir}/{modality}/{modality}_metrices/'
        intensities = enterprise.cargo.intensities_hqtr
        xy_intensities = enterprise.cargo.xy_intensities_hqtr
        texture_intensities = enterprise.cargo.texture_intensities_hqtr
    tmp_file = f"{tmp_folder}/{name}_output_{modality}.parquet"

    # Do not touch.
    if not enterprise.args.overwrite and os.path.exists(tmp_file):
        should_be_calculated = False

    # These are given my your metric calc function.
    args = [figure_path, xy_intensities, enterprise.cargo.imagedim, name, tmp_folder, modality]
    kwargs = None

    metric = core.metric.Metric(
        _pixel_energy, 
        name,
        submetrics,
        modality,
        should_be_calculated = should_be_calculated,
        needs_metrics = needs_metrics,
        step_when_it_is_calculated = step_when_it_is_calculated,
        loaded_for_analysis = loaded_for_analysis,
        loaded_for_visualization = loaded_for_visualization,
        args = args,
        kwargs = kwargs,
    )    
    
    return metric