import numpy as np
import cv2
import os

from ... import helperfuncs
from ... import core

def _pixel_edge_strength(figure_path, xy_intensities, imagedim, name, spoqc_tmp_folder, tmp_suffix):
    timer = helperfuncs.Timer()
    timer.start()

    # This I have to do else it breaks.
    xy_intensities = xy_intensities.astype(np.uint16)
    
    # Compute gradients in the x and y directions using Sobel filters
    grad_x = cv2.Sobel(xy_intensities, cv2.CV_64F, 1, 0, ksize=3)  # Gradient along x
    grad_y = cv2.Sobel(xy_intensities, cv2.CV_64F, 0, 1, ksize=3)  # Gradient along y

    # Compute the magnitude of the gradient (edge strength)
    # edge_strength = np.sqrt(grad_x**2 + grad_y**2)
    log10_edge_strength_image = np.log10( ( np.sqrt(grad_x**2 + grad_y**2) + 1 ) )

    helperfuncs.plot_pixels(
        figure_path,
        log10_edge_strength_image,
        imagedim,
        'edge_strength', 
        'Edge Strength', 
        'hot',
        False,
        False
    )

    timer.stop()

    helperfuncs.nparr_to_parquet(log10_edge_strength_image.flatten(), name, spoqc_tmp_folder, tmp_suffix)


def init_metric(enterprise):

    # These have to be defined.
    name = "edge_strength"
    submetrics = ["edge_strength"]
    modality = "hqpr"
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
    tmp_file = f"{tmp_folder}/{name}_output_{modality}_{enterprise.args.staining}.parquet"

    # Do not touch.
    if not enterprise.args.overwrite and os.path.exists(tmp_file):
        should_be_calculated = False

    # These are given my your metric calc function.
    args = [figure_path, xy_intensities, enterprise.cargo.imagedim, name, tmp_folder, 
            f"{modality}_{enterprise.args.staining}"]
    kwargs = None

    metric = core.metric.Metric(
        _pixel_edge_strength, 
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