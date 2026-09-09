import numpy as np
import os

from numba import njit
from numpy.typing import DTypeLike

type Array2D[T: np.generic] = np.ndarray[tuple[int, int], np.dtype[T]]
type Array3D[T: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[T]]
type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.integer]]

from spoqc.image_analysis._slidingwindow import sliding_window_padded
from ... import helperfuncs
from ... import core

@njit
def homogeneity(x: IntArray) -> np.floating:
    counts = np.bincount(x.ravel())
    m, n = x.shape
    v = x[m // 2, n // 2]  # central value

    counts[v] -= 1  # remove central pixel
    p = counts / (x.size - 1)
    # np.arange(len(p)) gets the corresponding value to each probability p
    abs_diff = np.fabs(np.arange(len(p)) - v)
    # If the absolute center pixel difference of the intensieties is large then the homogenity is low and vice versa.
    # If all values are the same then the homogenity is 1.
    return (p / (abs_diff + 1)).sum()

def _pixel_homogeneity(figure_path, img, imagedim, name, spoqc_tmp_folder, tmp_suffix, window_size=5, mode="reflect"):
    timer = helperfuncs.Timer()

    # numba.set_threads(threads)
    radius = (window_size - 1) // 2

    timer.start()
    print("... Parallel processing")
    homogeneity_image = sliding_window_padded(homogeneity, img, radius, mode=mode)
    timer.stop()

    helperfuncs.plot_pixels(
        figure_path,
        homogeneity_image,
        imagedim,
        "homogeneity",
        "Pixel Homogeneity",
        "hot",
        False,
        False,
    )

    helperfuncs.nparr_to_parquet(homogeneity_image.flatten(), name, spoqc_tmp_folder, tmp_suffix)


def init_metric(enterprise):

    # These have to be defined.
    name = "homogeneity"
    submetrics = ["homogeneity"]
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
    args = [figure_path, texture_intensities, enterprise.cargo.imagedim, name, tmp_folder, 
            f"{modality}_{enterprise.args.staining}"]
    kwargs = None

    metric = core.metric.Metric(
        _pixel_homogeneity, 
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