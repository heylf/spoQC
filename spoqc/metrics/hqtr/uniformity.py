import os
import numpy as np

from numba import njit
from numpy.typing import DTypeLike

type Array2D[T: np.generic] = np.ndarray[tuple[int, int], np.dtype[T]]
type Array3D[T: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[T]]
type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.integer]]

from spoqc.image_analysis._slidingwindow import sliding_window_padded
from ... import helperfuncs
from ... import core

@njit
def occurrence_probability(
    x: IntArray,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """Relative occurrence of each non-negative integer"""
    # assert x.size > 0
    counts = np.bincount(x.ravel())
    return counts / x.size

@njit
def kl_divergence_uniform(x: IntArray) -> np.number:
    """Kullback-Leibler divergence against a uniform distribution"""
    p = occurrence_probability(x)
    # removing zeros is faster than using nansum
    p = p[p > 0]

    # uniform distribution: probability for each element
    # if there are less observations than potential levels truncate
    q = max(1 / x.size, 1 / (np.iinfo(x.dtype).max + 1))

    return ( -(p * np.log(p / q)) ).sum()

def _pixel_uniformity(figure_path, img, imagedim, name, spoqc_tmp_folder, tmp_suffix, window_size=5, mode="reflect"):
    timer = helperfuncs.Timer()

    # numba.set_threads(threads)
    radius = (window_size - 1) // 2

    timer.start()
    print(f"... Parallel processing")
    uniformity_image = -sliding_window_padded(kl_divergence_uniform, img, radius, mode=mode)
    timer.stop()
    
    helperfuncs.plot_pixels(
        figure_path,
        uniformity_image,
        imagedim,
        "uniformity",
        "Pixel Uniformity",
        "hot",
        False,
        False,
    )

    helperfuncs.nparr_to_parquet(uniformity_image.flatten(), name, spoqc_tmp_folder, tmp_suffix)


def init_metric(enterprise):

    # These have to be defined.
    name = "uniformity"
    submetrics = ["uniformity"]
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
    args = [figure_path, texture_intensities, enterprise.cargo.imagedim, name, tmp_folder, modality]
    kwargs = None

    metric = core.metric.Metric(
        _pixel_uniformity, 
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