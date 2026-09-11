import numpy as np
import cv2
import os

from ... import helperfuncs
from ... import core

def _pixel_relevance(
        figure_path,
        xy_intensities,
        imagedim,
        name,
        spoqc_tmp_folder,
        tmp_suffix,
        sdata,
        image_type,
        resolution,
        staining,
        modality,
    ):
    """
    Determines if each pixel belongs to a segmented region and visualizes relevance.

    Args:
        image_path (str): Path to the input grayscale image.
        threshold (int): Threshold value for segmentation (0-255).

    Returns:
        segmented_image (ndarray): Binary segmented image (0 or 255).
        relevance_map (ndarray): Map showing relevance of each pixel to the segmented region.
    """
    timer = helperfuncs.Timer()

    background_intensity = 0.0  # this is valid for hqtr

    if ( modality == 'hqpr' ):
        background_intensity, _, _ = helperfuncs.estimate_background_intensity_dask(
            sdata,
            image_type,
            resolution,
            staining
        )

    # This I have to do else it breaks.
    xy_intensities = xy_intensities.astype(np.uint16)

    # Apply blur
    print("[NOTE] Gaussian Blur")
    timer.start()
    blur = cv2.GaussianBlur(xy_intensities,(5,5),0)
    timer.stop()

    # Apply binary thresholding for segmentation
    print("[NOTE] Otsu thresholding")
    timer.start()
    _, segmented_image = cv2.threshold(blur, background_intensity, 
                                       max(xy_intensities.flatten()), cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    timer.stop()

    # Compute relevance map: pixels in the segmented region have value 1, others have 0
    relevance_map_image = (segmented_image > 0).astype(np.uint8)

    print("[NOTE] Plotting")
    timer.start()
    helperfuncs.plot_pixels(
        figure_path,
        relevance_map_image,
        imagedim,
        'relevance', 
        'Pixel Relevance', 
        'hot',
        False,
        True,
        legend_dict={"high rel.": "#FFFFFF", "low rel.": "#000000"}
    )
    timer.stop()

    helperfuncs.nparr_to_parquet(relevance_map_image.flatten(), name, spoqc_tmp_folder, tmp_suffix)


def init_metric(enterprise):

    # These have to be defined.
    name = "relevance"
    submetrics = ["relevance"]
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
    args = [figure_path, xy_intensities, enterprise.cargo.imagedim, name, tmp_folder, modality, enterprise.cargo.sdata,
            enterprise.args.image_type, enterprise.args.resolution, enterprise.args.staining, modality]
    kwargs = None

    metric = core.metric.Metric(
        _pixel_relevance, 
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