import os
import sys
import numpy as np

from .. import helperfuncs
from .. import image_analysis
from .. import hqr

def start_image_struc_analyis(
        sdata,
        figure_path_base,
        spoqc_tmp_folder,
        modality,
        image_type,
        resolution,
        imagedim,
        dim_x,
        dim_y,
        overwrite,
        *,
        staining=None
):

    timer = helperfuncs.Timer()

    print(f'[NOTE] start HQPR analysis with {image_type} and {resolution}')
    
    tmp_suffix = modality
    if ( staining ):
        spoqc_tmp_folder = f'{spoqc_tmp_folder}/metrices/{modality}/{staining}/'
        figure_path = f'{figure_path_base}/{modality}/{staining}/{modality}_metrices/'
        tmp_suffix = f'{modality}_{staining}'
    else:
        figure_path = f'{figure_path_base}/{modality}/{modality}_metrices/'
        spoqc_tmp_folder = f'{spoqc_tmp_folder}/metrices/{modality}'

    xy_intensities = None
    intensities = None
    if ( modality == 'hqtr' ):
        # Intensities already flipped
        intensities = hqr.hqtr.generate_transcript_density_image(sdata, figure_path, imagedim, image_type, resolution)
        xy_intensities = intensities.reshape(dim_x, dim_y)
    else:
        xy_intensities = sdata[image_type][resolution].image.values[int(staining)]
        xy_intensities = np.flipud(xy_intensities)
        intensities = xy_intensities.flatten()
    
    # Plot intensities
    name = 'input'
    if ( modality == 'hqtr' ):
        name = f'{name}_transcript_densities'
    elif ( modality == 'hqpr' ):
        name = f'{name}_pixel_intensities'
    else:
        sys.exit('[ERROR] Modality not supported')

    # TODO I need to generate log here to make the image stick out more
    helperfuncs.plot_pixels(
        figure_path,
        np.log10(xy_intensities + 1),
        imagedim,
        name,
        name,
        'gray',
        False,
        False
    )

    steps = []

    for step in ['intensity', 'edge_strength', 'lbp', 'energy', 
                 'relevance', 'homogenity', 'entropy', 'uniformity',
                 'cluster']:
        if ( overwrite or not os.path.exists(f"{spoqc_tmp_folder}/{step}_output_{modality}.parquet") ):
            if ( not ( step == 'intensity' and modality == 'hqtr' ) ):
                steps.append(step)
                print(f"[NOTE] {step} will be performed")

    background_intensity = 0.0  # this is valid for hqtr
    hist = None
    bin_edges = None

    if ( modality == 'hqpr' ):
        background_intensity, hist, bin_edges = image_analysis.image_metrices.estimate_background_intensity_dask(
            sdata,
            image_type,
            resolution,
            staining
        )

    step = 'intensity'
    if ( step in steps and modality == 'hqpr' ):
        # General Singal/Noise ratio. Is the pixel noise or true positive?
        # Not valid for hqtr because the background is a constant of 0.0.
        print('[NOTE] Evaluate pixel intensity')
        timer.start()
        signal_noise_ratio_log2fc = image_analysis.image_metrices.pixel_intensity_qc(figure_path, intensities, 
                                                                 background_intensity, hist, bin_edges, 
                                                                 dim_x, dim_y, imagedim)
        timer.stop()
        helperfuncs.nparr_to_parquet(signal_noise_ratio_log2fc, step, spoqc_tmp_folder, tmp_suffix)

    # TODO not used so far
    step = 'lbp'
    if ( step in steps ):
        # Pixel pattern information. Does a pixel live in a specific pattern?
        # Mostly useful to identify if windows have specific patterns you want to cluster.
        print('[NOTE] Investigate local binary patterns')
        timer.start()
        lbp = image_analysis.image_metrices.pixel_lbp(figure_path, xy_intensities, 100, 3, imagedim)
        timer.stop()
        helperfuncs.nparr_to_parquet(lbp, step, spoqc_tmp_folder, tmp_suffix)

    #######################
    ###### Structure ######
    #######################

    step = 'edge_strength'
    if ( step in steps ):
        # EDGE STRENGTH - Edge detection
        print('[NOTE] Calculate edge strength')
        timer.start()
        edge_strength = image_analysis.image_metrices.pixel_edge_strength(figure_path, xy_intensities, imagedim)
        timer.stop()
        helperfuncs.nparr_to_parquet(edge_strength, step, spoqc_tmp_folder, tmp_suffix)

    step = 'energy'
    if ( step in steps ):
        # How much inforamtion has a pixel?
        print('[NOTE] Calculate pixel energy')
        timer.start()
        pixel_energy = image_analysis.image_metrices.pixel_energy(figure_path, xy_intensities, 5, imagedim)
        timer.stop()
        helperfuncs.nparr_to_parquet(pixel_energy, step, spoqc_tmp_folder, tmp_suffix)

    step = 'relevance'
    if ( step in steps ):
        # Just check which pixel are have intensities bigger than background.
        # TODO check Otsu thresholding here.
        print('[NOTE] Investigate pixel relevance')
        timer.start()
        pixel_relevance = image_analysis.image_metrices.pixel_relevance(figure_path, xy_intensities, 
                                                                        background_intensity, imagedim)
        timer.stop()
        helperfuncs.nparr_to_parquet(pixel_relevance, step, spoqc_tmp_folder, tmp_suffix)

    step = "entropy"
    if step in steps:
        # General Pixel Information. How much information contributes a pixel?
        # Computational expensive.
        print("[NOTE] Calculate pixel entropy")
        timer.start()
        pixel_entropy = image_analysis.image_metrices.pixel_entropy(figure_path, xy_intensities, 5, imagedim)
        timer.stop()
        helperfuncs.nparr_to_parquet(pixel_entropy, step, spoqc_tmp_folder, tmp_suffix)

    ############################
    ###### Anti structure ######
    ############################

    step = "uniformity"
    if step in steps:
        # Is the pixel in a noisy region?
        print("[NOTE] Calculate pixel uniformity with")
        timer.start()
        pixel_uniformity = image_analysis.image_metrices.pixel_uniformity(figure_path, xy_intensities, 5, imagedim)
        timer.stop()
        helperfuncs.nparr_to_parquet(pixel_uniformity, step, spoqc_tmp_folder, tmp_suffix)

    step = "homogenity"
    if step in steps:
        # How much does a pixel disrupt the local neighbourhood?
        # Or how homogenous is the pixel around the region?
        # Computational expensive.
        print("[NOTE] Calculate pixel homogeneity")
        timer.start()
        pixel_homogeneity = image_analysis.image_metrices.pixel_homogeneity(figure_path, xy_intensities, imagedim, 5)
        timer.stop()
        helperfuncs.nparr_to_parquet(pixel_homogeneity, step, spoqc_tmp_folder, tmp_suffix)
