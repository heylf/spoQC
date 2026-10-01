import os
import sys
import numpy as np

from .. import helperfuncs
from .. import metrics
from ..core import raster, transcripts
from ..metrics.image import pixel_metrics
from ._slidingwindow import texture_metrics

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
        threads,
        *,
        staining=None
):

    timer = helperfuncs.Timer()

    print(f'[NOTE] start structural analysis with {image_type} and {resolution} for {modality}')
    
    tmp_suffix = modality
    if ( staining ):
        spoqc_tmp_folder = f'{spoqc_tmp_folder}/metrices/{modality}/{staining}/'
        figure_path = f'{figure_path_base}/{modality}/{modality}_metrices/{staining}/'
        tmp_suffix = f'{modality}_{staining}'
    else:
        figure_path = f'{figure_path_base}/{modality}/{modality}_metrices/'
        spoqc_tmp_folder = f'{spoqc_tmp_folder}/metrices/{modality}'

    xy_intensities = None
    intensities = None
    # Quantized copy used only by the entropy/uniformity/homogeneity sliding-window metrics below:
    # each window's np.bincount allocates max(window_values)+1 elements, so raw uint16 image intensities 
    # (full 0-65535 dynamic range) make every window allocation far larger than the handful of values it summarizes.
    # Transcript density values (hqtr) are already small integers and don't need this.
    texture_intensities = None
    if ( modality == 'hqtr' ):
        # Intensities already flipped
        intensities = metrics.transcript_density.transcript_density_image.generate_transcript_density_image(
            sdata,
            figure_path,
            imagedim,
            image_type,
            resolution
        )
        xy_intensities = intensities.reshape(dim_x, dim_y)

        # Plot transcript point plot
        helperfuncs.plot_scatter_by_category(
            transcripts.load_transcripts(sdata, ['x', 'y']).to_pandas(),
            None, 
            figure_path, 
            'transcript_points',
            'transcript_points',
            None,
            pointsize=0.5
        )

        helperfuncs.nparr_to_parquet(intensities, 'transcript_density', spoqc_tmp_folder, tmp_suffix)
        texture_intensities = xy_intensities
    else:
        xy_intensities = raster.load_intensity_image(
            sdata, spoqc_tmp_folder, modality, image_type, resolution, dim_x, dim_y, threads, staining=staining
        )

        n_bins = 256
        texture_intensities = np.floor(
            (xy_intensities.astype(np.float64) - xy_intensities.min())
            / max(xy_intensities.max() - xy_intensities.min(), 1) * (n_bins - 1)
        ).astype(np.uint8)

    # Plot intensities
    name = 'input'
    if ( modality == 'hqtr' ):
        name = f'{name}_transcript_densities'
    elif ( modality == 'hqpr' ):
        name = f'{name}_pixel_intensities'
    else:
        sys.exit('[ERROR] Modality not supported')

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
                 'relevance', 'homogenity', 'entropy', 'uniformity']:
        if ( overwrite or not os.path.exists(f"{spoqc_tmp_folder}/{step}_output_{modality}.parquet") ):
            if ( not ( step == 'intensity' and modality == 'hqtr' ) ):
                steps.append(step)
                print(f"[NOTE] {step} will be performed")

    background_intensity = 0.0  # this is valid for hqtr
    hist = None
    bin_edges = None

    if ( modality == 'hqpr' ):
        background_intensity, hist, bin_edges = metrics.image.utility.estimate_background_intensity(xy_intensities)

    def run(names, kernel, image, plots, **kernel_args):
        # Every file the metrices folder holds with this suffix is a pixel clustering feature
        # (pixel_scoring_dask.dask_clustering_mini_batches), so each metric is written under its step name.
        timer.start()
        flat = pixel_metrics.pixel_metric(kernel, image, figure_path, imagedim, plots, **kernel_args)
        timer.stop()
        for name, values in zip(names, flat, strict=True):
            helperfuncs.nparr_to_parquet(values, name, spoqc_tmp_folder, tmp_suffix)

    if ( 'intensity' in steps and modality == 'hqpr' ):
        # General Singal/Noise ratio. Is the pixel noise or true positive?
        # Not valid for hqtr because the background is a constant of 0.0.
        print('[NOTE] Evaluate pixel intensity')
        metrics.image.utility.plot_intensity_histogram(figure_path, background_intensity, hist, bin_edges)
        run(['intensity'], pixel_metrics.signal_noise_ratio, xy_intensities,
            [('snr', 'Log2 Signal-Noise-Ratio', None)], background_intensity=background_intensity)

    if ( 'lbp' in steps ):
        # Pixel pattern information. Does a pixel live in a specific pattern?
        # Mostly useful to identify if windows have specific patterns you want to cluster.
        print('[NOTE] Investigate local binary patterns')
        run(['lbp'], pixel_metrics.lbp, xy_intensities, [('lbp', 'Local Binary Pattern (LBP)', None)],
            n_points=100, radius=3)

    #######################
    ###### Structure ######
    #######################

    if ( 'edge_strength' in steps ):
        # EDGE STRENGTH - Edge detection
        print('[NOTE] Calculate edge strength')
        run(['edge_strength'], pixel_metrics.edge_strength, xy_intensities, [('edge_strength', 'Edge Strength', None)])

    if ( 'energy' in steps ):
        # How much inforamtion has a pixel?
        print('[NOTE] Calculate pixel energy')
        run(['energy'], pixel_metrics.energy, xy_intensities, [('energy', 'Log10 Pixel Energy', None)],
            window_size=5)

    if ( 'relevance' in steps ):
        # Just check which pixel are have intensities bigger than background.
        print('[NOTE] Investigate pixel relevance')
        run(['relevance'], pixel_metrics.relevance, xy_intensities,
            [('relevance', 'Pixel Relevance', {"high rel.": "#FFFFFF", "low rel.": "#000000"})],
            background_intensity=background_intensity)

    ###########################################
    ###### Structure and anti structure ######
    ###########################################

    if ( {'entropy', 'uniformity', 'homogenity'} & set(steps) ):
        # Entropy: how much information contributes a pixel?
        # Uniformity: is the pixel in a noisy region?
        # Homogeneity: how much does a pixel disrupt the local neighbourhood?
        # All three come from the same 5x5 window histograms, built once.
        print("[NOTE] Calculate pixel entropy, uniformity and homogeneity")
        run(['entropy', 'uniformity', 'homogenity'], texture_metrics, texture_intensities,
            [('entropy', 'Pixel Entropy', None), ('uniformity', 'Pixel Uniformity', None),
             ('homogeneity', 'Pixel Homogeneity', None)],
            window_size=5)

    return background_intensity
