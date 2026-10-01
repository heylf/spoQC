import spatialdata as sd
import numpy as np

from ... import helperfuncs
from ... import priors
from . import transcript_density_image
from ...core import groupreduce, parquet, threads, transcripts

def generate_transcript_quality_density_image(
        sdata,
        figure_path,
        imagedim,
        image_type,
        resolution,
        *,
        kernel_radius=3,
        flip=False
):

    timer = helperfuncs.Timer()

    # Get general stuff
    dim_x = len(sdata[image_type][resolution].image.y.values)
    dim_y = len(sdata[image_type][resolution].image.x.values)

    pixels, rows, offsets, n_pixels = transcript_density_image.transcript_pixel_groups(sdata, imagedim)
    qv = transcripts.load_transcripts(sdata, ['qv'])['qv'].to_numpy()

    print("[NOTE] Calcualte pixel mean")
    timer.start()
    # Mean qv per pixel, as pandas' groupby mean computes it, aligned to the full grid
    transcript_density_list = (
        groupreduce.to_grid(pixels, groupreduce.group_mean(qv, rows, offsets), n_pixels, 0.0)
        .astype("float64")
    )
    timer.stop()

    xy_transcript_density = transcript_density_list.reshape(dim_x, dim_y)

    img_extent = sd.get_extent(sdata[image_type], coordinate_system='global')
    imagedim = helperfuncs.ImageDimStruct(img_extent['x'][0], img_extent['y'][0],
                                        img_extent['x'][1], img_extent['y'][1])
    nuclei_centroid_coords = sd.get_centroids(sdata['nucleus_boundaries'], coordinate_system='global').compute()

    print("[NOTE] Densitiy calculation")
    timer.start()
    xy_kernel_transcript_density = transcript_density_image.disk_density(xy_transcript_density, kernel_radius, threads.budget())
    del transcript_density_list, xy_transcript_density
    timer.stop()
    #xy_kernel_transcript_density = xy_kernel_transcript_density.astype(np.uint16) # conversion needed for cv2

    if ( figure_path != None ):
        if ( flip ):
            helperfuncs.plot_pixels(
                figure_path,
                np.flipud(xy_kernel_transcript_density),
                imagedim,
                'transcript_qv_density',
                'Transcript QV Density', 
                'gray',
                True,
                True,
                points=nuclei_centroid_coords
            )
        else:
            helperfuncs.plot_pixels(
                figure_path,
                xy_kernel_transcript_density,
                imagedim,
                'transcript_qv_density',
                'Transcript QV Density', 
                'gray',
                True,
                True,
                points=nuclei_centroid_coords
            )

    return xy_kernel_transcript_density.ravel()


def transcript_qv_image(
        sdata,
        figure_path,
        spoqc_tmp_folder,
        modality,
        image_type,
        resolution,
        dim_x,
        dim_y,
        imagedim,
        *,
        chunk_size=priors.hqtr.ac_or_qv.PART_ROWS
    ):
    figure_path = f'{figure_path}/hqtr/hqtr_qv/'
    timer = helperfuncs.Timer()

    print("[NOTE] Generate qv image")
    timer.start()
    np_arr = generate_transcript_quality_density_image(sdata, figure_path, imagedim, image_type, resolution)
    timer.stop()

    print("[NOTE] Generate qv histogram")
    timer.start()
    helperfuncs.plot_histogram_for_array(np_arr, 100, figure_path, "Transcript QV", "transcript_qv")
    timer.stop()
    
    # At 10x Genomics they use a threshold of qv < 20 (see 10xBaysor tutorial)
    print("[NOTE] Calculate qv probabilities")
    timer.start()
    norm_p, part_columns = priors.hqtr.ac_or_qv.calc_prob_pixel_stuff_v2(
        np_arr, figure_path, 20.0, 3, 'left', 'qv_density', threads.budget()
    )
    timer.stop()

    helperfuncs.plot_pixels(
        figure_path,
        norm_p.reshape(dim_x, dim_y),
        imagedim,
        'norm_p_qv_density',
        'Normalized probability of QV density pixel', 
        'hot',
        False,
        False
    )

    parquet.write_parts(f"{spoqc_tmp_folder}/{modality}_output_qv_prob", len(np_arr), part_columns, range(0, len(np_arr), chunk_size), threads.budget())
