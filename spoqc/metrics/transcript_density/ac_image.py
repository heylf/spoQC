import spatialdata as sd
import numpy as np
import pandas as pd

from ... import helperfuncs
from ... import priors
from . import local_moran_I
from . import transcript_density_image
from ...core import groupreduce, parquet, transcripts
from ...core.threads import map_rows


def _log10_1p(image, workers):
    return map_rows(lambda x: np.log10(x + 1), (image,), workers)

# We are calculating a kernel density at the end so you will not have your usual [-1,1] autocorraltion values.
def generate_transcript_ambient_density_image(
        sdata,
        figure_path,
        threads,
        imagedim,
        global_ambient,
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

    # Attach ambient score to transcript df.
    features = transcripts.load_transcripts(sdata, ['feature_name'])['feature_name']
    global_ambient.index = [i for i in range(0, len(global_ambient))]
    global_ambient.loc[np.isnan(global_ambient['morans_I']),'morans_I'] = 0.0 # Sometimes you have nan for moran's I.

    # I will not check for absolute values because negative autocorrelation might be biological meaningful.
    # Later rows win for a repeated gene; genes absent from the transcripts assign nothing.
    morans_I_of_gene = dict(zip(global_ambient['genes'], global_ambient['morans_I']))
    morans_I_by_code = transcripts.lookup_by_code(features, morans_I_of_gene, 0.0, np.float64)
    morans_I = morans_I_by_code[features.to_physical().to_numpy()]

    # Now we will add the local morans I
    local_morans_I = local_moran_I.calculate_local_moran_I_values(sdata, threads)

    # First calculate the ambient potential (global Moran's I) ----------------------------------
    print("[NOTE] Calcualte pixel max")
    timer.start()
    transcript_density_list = (
        groupreduce.to_grid(pixels, groupreduce.group_max(morans_I, rows, offsets), n_pixels, 0.0)
        .astype("float64")
    )
    timer.stop()

    xy_transcript_density = transcript_density_list.reshape(dim_x, dim_y)

    img_extent = sd.get_extent(sdata[image_type], coordinate_system='global')
    imagedim = helperfuncs.ImageDimStruct(img_extent['x'][0], img_extent['y'][0],
                                        img_extent['x'][1], img_extent['y'][1])

    xy_kernel_transcript_density = transcript_density_image.disk_density(xy_transcript_density, kernel_radius, threads)
    del transcript_density_list, xy_transcript_density  # free the grid before the next one
    # xy_kernel_transcript_density = xy_kernel_transcript_density.astype(np.uint16) # conversion needed for cv2

    if ( figure_path != None ):
        helperfuncs.plot_pixels(
            figure_path,
            _log10_1p(xy_kernel_transcript_density, threads),
            imagedim,
            'transcript_global_autocorrelation_density',
            'Transcript Global Autocorrelation Density (Potential)', 
            'gray',
            True,
            True
        )

    # Second calculate the ambient value (local Moran's I) ----------------------------------
    print("[NOTE] Calcualte pixel max")
    timer.start()
    local_transcript_density_list = (
        groupreduce.to_grid(pixels, groupreduce.group_max(local_morans_I, rows, offsets), n_pixels, 0.0)
        .astype("float64")
    )
    timer.stop()

    local_xy_transcript_density = local_transcript_density_list.reshape(dim_x, dim_y)

    local_xy_kernel_transcript_density = transcript_density_image.disk_density(local_xy_transcript_density, kernel_radius, threads)
    del local_transcript_density_list, local_xy_transcript_density

    if ( figure_path != None ):
        helperfuncs.plot_pixels(
            figure_path,
            _log10_1p(local_xy_kernel_transcript_density, threads),
            imagedim,
            'transcript_local_autocorrelation_density',
            'Transcript Local Autocorrelation Density (Value)', 
            'gray',
            True,
            True
        )

    # Now we combine both ambient potential with ambient value ---------------------------------
    # combined = -1     ---> -1 * 1 or 1 * -1 = disagreement, direction between global and local
    # combined = 1      ---> -1 * -1 or 1 * 1 = agreement, direction between global and local
    # combined = 0      ---> 0 * -1 or 0 * 1 or -1 * 0 or 1 * 0 = vanishing, RNA is either global or local ambient
    xy_kernel_ac_density = map_rows(
        lambda local, potential: np.abs(local * potential),
        (local_xy_kernel_transcript_density, xy_kernel_transcript_density),
        threads,
    )

    if ( figure_path != None ):
        helperfuncs.plot_pixels(
            figure_path,
            _log10_1p(xy_kernel_ac_density, threads),
            imagedim,
            'transcript_autocorrelation_density',
            'Transcript Autocorrelation Density (Combined)', 
            'gray',
            True,
            True
        )

    return xy_kernel_ac_density.ravel()


def transcript_ac_image(
        sdata,
        figure_path,
        spoqc_tmp_folder,
        modality,
        threads,
        image_type,
        resolution,
        dim_x,
        dim_y,
        imagedim,
        *,
        chunk_size=priors.hqtr.ac_or_qv.PART_ROWS
    ):
    figure_path = f'{figure_path}/hqtr/hqtr_ac/'
    timer = helperfuncs.Timer()

    print("[NOTE] Generate ac image")
    timer.start()
    global_ambient = pd.read_parquet(f"{spoqc_tmp_folder}/ambient_output_genes.parquet", engine="pyarrow")
    timer.stop()

    # I have now for every pixel the density of the max autocorrelation.
    # That means I know now which pixels have high global gene correlation patterns.
    timer.start()
    np_arr = generate_transcript_ambient_density_image(sdata, figure_path, threads, imagedim, global_ambient, image_type, 
                                                       resolution)
    timer.stop()

    print("[NOTE] Generate ac histogram")
    timer.start()
    helperfuncs.plot_histogram_for_array(
        np_arr[np_arr > 0],
        100,
        figure_path,
        "Transcript autocorrelation density historgram",
        "transcript_ac"
    )
    timer.stop()

    # Genes which have random or a constant values across the whole slide while have an autocorraltion around 0.0.
    # These genes might be ambient, i.e., there is a spillover of those genese counts across the whole slide.
    print("[NOTE] Calculate ac probabilities")
    timer.start()
    norm_p, part_columns = priors.hqtr.ac_or_qv.calc_prob_pixel_stuff_v2(
        np_arr, figure_path, 0.4, 1, 'left', 'ac_density', threads
    )
    timer.stop()

    helperfuncs.plot_pixels(
        figure_path,
        norm_p.reshape(dim_x, dim_y),
        imagedim,
        'norm_p_ac_density', 
        'Normalized probability of AC Density Pixel', 
        'hot',
        False,
        False
    )
    parquet.write_parts(f"{spoqc_tmp_folder}/{modality}_output_ac_prob", len(np_arr), part_columns, range(0, len(np_arr), chunk_size), threads)
