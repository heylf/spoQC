import spatialdata as sd
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd

from scipy.ndimage import convolve

from ... import helperfuncs
from ... import priors
from . import local_moran_I

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

    transcript_coords_df = sd.get_centroids(sdata['transcripts'], coordinate_system='global').compute()
    transcript_coords_df = transcript_coords_df.astype(int)
    xy_transcript_coords_df = transcript_coords_df.loc[:,['x','y']]
    xy_transcript_coords_df['morans_I'] = np.zeros(len(xy_transcript_coords_df))

    # Attach ambient score to transcript df.
    features = np.array(sdata['transcripts'].compute()['feature_name'])
    global_ambient.index = [i for i in range(0, len(global_ambient))]
    global_ambient.loc[np.isnan(global_ambient['morans_I']),'morans_I'] = 0.0 # Sometimes you have nan for moran's I.

    # I will not check for absolute values because negative autocorrelation might be biological meaningful.
    for i in range(0, len(global_ambient)):
        gene = global_ambient.loc[i, 'genes']
        morans_I = global_ambient.loc[i, 'morans_I']
        xy_transcript_coords_df.loc[features == gene, 'morans_I'] = morans_I

    # Now we will add the local morans I
    xy_transcript_coords_df['local_moran_I'] = local_moran_I.calculate_local_moran_I_values(sdata, threads)

    # First calculate the ambient potential (global Moran's I) ----------------------------------
    print("[NOTE] Calcualte pixel max")
    timer.start()
    gm = (
        xy_transcript_coords_df
        .groupby(["x", "y"])["morans_I"]
        .max()
        .rename("morans_I")
    )

    x_idx = range(int(imagedim.bb_xmin), int(imagedim.bb_xmax))
    y_idx = range(int(imagedim.bb_ymin), int(imagedim.bb_ymax))
    grid = [(x, y) for y in y_idx for x in x_idx]
    grid_mi = pd.MultiIndex.from_tuples(grid, names=["x", "y"])

    transcript_density_list = (
        gm.reindex(grid_mi)     # align to the full grid
        .fillna(0.0)
        .to_numpy()
        .astype("float64")
    )
    timer.stop()

    xy_transcript_density = np.array(transcript_density_list).reshape(dim_x, dim_y)

    img_extent = sd.get_extent(sdata[image_type], coordinate_system='global')
    imagedim = helperfuncs.ImageDimStruct(img_extent['x'][0], img_extent['y'][0],
                                        img_extent['x'][1], img_extent['y'][1])
    nuclei_centroid_coords = sd.get_centroids(sdata['nucleus_boundaries'], coordinate_system='global').compute()

    # Create circular kernel (disk mask)
    y, x = np.ogrid[-kernel_radius:kernel_radius+1, -kernel_radius:kernel_radius+1]
    mask = (x**2 + y**2) <= kernel_radius**2
    kernel = mask.astype(xy_transcript_density.dtype)
    xy_kernel_transcript_density = convolve(xy_transcript_density, kernel, mode='constant', cval=0)
    xy_kernel_transcript_density = np.flipud(xy_kernel_transcript_density)
    # xy_kernel_transcript_density = xy_kernel_transcript_density.astype(np.uint16) # conversion needed for cv2

    if ( figure_path != None ):
        helperfuncs.plot_pixels(
            figure_path,
            np.log10(xy_kernel_transcript_density + 1),
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
    gm = (
        xy_transcript_coords_df
        .groupby(["x", "y"])["local_moran_I"]
        .max()
        .rename("local_moran_I")
    )

    local_transcript_density_list = (
        gm.reindex(grid_mi)     # align to the full grid
        .fillna(0.0)
        .to_numpy()
        .astype("float64")
    )
    timer.stop()

    local_xy_transcript_density = np.array(local_transcript_density_list).reshape(dim_x, dim_y)

    # Create circular kernel (disk mask)
    kernel = mask.astype(local_xy_transcript_density.dtype)
    local_xy_kernel_transcript_density = convolve(local_xy_transcript_density, kernel, mode='constant', cval=0)
    local_xy_kernel_transcript_density = np.flipud(local_xy_kernel_transcript_density)

    if ( figure_path != None ):
        helperfuncs.plot_pixels(
            figure_path,
            np.log10(local_xy_kernel_transcript_density + 1),
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
    xy_kernel_ac_density = local_xy_kernel_transcript_density * xy_kernel_transcript_density

    if ( figure_path != None ):
        helperfuncs.plot_pixels(
            figure_path,
            np.log10(xy_kernel_ac_density + 1),
            imagedim,
            'transcript_autocorrelation_density',
            'Transcript Autocorrelation Density (Combined)', 
            'gray',
            True,
            True
        )

    return xy_kernel_ac_density.flatten()


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
        chunk_size=10000
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
    image_ddf = dd.from_dask_array(da.from_array(np_arr, chunks=chunk_size), columns=["ac_density"])
    timer.stop()

    print("[NOTE] Generate ac histogram")
    timer.start()
    helperfuncs.plot_histogram_for_array(
        image_ddf[image_ddf['ac_density'] > 0]['ac_density'].compute().to_numpy(),
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
    image_ddf = priors.hqtr.ac_or_qv.calc_prob_pixel_stuff_v2(image_ddf, figure_path, 0.4, 1, 'left', 'ac_density')
    timer.stop()

    helperfuncs.plot_pixels(
        figure_path,
        image_ddf['norm_p_ac_density'].compute().to_numpy().reshape(dim_x, dim_y),
        imagedim,
        'norm_p_ac_density', 
        'Normalized probability of AC Density Pixel', 
        'hot',
        False,
        False
    )
    helperfuncs.ddf_to_parquet(image_ddf, modality, spoqc_tmp_folder, [], 'ac_prob')