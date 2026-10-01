import spatialdata as sd
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd

from scipy.ndimage import convolve

from ... import helperfuncs
from ... import priors

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

    transcript_coords_df = sd.get_centroids(sdata['transcripts'], coordinate_system='global').compute()
    transcript_coords_df = transcript_coords_df.astype(int)
    xy_transcript_coords_df = transcript_coords_df.loc[:,['x','y']]
    xy_transcript_coords_df['qv'] = sdata['transcripts'].compute()['qv']

    print("[NOTE] Calcualte pixel mean")
    timer.start()
    # Group by (x, y) and compute mean qv
    gm = (
        xy_transcript_coords_df
        .groupby(["x", "y"])["qv"]
        .mean()
        .rename("qv_means")
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

    print("[NOTE] Densitiy calculation")
    timer.start()
    xy_kernel_transcript_density = convolve(xy_transcript_density, kernel, mode='constant', cval=0)
    xy_kernel_transcript_density = np.flipud(xy_kernel_transcript_density)
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

    return xy_kernel_transcript_density.flatten()


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
        chunk_size=10000
    ):
    figure_path = f'{figure_path}/hqtr/hqtr_qv/'
    timer = helperfuncs.Timer()

    print("[NOTE] Generate qv image")
    timer.start()
    np_arr = generate_transcript_quality_density_image(sdata, figure_path, imagedim, image_type, resolution)
    image_ddf = dd.from_dask_array(da.from_array(np_arr, chunks=chunk_size), columns=["qv_density"])
    timer.stop()

    print("[NOTE] Generate qv histogram")
    timer.start()
    helperfuncs.plot_histogram_for_array(image_ddf['qv_density'].compute().to_numpy(), 100,
                                         figure_path, "Transcript QV", "transcript_qv")
    timer.stop()
    
    # At 10x Genomics they use a threshold of qv < 20 (see 10xBaysor tutorial)
    print("[NOTE] Calculate qv probabilities")
    timer.start()
    image_ddf = priors.hqtr.ac_or_qv.calc_prob_pixel_stuff_v2(image_ddf, figure_path, 20.0, 3, 'left', 'qv_density')
    timer.stop()

    helperfuncs.plot_pixels(
        figure_path,
        image_ddf['norm_p_qv_density'].compute().to_numpy().reshape(dim_x, dim_y),
        imagedim,
        'norm_p_qv_density',
        'Normalized probability of QV density pixel', 
        'hot',
        False,
        False
    )

    helperfuncs.ddf_to_parquet(image_ddf, modality, spoqc_tmp_folder, [], 'qv_prob')

