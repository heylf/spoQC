import spatialdata as sd
import numpy as np
import pandas as pd

from scipy.ndimage import convolve

from ... import helperfuncs

def generate_transcript_density_image(
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

    # These list I need later because the image matrix has not the same index range as the centroid coords.
    x_idx = [i for i in range(int(imagedim.bb_xmin), int(imagedim.bb_xmax))]
    y_idx = [i for i in range(int(imagedim.bb_ymin), int(imagedim.bb_ymax))]

    print("[NOTE] Translate cooridnates")
    timer.start()
    counts = (
        xy_transcript_coords_df
        .value_counts(subset=['x','y'])      # returns a Series indexed by MultiIndex (x,y)
        .rename('count')
    )
    grid_tuples = [(x, y) for y in y_idx for x in x_idx]
    grid_mi = pd.MultiIndex.from_tuples(grid_tuples, names=['x', 'y'])
    idxer = counts.index.get_indexer(grid_mi)  # -1 where (x,y) is missing
    vals = counts.to_numpy()
    transcript_density_list = np.where(idxer >= 0, vals[idxer], 0) # fill 0 where it is missing
    timer.stop()

    xy_transcript_density = np.array(transcript_density_list).reshape(dim_x, dim_y)

    img_extent = sd.get_extent(sdata[image_type], coordinate_system='global')
    imagedim = helperfuncs.ImageDimStruct(img_extent['x'][0], img_extent['y'][0],
                                        img_extent['x'][1], img_extent['y'][1])
    nuclei_centroid_coords = sd.get_centroids(sdata['nucleus_boundaries'], coordinate_system='global').compute()

    # kernel_size = 2 * r + 1

    # Create circular kernel (disk mask)
    y, x = np.ogrid[-kernel_radius:kernel_radius+1, -kernel_radius:kernel_radius+1]
    mask = (x**2 + y**2) <= kernel_radius**2
    kernel = mask.astype(xy_transcript_density.dtype)

    print("[NOTE] Densitiy calculation")
    timer.start()
    xy_kernel_transcript_density = convolve(xy_transcript_density, kernel, mode='constant', cval=0)
    xy_kernel_transcript_density = np.flipud(xy_kernel_transcript_density)
    timer.stop()
    # xy_kernel_transcript_density = xy_kernel_transcript_density.astype(np.uint16) # conversion needed for cv2

    # TODO this plot needs to be checked again.
    if ( figure_path != None ):

        if ( flip ):
            helperfuncs.plot_pixels(
                figure_path,
                xy_kernel_transcript_density,
                imagedim,
                'transcript_density',
                'Transcript Density', 
                'gray',
                True,
                True,
                points=nuclei_centroid_coords
            )
        else:
            helperfuncs.plot_pixels(
                figure_path,
                np.flipud(xy_kernel_transcript_density),
                imagedim,
                'transcript_density',
                'Transcript Density', 
                'gray',
                True,
                True,
                points=nuclei_centroid_coords
            )

    return xy_kernel_transcript_density.flatten()