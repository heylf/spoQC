import spatialdata as sd
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd
import os

from scipy.ndimage import convolve

from .. import helperfuncs

def generate_transcript_density_image(
        sdata,
        figure_path,
        spoqc_tmp_folder,
        imagedim,
        dim_x,
        dim_y,
        overwrite,
        *,
        chunk_size=10000,
        kernel_radius=3,
        flip=False
):

    timer = helperfuncs.Timer()

    tmp_path = f"{spoqc_tmp_folder}/hqtr_output_transcript_density"
    if not os.path.exists(tmp_path):
    #if not os.path.exists(tmp_path) and not overwrite:

        print("[NOTE] Generate transcript density image")

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

        np_arr = xy_kernel_transcript_density.flatten()
        image_ddf = dd.from_dask_array(da.from_array(np_arr, chunks=chunk_size), columns=["transcript_density"])
        helperfuncs.ddf_to_parquet(image_ddf, 'hqtr', spoqc_tmp_folder, [], 'transcript_density')
        return np_arr
    else:
        print("[NOTE] Load transcript density image")

        ddf = dd.read_parquet(
            tmp_path,
            columns=["transcript_density"],
            engine="pyarrow",
            calculate_divisions=True,
        )
        return ddf[f"transcript_density"].compute().to_numpy()