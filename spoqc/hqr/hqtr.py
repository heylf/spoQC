import spatialdata as sd
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd

from scipy.ndimage import convolve
from dask_ml.preprocessing import MinMaxScaler

from .. import image_analysis
from .. import ambient
from .. import helperfuncs

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

    # TODO this plot needs to be checked again.
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
    xy_transcript_coords_df['local_moran_I'] = ambient.qc_ambient.calculate_local_ambient_values(sdata, threads)

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

    # TODO this plot needs to be checked again.
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

    # TODO this plot needs to be checked again.
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


def calc_prob_pixel_stuff_v2(image_ddf, thresh, std, tail, col):

    if std <= 0:
        raise ValueError("std must be > 0")

    inv_std = 1.0 / std
    norm_const = inv_std / np.sqrt(2.0 * np.pi)

    def _part(part: pd.DataFrame) -> pd.Series:
        x = part[col].to_numpy()
        # Gaussian PDF centered at `thresh`
        z = (x - thresh) * inv_std
        pdf = norm_const * np.exp(-0.5 * z * z)

        # Tail overwrite to 1.0 (then we'll invert below)
        if tail == "left":
            pdf = np.where(x < thresh, 1.0, pdf)
        elif tail == "right":
            pdf = np.where(x > thresh, 1.0, pdf)
        # else: no tail tweak

        out = 1.0 - pdf
        return pd.Series(out, index=part.index, name=f"p_{col}")

    p_series = image_ddf.map_partitions(_part, meta=(f"p_{col}", "f8"))
    image_ddf = image_ddf.assign(**{f"p_{col}": p_series})

    # Min-Max normalize using dask-ml
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(image_ddf[[f'p_{col}']])
    scaled_series = scaled_df.iloc[:, 0]

    image_ddf = image_ddf.assign(
        **{f'norm_p_{col}': scaled_series}
    )

    return image_ddf


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
    
    # TODO should I apply zero truncated negative binomial instead?
    # At 10x Genomics they use a threshold of qv < 20 (see 10xBaysor tutorial)
    print("[NOTE] Calculate qv probabilities")
    timer.start()
    image_ddf = calc_prob_pixel_stuff_v2(image_ddf, 20.0, 3, 'left', 'qv_density')
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
    image_ddf = calc_prob_pixel_stuff_v2(image_ddf, 0.4, 1, 'left', 'ac_density')
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


def get_hqtr(sdata, spoqc_tmp_folder, imagedim, dim_x, dim_y, CONST, seed):

    if ( CONST.STEP in ['all', 'hqtr', 'hqtr_metrices'] ):

        image_analysis.structure_analysis.start_image_struc_analyis(
            sdata,
            CONST.FIGURE_PATH,
            spoqc_tmp_folder,
            'hqtr',
            CONST.IMAGE_TYPE,
            CONST.RESOLUTION,
            imagedim,
            dim_x,
            dim_y,
            CONST.OVERWRITE
        )

        print('[finish]')

    if ( CONST.STEP in ['all', 'hqtr', 'hqtr_qv'] ):

        transcript_qv_image(
            sdata,
            CONST.FIGURE_PATH,
            spoqc_tmp_folder,
            'hqtr',
            CONST.IMAGE_TYPE,
            CONST.RESOLUTION,
            dim_x,
            dim_y,
            imagedim,
        )

        print('[finish]')

    if ( CONST.STEP in ['all', 'hqtr', 'hqtr_ac'] ):

        transcript_ac_image(
            sdata,
            CONST.FIGURE_PATH,
            spoqc_tmp_folder,
            'hqtr',
            CONST.THREADS,
            CONST.IMAGE_TYPE,
            CONST.RESOLUTION,
            dim_x,
            dim_y,
            imagedim,
        )

        print('[finish]')


    if ( CONST.STEP in ['all', 'hqtr', 'hqtr_clustering'] ):

        image_analysis.pixel_scoring_dask.start_pixel_qc(
            sdata,
            CONST.FIGURE_PATH,
            spoqc_tmp_folder,
            'hqtr',
            CONST.IMAGE_TYPE,
            CONST.RESOLUTION,
            dim_x,
            dim_y,
            imagedim,
            seed,
            CONST.THREADS
        )

        print("[finish]")


    if ( CONST.STEP in ['all', 'hqtr', 'hqtr_refinement'] ):

        image_analysis.pixel_scoring_refinement.start_pixel_mask_refinement (
                CONST.FIGURE_PATH,
                spoqc_tmp_folder,
                'hqtr',
                dim_x,
                dim_y,
                1.5,
                15
        )

        print('[finish]')

    if ( CONST.STEP in ['all', 'hqtr', 'hqtr_bounding_box'] ):

        image_analysis.bounding_boxes.define_bounding_boxes(
            sdata,
            CONST.FIGURE_PATH,
            spoqc_tmp_folder,
            'hqtr',
            CONST.IMAGE_TYPE,
            CONST.RESOLUTION,
            dim_x,
            dim_y,
            imagedim,
            'raw',
            dilation_radius=1
        )

        print('[finish]')


def celltype_refinement_of_hqtr(sdata, spoqc_tmp_folder, imagedim, dim_x, dim_y, CONST):

    if ( CONST.STEP in ['all', 'hqtr_celltype'] ):
        
        image_analysis.celltype_analysis.start_image_celltype_analysis(
            sdata,
            CONST.FIGURE_PATH,
            spoqc_tmp_folder,
            'hqtr',
            CONST.IMAGE_TYPE,
            CONST.RESOLUTION,
            imagedim,
            dim_x,
            dim_y,
            CONST.ANNOTATION_KEY,
            CONST.CANORM
        )

        print("[finish]")