import spatialdata as sd
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd

from scipy.ndimage import convolve
from dask_ml.preprocessing import MinMaxScaler

from .. import image_analysis
from .. import helperfuncs

def generate_transcript_density_image(sdata, figure_path, imagedim, image_type, resolution, kernel_radius=3):

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
    timer.stop()
    # xy_kernel_transcript_density = xy_kernel_transcript_density.astype(np.uint16) # conversion needed for cv2

    # TODO this plot needs to be checked again.
    if ( figure_path != None ):
        # Check up plot
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

    return xy_kernel_transcript_density.flatten()


def generate_transcript_quality_density_image(sdata, figure_path, imagedim, image_type, resolution, kernel_radius=3):

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
    timer.stop()
    #xy_kernel_transcript_density = xy_kernel_transcript_density.astype(np.uint16) # conversion needed for cv2

    # TODO this plot needs to be checked again.
    if ( figure_path != None ):
        # Check up plot
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

def generate_transcript_quality_density_image_memopt(
    sdata,
    figure_path,
    imagedim,
    image_type,
    resolution,
    kernel_radius=3
):
    """
    Fully-Dask version (no pandas).
    Lazily computes per-pixel mean QV, convolves with a circular kernel via map_overlap,
    and only computes to NumPy when plotting.
    Returns a Dask array (flattened).
    """

    timer = helperfuncs.Timer()

    # image dims (these are small ints, fine to be eager)
    dim_x = len(sdata[image_type][resolution].image.y.values)
    dim_y = len(sdata[image_type][resolution].image.x.values)

    # centroids and qv as Dask DataFrame/Array
    # NOTE: we avoid .compute() here; keep it lazy
    # sd.get_centroids(...) is assumed to return a Dask DataFrame
    ddf_centroids = sd.get_centroids(sdata['transcripts'], coordinate_system='global')[['x', 'y']]
    ddf_centroids = ddf_centroids.astype({'x': 'int64', 'y': 'int64'})

    # qv column as Dask Series (same index alignment assumed)
    # if transcripts is a Dask DataFrame, this works; otherwise adapt to your container
    qv_series = sdata['transcripts']['qv']

    # attach qv lazily
    ddf = ddf_centroids.assign(qv=qv_series)

    # Boundaries and grid size (use the bounding box imagedim passed in)
    xmin = int(imagedim.bb_xmin)
    xmax = int(imagedim.bb_xmax)
    ymin = int(imagedim.bb_ymin)
    ymax = int(imagedim.bb_ymax)

    width  = int(xmax - xmin)
    height = int(ymax - ymin)

    # Keep only in-bounds points (lazy filter)
    ddf = ddf[(ddf['x'] >= xmin) & (ddf['x'] < xmax) & (ddf['y'] >= ymin) & (ddf['y'] < ymax)]

    # Flat index per point: i = (y - ymin) * width + (x - xmin)
    # Convert to Dask Arrays
    x_da  = ddf['x'].to_dask_array(lengths=True)
    y_da  = ddf['y'].to_dask_array(lengths=True)
    qv_da = ddf['qv'].to_dask_array(lengths=True).astype('float64')

    flat_idx = (y_da - ymin) * width + (x_da - xmin)
    flat_idx = flat_idx.astype('int64')

    print("[NOTE] Calcualte pixel mean")
    timer.start()

    # Sum of qv per flat pixel and count per flat pixel via bincount
    minlength = width * height
    qv_sums   = da.bincount(flat_idx, weights=qv_da, minlength=minlength)
    qv_counts = da.bincount(flat_idx, minlength=minlength).astype('float64')

    # Mean per pixel, fill missing with 0.0
    qv_means_flat = da.where(qv_counts > 0, qv_sums / qv_counts, 0.0)

    # Reshape to (height, width); original code reshaped to (dim_x, dim_y) where
    # dim_x corresponds to y-axis length and dim_y to x-axis length.
    # Here, height = len(y), width = len(x) — same orientation.
    xy_transcript_density = qv_means_flat.reshape((height, width))

    timer.stop()

    # Use provided image extent to update imagedim (unchanged from your code)
    img_extent = sd.get_extent(sdata[image_type], coordinate_system='global')
    imagedim = helperfuncs.ImageDimStruct(
        img_extent['x'][0], img_extent['y'][0],
        img_extent['x'][1], img_extent['y'][1]
    )

    # nuclei centroids for plotting points (plotting needs concrete coords -> compute later)
    nuclei_centroid_coords_ddf = sd.get_centroids(sdata['nucleus_boundaries'], coordinate_system='global')[['x', 'y']]

    # Create circular kernel (NumPy small array is fine to be eager)
    yk, xk = np.ogrid[-kernel_radius:kernel_radius+1, -kernel_radius:kernel_radius+1]
    mask = (xk**2 + yk**2) <= kernel_radius**2
    kernel = mask.astype(np.float64)

    print("[NOTE] Densitiy calculation")
    timer.start()

    # Convolution with overlap using scipy.ndimage on each block
    def _convolve_block(block):
        # block is a NumPy ndarray chunk
        return convolve(block, kernel, mode='constant', cval=0.0)

    # depth equals kernel_radius in both axes for proper overlap
    xy_kernel_transcript_density = da.map_overlap(
        _convolve_block,
        xy_transcript_density,
        depth=(kernel_radius, kernel_radius),
        boundary=0,  # zeros outside
        dtype=xy_transcript_density.dtype
    )

    timer.stop()

    # Plot (only if requested). Compute just what's needed for the figure.
    if figure_path is not None:
        # Compute arrays for plotting
        img_np = xy_kernel_transcript_density.compute()

        print(img_np.shape())

        # nuclei points (as NumPy for the scatter overlay)
        nuclei_pts_np = nuclei_centroid_coords_ddf.compute().to_numpy()

        helperfuncs.plot_pixels(
            figure_path,
            img_np,
            imagedim,
            'transcript_qv_density',
            'Transcript QV Density',
            'gray',
            True,
            True,
            points=nuclei_pts_np
        )

    # Return a Dask array (flattened) to keep things lazy/downstream-Dask-friendly
    return xy_kernel_transcript_density.ravel()



# We are calculating a kernel density at the end so you will not have your usual [-1,1] autocorraltion values.
def generate_transcript_ambient_density_image(sdata, figure_path, imagedim, global_ambient, image_type, resolution, 
                                              kernel_radius=3):

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

    timer.start()
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
    # xy_kernel_transcript_density = xy_kernel_transcript_density.astype(np.uint16) # conversion needed for cv2

    # TODO this plot needs to be checked again.
    if ( figure_path != None ):
        # Check up plot
        helperfuncs.plot_pixels(
            figure_path,
            xy_kernel_transcript_density,
            imagedim,
            'transcript_autocorrelation_density',
            'Transcript Autocorrelation Density', 
            'gray',
            True,
            True,
            points=nuclei_centroid_coords
        )

    return xy_kernel_transcript_density.flatten()


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
    image_ddf = calc_prob_pixel_stuff_v2(image_ddf, 20, 3, 'left', 'qv_density')
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
    np_arr = generate_transcript_ambient_density_image(sdata, figure_path, imagedim, global_ambient, image_type, 
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
    image_ddf = calc_prob_pixel_stuff_v2(image_ddf, 1, 0.5, 'left', 'ac_density')
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