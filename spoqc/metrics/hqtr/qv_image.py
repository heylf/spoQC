import spatialdata as sd
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd

from scipy.ndimage import convolve

from ... import helperfuncs
from ... import core

def _generate_transcript_quality_density_image(
        sdata,
        figure_path,
        imagedim,
        dim_x,
        dim_y,
        *,
        kernel_radius=3,
        flip=False
):

    timer = helperfuncs.Timer()

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
            )

    return xy_kernel_transcript_density.flatten()


def _transcript_qv_image(
        sdata,
        figure_path,
        spoqc_tmp_folder,
        modality,
        imagedim,
        dim_x,
        dim_y,
        chunk_size,
    ):
    timer = helperfuncs.Timer()

    print("[NOTE] Generate qv image")
    timer.start()
    np_arr = _generate_transcript_quality_density_image(sdata, figure_path, imagedim, dim_x, dim_y)
    image_ddf = dd.from_dask_array(da.from_array(np_arr, chunks=chunk_size), columns=["qv_density"])
    timer.stop()

    print("[NOTE] Generate qv histogram")
    timer.start()
    helperfuncs.plot_histogram_for_array(image_ddf['qv_density'].compute().to_numpy(), 100,
                                         figure_path, "Transcript QV", "transcript_qv")
    timer.stop()

    helperfuncs.ddf_to_parquet(image_ddf, 'qv_density', spoqc_tmp_folder, [], modality)


def init_metric(enterprise):

    # These have to be defined.
    name = "qv_density"
    submetrics = ["qv_density"] # use the name above or fill in further metrics calculated by this metric
    modality = "hqtr"
    needs_metrics = []
    step_when_it_is_calculated = ['all', 'unittest', 'hqtr', 'hqtr_qv']
    loaded_for_analysis = True
    loaded_for_visualization = True

    # These are given my your metric calc function.
    args = [enterprise.cargo.sdata, f'{enterprise.args.output_dir}/hqtr/hqtr_qv/', enterprise.args.tmp_dir, 'hqtr',
            enterprise.cargo.imagedim, enterprise.cargo.dim_x, enterprise.cargo.dim_y, enterprise.args.chunk_size]
    kwargs = None

    metric = core.metric.Metric(
        _transcript_qv_image, 
        name,
        submetrics,
        modality,
        needs_metrics = needs_metrics,
        step_when_it_is_calculated = step_when_it_is_calculated,
        loaded_for_analysis = loaded_for_analysis,
        loaded_for_visualization = loaded_for_visualization,
        args = args,
        kwargs = kwargs,
    )    
    
    return metric