import spatialdata as sd
import numpy as np

from scipy.ndimage import convolve

from ... import helperfuncs
from ...core import groupreduce, threads, transcripts


def disk_density(image, kernel_radius, workers):
    """
    np.flipud(convolve(image, disk, mode='constant', cval=0)) with the circular 0/1 kernel of
    kernel_radius, in image's dtype, as a C-contiguous array.

    Row blocks are convolved on `workers` threads, each with kernel_radius halo rows (real
    neighbours inside the image, the constant 0 beyond it), so every pixel sums the same terms
    in the same order as one convolve of the whole image.
    """
    y, x = np.ogrid[-kernel_radius:kernel_radius+1, -kernel_radius:kernel_radius+1]
    mask = (x**2 + y**2) <= kernel_radius**2
    kernel = mask.astype(image.dtype)
    n_rows = image.shape[0]
    flipped = np.empty_like(image)

    def block(rows):
        top, bottom = max(rows.start - kernel_radius, 0), min(rows.stop + kernel_radius, n_rows)
        density = convolve(image[top:bottom], kernel, mode='constant', cval=0)
        flipped[n_rows - rows.stop:n_rows - rows.start] = density[rows.start - top:rows.stop - top][::-1]

    threads.map_slices(block, n_rows, max(-(-n_rows // (4 * workers)), 1), workers)
    return flipped


def transcript_pixel_groups(sdata, imagedim):
    """
    The transcripts grouped by the pixel of imagedim's grid their truncated global (x, y) falls on
    (groupreduce.pixel_groups), plus the grid's pixel count.

    Returns (pixels, rows, offsets, n_pixels); pixel order is that of
    MultiIndex.from_product([range(bb_ymin, bb_ymax), range(bb_xmin, bb_xmax)]).
    """
    transcript_coords_df = transcripts.global_coordinates(sdata).to_pandas()
    transcript_coords_df = transcript_coords_df.astype(int)
    x_range = (int(imagedim.bb_xmin), int(imagedim.bb_xmax))
    y_range = (int(imagedim.bb_ymin), int(imagedim.bb_ymax))
    n_pixels = max(x_range[1] - x_range[0], 0) * max(y_range[1] - y_range[0], 0)
    pixels, rows, offsets = groupreduce.pixel_groups(
        transcript_coords_df['x'].to_numpy(), transcript_coords_df['y'].to_numpy(), x_range, y_range
    )
    return pixels, rows, offsets, n_pixels

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

    print("[NOTE] Translate cooridnates")
    timer.start()
    pixels, _, offsets, n_pixels = transcript_pixel_groups(sdata, imagedim)
    transcript_density_list = groupreduce.to_grid(pixels, np.diff(offsets), n_pixels, 0)
    timer.stop()

    xy_transcript_density = transcript_density_list.reshape(dim_x, dim_y)

    print("[NOTE] Densitiy calculation")
    timer.start()
    xy_kernel_transcript_density = disk_density(xy_transcript_density, kernel_radius, threads.budget())
    del transcript_density_list, xy_transcript_density
    timer.stop()
    # xy_kernel_transcript_density = xy_kernel_transcript_density.astype(np.uint16) # conversion needed for cv2

    if ( figure_path != None ):
        img_extent = sd.get_extent(sdata[image_type], coordinate_system='global')
        imagedim = helperfuncs.ImageDimStruct(img_extent['x'][0], img_extent['y'][0],
                                            img_extent['x'][1], img_extent['y'][1])
        nuclei_centroid_coords = sd.get_centroids(sdata['nucleus_boundaries'], coordinate_system='global').compute()

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

    return xy_kernel_transcript_density.ravel()
