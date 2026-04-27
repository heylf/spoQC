import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt

from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk

from .. import helperfuncs
from .. import hqr
from .. import metrics

def _overlap(a, b):
    # boxes: [min_row, min_col, max_row, max_col]
    return (a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1])

def _merge_two(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]

def _merge_overlapping_boxes(boxes):
    # Iteratively merge until no overlaps remain
    boxes = [list(map(int, box)) for box in boxes]
    changed = True
    while changed:
        changed = False
        used = [False] * len(boxes)
        new_boxes = []
        for i in range(len(boxes)):
            if used[i]:
                continue
            curr = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if _overlap(curr, boxes[j]):
                    curr = _merge_two(curr, boxes[j])
                    used[j] = True
                    changed = True
            used[i] = True
            new_boxes.append(curr)
        boxes = new_boxes
    return boxes

def _boudning_box_plot(bounding_boxes, figure_path, suffix, image, imagedim, flip = False):
    plt.figure(figsize=(12, 6))

    if ( flip ):
        plt.imshow(
            np.flipud( np.log10 (image + 1) ),
            cmap='gray',
            extent=[imagedim.bb_xmin, imagedim.bb_xmax, imagedim.bb_ymin, imagedim.bb_ymax],
            aspect='equal'
        )
    else:
        plt.imshow(
            np.log10 (image + 1),
            cmap='gray',
            extent=[imagedim.bb_xmin, imagedim.bb_xmax, imagedim.bb_ymin, imagedim.bb_ymax],
            aspect='equal'
        )
    plt.title(f"Refined Image Log10p1 with Subfigures")

    for bbox in bounding_boxes:
        min_row, min_col, max_row, max_col = bbox

        # Draw the flipped rectangle
        if ( flip ):
            plt.plot(
                [min_col, min_col, max_col, max_col, min_col],
                [min_row, max_row, max_row, min_row, min_row],
                color="red",
                linewidth=2,
            )
        else:
            plt.plot(
                [min_col, max_col, max_col, min_col, min_col],  # x (columns)
                [min_row, min_row, max_row, max_row, min_row],  # y (rows)
                color="red",
                linewidth=2,
            )
        
    plt.savefig(f'{figure_path}/imageplot_{suffix}.png', bbox_inches='tight', dpi=300)
    plt.close()


def define_bounding_boxes(
        sdata,
        figure_path,
        spoqc_tmp_folder,
        modality,
        image_type,
        resolution,
        dim_x,
        dim_y, 
        imagedim,
        suffix,
        *,
        dilation_radius=10,
        minum_num_pixel=100_000,
        staining=None,
        flip=False
    ):

    prefix = modality
    if ( staining ):
        figure_path = f'{figure_path}/{modality}/{staining}/{modality}_bounding_box/'
        prefix = f'{modality}_{staining}'
    else:
        figure_path = f'{figure_path}/{modality}/{modality}_bounding_box/'

    image = None
    if ( staining ):
        image = sdata[image_type][resolution].image.values[int(staining)]
    else:
        image = sdata[image_type][resolution].image.values[0]
    image = np.flipud(image)
    intensities = image.flatten()

    if ( modality == 'hqtr' ):
        # Intensities already flipped
        intensities = metrics.transcript_density.transcript_density_image.generate_transcript_density_image(
            sdata,
            figure_path,
            imagedim,
            image_type,
            resolution
        )
        image = intensities.reshape(dim_x, dim_y)

    mask = dd.read_parquet(f'{spoqc_tmp_folder}/{prefix}_output_mask_{suffix}', 
                           columns=[f"{prefix}_mask"], engine="pyarrow")

    # Convert DataFrame to a NumPy array for processing
    binary_image = mask[f"{prefix}_mask"].compute().to_numpy().reshape(dim_x, dim_y)

    # Apply dilation to merge nearby regions
    structuring_element = disk(dilation_radius)
    dilated_image = dilation(binary_image, structuring_element)

    plt.figure(figsize=(12, 6))
    if ( flip ):
        plt.imshow(
            np.flipud( dilated_image ),
            cmap='gray',
            extent=[imagedim.bb_xmin, imagedim.bb_xmax, imagedim.bb_ymin, imagedim.bb_ymax],
            aspect='equal'
        )
    else:
        plt.imshow(
            dilated_image,
            cmap='gray',
            extent=[imagedim.bb_xmin, imagedim.bb_xmax, imagedim.bb_ymin, imagedim.bb_ymax],
            aspect='equal'
        )
    plt.title(f"Dilated image")
    plt.savefig(f'{figure_path}/imageplot_dilated_image_for_bounding_box.png', bbox_inches='tight', dpi=300)
    plt.close()


    # Label connected components in the dilated binary image
    labeled_image = label(dilated_image)

    # Extract subfigures based on connected components
    subfigures = []
    bounding_boxes = []
    idx = 0
    for region in regionprops(labeled_image):
        # Get bounding box for the region
        min_row, min_col, max_row, max_col = region.bbox

        # Calculate number of pixels
        num_pixels = (max_row - min_row) * (max_col - min_col)

        if ( num_pixels > minum_num_pixel ):
            # Extract the subfigure with minimal background
            subfigure = image[min_row:max_row, min_col:max_col]
            subfigures.append(subfigure)
            bounding_boxes.append([min_row, min_col, max_row, max_col])
            subfigure_imagedim = helperfuncs.ImageDimStruct(min_row, min_col, max_row, max_col)
            helperfuncs.plot_pixels(
                f'{figure_path}/subfigures/',
                subfigure,
                subfigure_imagedim,
                f'subfigure{idx+1}',
                f'Log10p1 Subfigure {idx+1}', 
                'gray',
                True,
                True
            )
            idx += 1

    helperfuncs.plot_pixels(
        f'{figure_path}/subfigures/',
        image,
        imagedim,
        f'subfigure{idx+1}', 
        f'Log10p1 Subfigure {idx+1}', 
        'gray',
        True,
        False
    )

    # Correct the coordinates of the bounding box
    for i, box in enumerate(bounding_boxes):
        bounding_boxes[i] = [box[0]+imagedim.bb_ymin, box[1]+imagedim.bb_xmin, 
                             box[2]+imagedim.bb_ymin, box[3]+imagedim.bb_xmin]

    # Merge overlapping bounding boxes
    merged_bounding_boxes = _merge_overlapping_boxes(bounding_boxes)

    # Plots
    _boudning_box_plot(bounding_boxes, figure_path, 'marked_subfigures', image, imagedim)
    _boudning_box_plot(merged_bounding_boxes, figure_path, 'marked_merged_subfigures', image, imagedim)
    
    # write out txt that can be used later and saved in sdata.attrs or anndata.uns
    metadata_file = f"{figure_path}/{modality}s.txt"
    if ( modality == 'hqpr' ):
        metadata_file = f"{figure_path}/{modality}s_{staining}.txt"
    with open(metadata_file, "w") as f:
        f.write(str(bounding_boxes))

    return bounding_boxes