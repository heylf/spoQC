import numpy as np
import matplotlib.pyplot as plt

from .. import helperfuncs
from spoqc.core import raster
from spoqc.core import figures
from spoqc.core.figures import save_figure

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
        figures.imshow(
            plt.gca(),
            np.flipud( np.log10 (image + 1) ),
            dpi=300,
            cmap='gray',
            extent=[imagedim.bb_xmin, imagedim.bb_xmax, imagedim.bb_ymin, imagedim.bb_ymax],
            aspect='equal'
        )
    else:
        figures.imshow(
            plt.gca(),
            np.log10 (image + 1),
            dpi=300,
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
            print("[NOTE] Not flipping red boxes")
            plt.plot(
                [min_col, min_col, max_col, max_col, min_col],
                [imagedim.bb_ymin + imagedim.bb_ymax - min_row,
                 imagedim.bb_ymin + imagedim.bb_ymax - max_row,
                 imagedim.bb_ymin + imagedim.bb_ymax - max_row,
                 imagedim.bb_ymin + imagedim.bb_ymax - min_row,
                 imagedim.bb_ymin + imagedim.bb_ymax - min_row],
                color="red",
                linewidth=2,
            )

    save_figure(plt.gcf(), f'{figure_path}/imageplot_{suffix}.png', f'{figure_path}/imageplot_{suffix}.pdf', bbox_inches='tight', dpi=300)
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
        threads,
        *,
        dilation_radius=10,
        minum_num_pixel=100_000,
        staining=None,
        flip=False
    ):

    prefix = modality
    if ( staining ):
        figure_path = f'{figure_path}/{modality}/{modality}_bounding_box/{staining}/'
        prefix = f'{modality}_{staining}'
    else:
        figure_path = f'{figure_path}/{modality}/{modality}_bounding_box/'

    # Flipped intensity image; for hqtr the transcript density image saved by the metrices step.
    image = raster.load_intensity_image(
        sdata, spoqc_tmp_folder, modality, image_type, resolution, dim_x, dim_y, threads, staining=staining
    )

    binary_image = raster.read_pixel_columns(
        f'{spoqc_tmp_folder}/{prefix}_output_mask_smoothed_{suffix}', [f"{prefix}_mask_smoothed"], dim_x * dim_y, threads
    )[f"{prefix}_mask_smoothed"].reshape(dim_x, dim_y)

    # Apply dilation to merge nearby regions
    dilated_image = raster.dilate_disk(binary_image, dilation_radius)

    plt.figure(figsize=(12, 6))
    if ( flip ):
        figures.imshow(
            plt.gca(),
            np.flipud( dilated_image ),
            dpi=300,
            cmap='gray',
            extent=[imagedim.bb_xmin, imagedim.bb_xmax, imagedim.bb_ymin, imagedim.bb_ymax],
            aspect='equal'
        )
    else:
        figures.imshow(
            plt.gca(),
            dilated_image,
            dpi=300,
            cmap='gray',
            extent=[imagedim.bb_xmin, imagedim.bb_xmax, imagedim.bb_ymin, imagedim.bb_ymax],
            aspect='equal'
        )
    plt.title(f"Dilated image")
    save_figure(plt.gcf(), f'{figure_path}/imageplot_dilated_image_for_bounding_box.png', f'{figure_path}/imageplot_dilated_image_for_bounding_box.pdf', bbox_inches='tight', dpi=300)
    plt.close()


    # Bounding boxes of the connected components of the dilated binary image, in label order
    boxes = raster.component_boxes(dilated_image, threads)
    num_pixels = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    kept_boxes = boxes[num_pixels > minum_num_pixel]

    # Extract the subfigures with minimal background
    for idx, (min_row, min_col, max_row, max_col) in enumerate(kept_boxes.tolist()):
        subfigure = image[min_row:max_row, min_col:max_col]
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
    idx = len(kept_boxes)

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
    offset = np.array([imagedim.bb_ymin, imagedim.bb_xmin, imagedim.bb_ymin, imagedim.bb_xmin], dtype=np.float64)
    bounding_boxes = (kept_boxes + offset).tolist()

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