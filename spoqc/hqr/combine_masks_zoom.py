import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
import sys

from matplotlib.colors import LinearSegmentedColormap
from matplotlib_venn import venn3

from .. import helperfuncs
from .. import metrics

def start_combining_masks(
        sdata,
        figure_path,
        spoqc_tmp_folder,
        image_type,
        resolution,
        imagedim,
        dim_x,
        dim_y,
        staining,
        *,
        celltype_refined=False
):

    x_1_org = 18000
    y_1_org = 0
    x_2_org = 22000
    y_2_org = 2500

    imagedim_zoom = helperfuncs.ImageDimStruct(
        imagedim.bb_xmin + x_1_org,
        imagedim.bb_ymin + y_1_org,
        imagedim.bb_xmin + x_2_org,
        imagedim.bb_ymin + y_2_org
    )

    figure_path = f"{figure_path}/combine_masks_zoom/"

    suffix = 'raw'
    if ( celltype_refined ):
        suffix = 'celltype_refined'

    hqcr_mask = pd.read_parquet(f'{spoqc_tmp_folder}/hqcr_output_mask_{suffix}.parquet')
    hqcr_mask['hqcr_beliefs'] = np.array(hqcr_mask['hqcr_beliefs']).reshape(dim_x, dim_y).flatten()
    hqcr_mask['hqcr_mask'] = np.array(hqcr_mask['hqcr_mask']).reshape(dim_x, dim_y).flatten()
    
    hqpr_mask = dd.read_parquet(f'{spoqc_tmp_folder}/hqpr_{staining}_output_mask_{suffix}', 
                                columns=[f"hqpr_{staining}_beliefs", f"hqpr_{staining}_mask"], engine="pyarrow")
    hqtr_mask = dd.read_parquet(f'{spoqc_tmp_folder}/hqtr_output_mask_{suffix}',
                                columns=["hqtr_beliefs", "hqtr_mask"], engine="pyarrow")

    mask_df = pd.DataFrame({
        'hqcr_mask': hqcr_mask['hqcr_mask'],
        f'hqpr_{staining}_mask': hqpr_mask[f"hqpr_{staining}_mask"].compute().to_numpy(),
        'hqtr_mask': hqtr_mask["hqtr_mask"].compute().to_numpy()
    })

    beliefs_df = pd.DataFrame({
        'hqcr_beliefs': hqcr_mask['hqcr_beliefs'],
        f'hqpr_{staining}_beliefs': hqpr_mask[f"hqpr_{staining}_beliefs"].compute().to_numpy(),
        'hqtr_beliefs': hqtr_mask["hqtr_beliefs"].compute().to_numpy()
    })


    y_1 = dim_x - 1 - y_2_org
    y_2 = dim_x - 1

    final_mask = np.zeros(dim_x*dim_y)
    for m in ['hqcr', f'hqpr_{staining}', 'hqtr']:
        final_mask += np.array(mask_df[f'{m}_mask'])

        helperfuncs.plot_pixels(
            figure_path,
            np.array(mask_df[f'{m}_mask']).reshape(dim_x, dim_y)[y_1:y_2, x_1_org:x_2_org],
            imagedim_zoom,
            f'{m}_zoom',
            f'{m}_zoom',
            'gray',
            False,
            True,
            legend_dict={f"{m}": "#FFFFFF", "low Q": "#000000"}
        )

        helperfuncs.plot_pixels(
            figure_path,
            np.array(beliefs_df[f'{m}_beliefs']).reshape(dim_x, dim_y)[y_1:y_2, x_1_org:x_2_org],
            imagedim_zoom,
            f'{m}_beliefs_zoom',
            f'{m}_beliefs_zoom',
            'hot',
            False,
            False
        )

    colors = [
        (0.0, 'black'),   # 0
        (1/3, 'blue'),    # 1
        (2/3, 'green'),   # 2
        (1.0, 'yellow')   # 3
    ]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    helperfuncs.plot_pixels(
        figure_path,
        final_mask.reshape(dim_x, dim_y)[y_1:y_2, x_1_org:x_2_org],
        imagedim_zoom,
        'final_zoom',
        'final_zoom',
        cmap,
        False,
        True,
        legend_dict={"no mask": "#000000", "1 mask": "#0000FF", "2 masks": "#008000", "all masks": "#FFFF00"}
    )

    combined_beliefs = beliefs_df['hqcr_beliefs'] + beliefs_df[f'hqpr_{staining}_beliefs'] + beliefs_df['hqtr_beliefs']
    combined_beliefs /= 3

    helperfuncs.plot_pixels(
        figure_path,
        np.array(combined_beliefs).reshape(dim_x, dim_y)[y_1:y_2, x_1_org:x_2_org],
        imagedim,
        f'combined_beliefs', 
        f'combined_beliefs', 
        'hot',
        False,
        False
    )


    # Generate the input figures again
    for seg in ['cell_labels', 'nucleus_labels']:
        values = sdata.labels[seg][resolution].image.values
        values = (values > 0.0).astype(np.uint8)
        helperfuncs.plot_pixels(
            figure_path,
            values[y_1_org:y_2_org, x_1_org:x_2_org],
            imagedim_zoom,
            f'input_segmentation_{seg}_zoom',
            f'input_segmentation_{seg}_zoom',
            'gray',
            False,
            True,
            legend_dict={"mask": "#FFFFFF", "empty": "#000000"}
        )

    for modality in [f'hqpr', 'hqtr']:

        if modality == 'hqtr':
            staining = None
        else:
            staining = '0'

        if ( staining ):
            spoqc_tmp_folder = f'{spoqc_tmp_folder}/metrices/{modality}/{staining}/'
        else:
            spoqc_tmp_folder = f'{spoqc_tmp_folder}/metrices/{modality}'

        xy_intensities = None
        intensities = None
        if ( modality == 'hqtr' ):
            # Intensities already flipped
            intensities = metrics.transcript_density.transcript_density_image.generate_transcript_density_image(
                sdata,
                figure_path,
                imagedim,
                image_type,
                resolution
            )
            xy_intensities = intensities.reshape(dim_x, dim_y)
        else:
            xy_intensities = sdata[image_type][resolution].image.values[int(staining)]
            xy_intensities = np.flipud(xy_intensities)
            intensities = xy_intensities.flatten()
        
        # Plot intensities
        name = 'input'
        if ( modality == 'hqtr' ):
            name = f'{name}_transcript_densities_zoom'
        elif ( modality == 'hqpr' ):
            name = f'{name}_pixel_intensities_zoom'
        else:
            sys.exit('[ERROR] Modality not supported')

        helperfuncs.plot_pixels(
            figure_path,
            np.log10(xy_intensities + 1)[y_1:y_2, x_1_org:x_2_org],
            imagedim_zoom,
            name,
            name,
            'gray',
            False,
            False
        )