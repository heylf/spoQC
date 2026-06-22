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

    file_hqcr = ''
    hqcr_belief_name = ''
    hqcr_mask_name = ''
    file_hqpr = ''
    hqpr_belief_name = ''
    hqpr_mask_name = ''
    file_hqtr = ''
    hqtr_belief_name = ''
    hqtr_mask_name = ''
    for type_of_belief in ['_smoothed', '']:
        suf = ''
        if type_of_belief == '_smoothed':
           suf = '_smoothed'
        file_hqcr = f'{spoqc_tmp_folder}/hqcr_output_mask{suf}_{suffix}.parquet'
        hqcr_belief_name = f'hqcr_beliefs{type_of_belief}'
        hqcr_mask_name = f'hqcr_mask{type_of_belief}'
        file_hqpr = f'{spoqc_tmp_folder}/hqpr_{staining}_output_mask{suf}_{suffix}'
        hqpr_belief_name = f"hqpr_{staining}_beliefs{type_of_belief}"
        hqpr_mask_name = f"hqpr_{staining}_mask{type_of_belief}"
        file_hqtr = f'{spoqc_tmp_folder}/hqtr_output_mask{suf}_{suffix}'
        hqtr_belief_name = f'hqtr_beliefs{type_of_belief}'
        hqtr_mask_name = f'hqtr_mask{type_of_belief}'


        hqcr_mask = pd.read_parquet(file_hqcr)
        hqcr_mask[hqcr_belief_name] = np.array(hqcr_mask[hqcr_belief_name]).reshape(dim_x, dim_y).flatten()
        hqcr_mask[hqcr_mask_name] = np.array(hqcr_mask[hqcr_mask_name]).reshape(dim_x, dim_y).flatten()
        
        hqpr_mask = dd.read_parquet(
            file_hqpr, 
            columns=[hqpr_belief_name,hqpr_mask_name], engine="pyarrow"
        )
        hqtr_mask = dd.read_parquet(
            file_hqtr,
            columns=[hqtr_belief_name, hqtr_mask_name], engine="pyarrow"
        )

        mask_df = pd.DataFrame({
            'hqcr_mask': hqcr_mask[hqcr_mask_name],
            f'hqpr_{staining}_mask': hqpr_mask[hqpr_mask_name].compute().to_numpy(),
            'hqtr_mask': hqtr_mask[hqtr_mask_name].compute().to_numpy()
        })

        beliefs_df = pd.DataFrame({
            'hqcr_beliefs': hqcr_mask[hqcr_belief_name],
            f'hqpr_{staining}_beliefs': hqpr_mask[hqpr_belief_name].compute().to_numpy(),
            'hqtr_beliefs': hqtr_mask[hqtr_belief_name].compute().to_numpy()
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
                f'{m}_zoom{type_of_belief}',
                f'{m}_zoom{type_of_belief}',
                'gray',
                False,
                True,
                legend_dict={f"{m}": "#FFFFFF", "low Q": "#000000"}
            )

            helperfuncs.plot_pixels(
                figure_path,
                np.array(beliefs_df[f'{m}_beliefs']).reshape(dim_x, dim_y)[y_1:y_2, x_1_org:x_2_org],
                imagedim_zoom,
                f'{m}_beliefs_zoom{type_of_belief}',
                f'{m}_beliefs_zoom{type_of_belief}',
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
            f'final_zoom{type_of_belief}',
            f'final_zoom{type_of_belief}',
            cmap,
            False,
            True,
            legend_dict={"no mask": "#000000", "1 mask": "#0000FF", "2 masks": "#008000", "all masks": "#FFFF00"}
        )

        combined_beliefs = beliefs_df['hqcr_beliefs_smoothed'] + beliefs_df[f'hqpr_{staining}_beliefs_smoothed'] + beliefs_df['hqtr_beliefs_smoothed']
        combined_beliefs /= 3

        helperfuncs.plot_pixels(
            figure_path,
            np.array(combined_beliefs).reshape(dim_x, dim_y)[y_1:y_2, x_1_org:x_2_org],
            imagedim_zoom,
            f'combined_beliefs{type_of_belief}', 
            f'combined_beliefs{type_of_belief}', 
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
                np.flipud(values[y_1_org:y_2_org, x_1_org:x_2_org]),
                imagedim_zoom,
                f'input_segmentation_{seg}_zoom{type_of_belief}',
                f'input_segmentation_{seg}_zoom{type_of_belief}',
                'gray',
                False,
                True,
                legend_dict={"mask": "#FFFFFF", "empty": "#000000"}
            )

        for modality in ['hqpr', 'hqtr']:

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
                name = f'{name}_transcript_densities_zoom{type_of_belief}'
            elif ( modality == 'hqpr' ):
                name = f'{name}_pixel_intensities_zoom{type_of_belief}'
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