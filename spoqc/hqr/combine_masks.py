import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
import os

from matplotlib.colors import LinearSegmentedColormap
from matplotlib_venn import venn3

from .. import helperfuncs

def start_combining_masks(
        figure_path,
        spoqc_tmp_folder,
        imagedim,
        dim_x,
        dim_y,
        staining,
        *,
        celltype_refined=False
):

    figure_path = f"{figure_path}/combine_masks/{staining}"

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

        final_mask = np.zeros(dim_x*dim_y)
        for m in ['hqcr', f'hqpr_{staining}', 'hqtr']:
            final_mask += np.array(mask_df[f'{m}_mask'])

            helperfuncs.plot_pixels(
                figure_path,
                np.array(mask_df[f'{m}_mask']).reshape(dim_x, dim_y),
                imagedim,
                f'{m}_mask{type_of_belief}', 
                f'{m}_mask{type_of_belief}', 
                'gray',
                False,
                True,
                legend_dict={f"{m}": "#FFFFFF", "low Q": "#000000"}
            )

            helperfuncs.plot_pixels(
                figure_path,
                np.array(beliefs_df[f'{m}_beliefs']).reshape(dim_x, dim_y),
                imagedim,
                f'{m}_beliefs{type_of_belief}', 
                f'{m}_beliefs{type_of_belief}', 
                'hot',
                False,
                False
            )

            # --- general histograms ---
            fig = None
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(np.array(beliefs_df[f'{m}_beliefs']), bins=50)
            ax.set_yscale('log')
            ax.set_xlabel(f'{m}_beliefs')
            ax.set_ylabel('Log count')
            ax.set_title(f'Distribution of {m} beliefs')
            fig.savefig(os.path.join(figure_path, f'hist_{m}_beliefs{type_of_belief}_log.png'), bbox_inches='tight')
            fig.savefig(os.path.join(figure_path, f'hist_{m}_beliefs{type_of_belief}_log.pdf'), bbox_inches='tight')
            plt.close(fig)

            fig = None
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(np.array(beliefs_df[f'{m}_beliefs']), bins=50)
            ax.set_xlabel(f'{m}_beliefs')
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution of {m} beliefs')
            fig.savefig(os.path.join(figure_path, f'hist_{m}_beliefs{type_of_belief}.png'), bbox_inches='tight')
            fig.savefig(os.path.join(figure_path, f'hist_{m}_beliefs{type_of_belief}.pdf'), bbox_inches='tight')
            plt.close(fig)


        colors = [
            (0.0, 'black'),   # 0
            (1/3, 'blue'),    # 1
            (2/3, 'green'),   # 2
            (1.0, 'yellow')   # 3
        ]
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

        helperfuncs.plot_pixels(
            figure_path,
            final_mask.reshape(dim_x, dim_y),
            imagedim,
            f'combined_masks{type_of_belief}',
            f'combined_masks{type_of_belief}',
            cmap,
            False,
            True,
            legend_dict={"no mask": "#000000", "1 mask": "#0000FF", "2 masks": "#008000", "all masks": "#FFFF00"}
        )

        # How much agreement is between the maps?
        # Define the sizes of the three sets and their intersections
        subsets={
                '100': np.sum(((mask_df['hqcr_mask'] - mask_df[f'hqpr_{staining}_mask'] - mask_df['hqtr_mask']) == 1).astype(np.uint8)),
                '010': np.sum(((mask_df[f'hqpr_{staining}_mask'] - mask_df['hqcr_mask'] - mask_df['hqtr_mask']) == 1).astype(np.uint8)),
                '001': np.sum(((mask_df['hqtr_mask'] - mask_df[f'hqpr_{staining}_mask'] - mask_df['hqcr_mask']) == 1).astype(np.uint8)),
                '110': np.sum(((mask_df['hqcr_mask'] + mask_df[f'hqpr_{staining}_mask'] - mask_df['hqtr_mask']) == 2).astype(np.uint8)),
                '101': np.sum(((mask_df['hqcr_mask'] - mask_df[f'hqpr_{staining}_mask'] + mask_df['hqtr_mask']) == 2).astype(np.uint8)),
                '011': np.sum(((- mask_df['hqcr_mask'] + mask_df[f'hqpr_{staining}_mask'] + mask_df['hqtr_mask']) == 2).astype(np.uint8)),
                '111': np.sum(((mask_df['hqcr_mask'] + mask_df[f'hqpr_{staining}_mask'] + mask_df['hqtr_mask']) == 3).astype(np.uint8))
        }

        covered = 0
        for key in subsets:
            subsets[key] = np.round(subsets[key] / (dim_x * dim_y), 3)
            covered += subsets[key]

        uncovered = np.sum(((mask_df['hqcr_mask'] + mask_df[f'hqpr_{staining}_mask'] + mask_df['hqtr_mask']) == 0).astype(np.uint8))
        uncovered = np.round(uncovered / (dim_x * dim_y), 3)

        # sanity check for venndiagram
        assert ( np.abs((covered + uncovered) - 1.0) < 1e-2 ), "Venn diagram error. Please check."

        subsets_pct = {key: np.round(value * 100, 2) for key, value in subsets.items()}
        venn = venn3(subsets_pct, set_labels=('HQCR', 'HQPR', 'HQTR'))
        plt.title(f"Venndiagram of masks with {np.round(uncovered * 100,2)}% uncovered area")
        plt.savefig(f'{figure_path}/venn_combined_masks{type_of_belief}.png', bbox_inches='tight', dpi=300)
        plt.savefig(f'{figure_path}/venn_combined_masks{type_of_belief}.pdf', bbox_inches='tight', dpi=300)
        plt.close()

        combined_beliefs = beliefs_df['hqcr_beliefs'] + beliefs_df[f'hqpr_{staining}_beliefs'] + beliefs_df['hqtr_beliefs']
        combined_beliefs /= 3

        helperfuncs.plot_pixels(
                figure_path,
                np.array(combined_beliefs).reshape(dim_x, dim_y),
                imagedim,
                f'combined_beliefs{type_of_belief}', 
                f'combined_beliefs{type_of_belief}', 
                'hot',
                False,
                False
        )
