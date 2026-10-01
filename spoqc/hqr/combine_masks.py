import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ThreadPoolExecutor

from matplotlib.colors import LinearSegmentedColormap
from matplotlib_venn import venn3

from .. import helperfuncs
from spoqc.core import raster

_CHUNK = 1 << 20  # pixels per task: the chunk's temporaries stay in cache


def _chunks(n):
    return [slice(i, min(i + _CHUNK, n)) for i in range(0, n, _CHUNK)]


def _venn_counts(c, p, t, threads):
    """origin/dev's 7 Venn region counts and the uncovered count, in one pass.

    The same numpy expressions origin/dev evaluated on whole pandas columns (same dtypes,
    so the same promotions and wrap-around), evaluated per chunk on a thread pool; the
    integer counts add up exactly.
    """
    def count(sl):
        c_, p_, t_ = c[sl], p[sl], t[sl]
        return np.array([
            np.sum(((c_ - p_ - t_) == 1).astype(np.uint8)),
            np.sum(((p_ - c_ - t_) == 1).astype(np.uint8)),
            np.sum(((t_ - p_ - c_) == 1).astype(np.uint8)),
            np.sum(((c_ + p_ - t_) == 2).astype(np.uint8)),
            np.sum(((c_ - p_ + t_) == 2).astype(np.uint8)),
            np.sum(((- c_ + p_ + t_) == 2).astype(np.uint8)),
            np.sum(((c_ + p_ + t_) == 3).astype(np.uint8)),
            np.sum(((c_ + p_ + t_) == 0).astype(np.uint8)),
        ], dtype=np.int64)
    with ThreadPoolExecutor(threads) as pool:
        return sum(pool.map(count, _chunks(len(c))))


def _mean_of_three(a, b, c, threads):
    """(a + b + c) / 3 elementwise as origin/dev's pandas expression computed it, per chunk on a pool."""
    out = np.empty(len(a), dtype=np.result_type(a, b, c))
    def mean(sl):
        v = a[sl] + b[sl] + c[sl]
        v /= 3
        out[sl] = v
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(mean, _chunks(len(a))))
    return out

def start_combining_masks(
        figure_path,
        spoqc_tmp_folder,
        imagedim,
        dim_x,
        dim_y,
        staining,
        *,
        celltype_refined=False,
        threads=1
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

        # Each file is read once for both of its columns, straight into numpy (origin/dev
        # computed each dask column separately and copied every column into two DataFrames).
        hqcr = raster.read_pixel_columns(file_hqcr, [hqcr_mask_name, hqcr_belief_name], dim_x * dim_y, threads)
        hqpr = raster.read_pixel_columns(file_hqpr, [hqpr_mask_name, hqpr_belief_name], dim_x * dim_y, threads)
        hqtr = raster.read_pixel_columns(file_hqtr, [hqtr_mask_name, hqtr_belief_name], dim_x * dim_y, threads)
        mask_df = {
            'hqcr_mask': hqcr[hqcr_mask_name],
            f'hqpr_{staining}_mask': hqpr[hqpr_mask_name],
            'hqtr_mask': hqtr[hqtr_mask_name]
        }

        beliefs_df = {
            'hqcr_beliefs': hqcr[hqcr_belief_name],
            f'hqpr_{staining}_beliefs': hqpr[hqpr_belief_name],
            'hqtr_beliefs': hqtr[hqtr_belief_name]
        }
        del hqcr, hqpr, hqtr

        final_mask = np.zeros(dim_x*dim_y)
        for m in ['hqcr', f'hqpr_{staining}', 'hqtr']:
            final_mask += mask_df[f'{m}_mask']

            helperfuncs.plot_pixels(
                figure_path,
                mask_df[f'{m}_mask'].reshape(dim_x, dim_y),
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
                beliefs_df[f'{m}_beliefs'].reshape(dim_x, dim_y),
                imagedim,
                f'{m}_beliefs{type_of_belief}', 
                f'{m}_beliefs{type_of_belief}', 
                'hot',
                False,
                False
            )

            # --- general histograms ---
            # These will help later to figure out thresholds for filtering.
            # These are on spatial observation (no cell agglomeration).
            fig = None
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(beliefs_df[f'{m}_beliefs'], bins=50)
            ax.set_yscale('log')
            ax.set_xlabel(f'{m}_beliefs')
            ax.set_ylabel('Log count')
            ax.set_title(f'Distribution of {m} beliefs')
            fig.savefig(os.path.join(figure_path, f'hist_{m}_beliefs{type_of_belief}_log.png'), bbox_inches='tight')
            fig.savefig(os.path.join(figure_path, f'hist_{m}_beliefs{type_of_belief}_log.pdf'), bbox_inches='tight')
            plt.close(fig)

            fig = None
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(beliefs_df[f'{m}_beliefs'], bins=50)
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
        counts = _venn_counts(mask_df['hqcr_mask'], mask_df[f'hqpr_{staining}_mask'], mask_df['hqtr_mask'], threads)
        subsets = dict(zip(['100', '010', '001', '110', '101', '011', '111'], counts[:7]))

        covered = 0
        for key in subsets:
            subsets[key] = np.round(subsets[key] / (dim_x * dim_y), 3)
            covered += subsets[key]

        uncovered = counts[7]
        uncovered = np.round(uncovered / (dim_x * dim_y), 3)

        # sanity check for venndiagram
        assert ( np.abs((covered + uncovered) - 1.0) < 1e-2 ), "Venn diagram error. Please check."

        subsets_pct = {key: np.round(value * 100, 2) for key, value in subsets.items()}
        venn = venn3(subsets_pct, set_labels=('HQCR', 'HQPR', 'HQTR'))
        plt.title(f"Venndiagram of masks with {np.round(uncovered * 100,2)}% uncovered area")
        plt.savefig(f'{figure_path}/venn_combined_masks{type_of_belief}.png', bbox_inches='tight', dpi=300)
        plt.savefig(f'{figure_path}/venn_combined_masks{type_of_belief}.pdf', bbox_inches='tight', dpi=300)
        plt.close()

        combined_beliefs = _mean_of_three(beliefs_df['hqcr_beliefs'], beliefs_df[f'hqpr_{staining}_beliefs'], beliefs_df['hqtr_beliefs'], threads)

        helperfuncs.plot_pixels(
                figure_path,
                combined_beliefs.reshape(dim_x, dim_y),
                imagedim,
                f'combined_beliefs{type_of_belief}', 
                f'combined_beliefs{type_of_belief}', 
                'hot',
                False,
                False
        )
