# In[]
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import dask.dataframe as dd

from sklearn.metrics import silhouette_score

from .. import helperfuncs
from .. import missions

def create_fraction_df(x, label, rna):
        
    # adata: your AnnData object
    df = rna.obs[[label, x]].copy()

    # Count cells per x–celltype combination
    fractions_df = df.groupby([x, label]).size().reset_index(name='count')

    # Compute fractions within each leiden cluster
    fractions_df['fractions'] = fractions_df['count'] / fractions_df.groupby(x)['count'].transform('sum')

    # Rename columns to your requested names
    fractions_df = fractions_df.rename(columns={
        x: 'x',
        label: 'label'
    })

    # Sort x numerically if possible, otherwise fall back to string sort
    try:
        fractions_df['x'] = fractions_df['x'].astype(int)
        fractions_df = fractions_df.sort_values('x').reset_index(drop=True)
        fractions_df['x'] = fractions_df['x'].astype(str)
    except (ValueError, TypeError):
        fractions_df['x'] = fractions_df['x'].astype(str)
        fractions_df = fractions_df.sort_values('x').reset_index(drop=True)

    return(fractions_df)


def evaluate_res(adata, resolutions, seed):
    results = []

    for res in resolutions:
        sc.tl.leiden(
            adata,
            resolution=float(res),
            key_added="_temp_leiden",
            random_state=seed,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )

        results.append(
            (float(res), adata.obs["_temp_leiden"].nunique())
        )

    return results


def find_resolution_coarse_to_fine(
    adata,
    target_clusters,
    res,
    seed=123,
):

    # Broad search
    coarse = evaluate_res(adata, res, seed)

    best_res, _ = min(
        coarse,
        key=lambda x: abs(x[1] - target_clusters)
    )

    # Refine ±0.25 around best result
    fine_resolutions = np.linspace(
        max(0.01, best_res - 0.25),
        best_res + 0.25,
        5
    )

    fine = evaluate_res(adata, fine_resolutions, seed)

    results = coarse + fine

    adata.obs.drop(columns="_temp_leiden", inplace=True)

    best = min(
        results,
        key=lambda x: abs(x[1] - target_clusters)
    )

    print("Target number of clsuters:", target_clusters)
    print("Tested:", results)
    print("Selected:", best[0])

    return best[0]



def load_cell_metrices(
        sdata,
        spoqc_tmp_folder,
        canorm,
        *,
        include_nucleus_free=False
    ):

    counts = 'transcript_counts'
    if canorm:
        counts = 'canorm_transcript_counts'
    helperfuncs.read_sdata_parquet_tmp_files(sdata, spoqc_tmp_folder, 'hqcr')
    cell_metrices = [
        counts,
        'control_probe_counts',
        'n_genes_by_counts',
        'convexity_metric_cell',
        'convexity_min_nuceli',
        'nuceli_count',
        'border_scores',
        'thinness_score',
        'island_score',
        'doublet',
        'cell_overlap_area',
        'convexhull_outside_trnascripts',
        'num_low_qc_transcript'
    ]
    if ( include_nucleus_free ):
        cell_metrices.insert(cell_metrices.index('doublet') + 1, 'nucleus_free')
    return cell_metrices


def map_modality_metrics_to_cells(sdata, imagedim, image_type, resolution, spoqc_tmp_folder, suffix, dim_x, dim_y, stainings, figure_path):
    umap_cats = []
    polys = missions.hqcr.create_polygon_dataframe(sdata, imagedim, 'cell_boundaries')

    for modality in ['hqcr', 'hqpr', 'hqtr']:
        if ( modality == 'hqcr' ):
            metric_df = pd.read_parquet(
                f'{spoqc_tmp_folder}/hqcr_output_mask_smoothed_{suffix}.parquet',
                columns=["hqcr_beliefs_smoothed", "hqcr_mask_smoothed"], engine="pyarrow"
            )
            metric_df['hqcr_beliefs_smoothed'] = np.array(metric_df['hqcr_beliefs_smoothed']).reshape(dim_x, dim_y).flatten()
            metric_df['hqcr_mask_smoothed'] = np.array(metric_df['hqcr_mask_smoothed']).reshape(dim_x, dim_y).flatten()
            missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_df["hqcr_mask_smoothed"], "hqcr_mask_smoothed", figure_path, 'markov_labels', true_false_binary=True)
            missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_df["hqcr_mask_smoothed"], "hqcr_mask_mean_smoothed", figure_path, 'mean_values')
            missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_df["hqcr_beliefs_smoothed"], "hqcr_beliefs_smoothed", figure_path, 'mean_values')
            metric_df = pd.read_parquet(
                f'{spoqc_tmp_folder}/hqcr_output_mask_{suffix}.parquet',
                columns=["hqcr_beliefs", "hqcr_mask"], engine="pyarrow"
            )
            metric_df['hqcr_beliefs'] = np.array(metric_df['hqcr_beliefs']).reshape(dim_x, dim_y).flatten()
            metric_df['hqcr_mask'] = np.array(metric_df['hqcr_mask']).reshape(dim_x, dim_y).flatten()
            missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_df["hqcr_beliefs"], "hqcr_beliefs", figure_path, 'mean_values')
            missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_df["hqcr_mask"], "hqcr_mask", figure_path, 'markov_labels', true_false_binary=True)
            missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_df["hqcr_mask"], "hqcr_mask_mean", figure_path, 'mean_values')
            umap_cats.extend([
                'hqcr_mask_smoothed',
                'hqcr_mask_mean_smoothed',
                'hqcr_beliefs_smoothed',
                'hqcr_beliefs',
                'hqcr_mask',
                'hqcr_mask_mean',
            ])

        if ( modality == 'hqpr' ):
            for staining in stainings:
                metric_dd = dd.read_parquet(f'{spoqc_tmp_folder}/hqpr_{staining}_output_mask_smoothed_{suffix}', columns=[f"hqpr_{staining}_beliefs_smoothed", f"hqpr_{staining}_mask_smoothed"], engine="pyarrow")
                mask_smoothed = metric_dd[f"hqpr_{staining}_mask_smoothed"].compute().to_numpy()
                beliefs_smoothed = metric_dd[f"hqpr_{staining}_beliefs_smoothed"].compute().to_numpy()
                missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, mask_smoothed, f"hqpr_{staining}_mask_smoothed", figure_path, 'markov_labels', true_false_binary=True)
                missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, mask_smoothed, f"hqpr_{staining}_mask_mean_smoothed", figure_path, 'mean_values')
                missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, beliefs_smoothed, f"hqpr_{staining}_beliefs_smoothed", figure_path, 'mean_values_nonzero')
                missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, beliefs_smoothed, f"hqpr_{staining}_beliefs_mean_informative_smoothed", figure_path, 'mean_values_informative')
                metric_dd = dd.read_parquet(
                    f'{spoqc_tmp_folder}/hqpr_{staining}_output_mask_raw',
                    columns=[f"intensity", f"hqpr_{staining}_beliefs", f"hqpr_{staining}_mask"],
                    engine="pyarrow"
                )
                intensity = metric_dd[f"intensity"].compute().to_numpy()
                mask = metric_dd[f"hqpr_{staining}_mask"].compute().to_numpy()
                beliefs = metric_dd[f"hqpr_{staining}_beliefs"].compute().to_numpy()
                missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, intensity, f'hqpr_{staining}_intensity', figure_path, 'mean_values')
                missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, mask, f'hqpr_{staining}_mask', figure_path, 'markov_labels', true_false_binary=True)
                missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, mask, f'hqpr_{staining}_mask_mean', figure_path, 'mean_values')
                missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, beliefs, f'hqpr_{staining}_beliefs', figure_path, 'mean_values_nonzero')
                missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, beliefs, f'hqpr_{staining}_beliefs_mean_informative', figure_path, 'mean_values_informative')
                umap_cats.extend([
                    f'hqpr_{staining}_mask_smoothed',
                    f'hqpr_{staining}_mask_mean_smoothed',
                    f'hqpr_{staining}_beliefs_smoothed',
                    f'hqpr_{staining}_beliefs_mean_informative_smoothed',
                    f'hqpr_{staining}_intensity',
                    f'hqpr_{staining}_beliefs',
                    f'hqpr_{staining}_beliefs_mean_informative',
                    f'hqpr_{staining}_mask',
                    f'hqpr_{staining}_mask_mean',
                ])

                metrices = ['edge_strength', 'energy', 'relevance', 'entropy', 'homogenity', 'uniformity']
                for metric in metrices:
                    parquet_folder = f'{spoqc_tmp_folder}/metrices/{modality}/{staining}'
                    metric_dd = dd.read_parquet(f'{parquet_folder}/{metric}_output_{modality}_{staining}.parquet')
                    missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[metric].compute().to_numpy(), f'{metric}_{modality}_{staining}', figure_path, 'mean_values')
                    umap_cats.append(f'{metric}_{modality}_{staining}')

        if ( modality == 'hqtr' ):
            metric_dd = dd.read_parquet(f'{spoqc_tmp_folder}/hqtr_output_mask_smoothed_{suffix}', columns=["hqtr_beliefs_smoothed", "hqtr_mask_smoothed"], engine="pyarrow")
            mask_smoothed = metric_dd["hqtr_mask_smoothed"].compute().to_numpy()
            beliefs_smoothed = metric_dd["hqtr_beliefs_smoothed"].compute().to_numpy()
            missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, mask_smoothed, "hqtr_mask_smoothed", figure_path, 'markov_labels', true_false_binary=True)
            missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, mask_smoothed, "hqtr_mask_mean_smoothed", figure_path, 'mean_values')
            missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, beliefs_smoothed, "hqtr_beliefs_smoothed", figure_path, 'mean_values_nonzero')
            missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, beliefs_smoothed, "hqtr_beliefs_mean_informative_smoothed", figure_path, 'mean_values_informative')
            metric_dd = dd.read_parquet(f'{spoqc_tmp_folder}/hqtr_output_qv_prob', columns=["qv_density"], engine="pyarrow")
            missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"qv_density"].compute().to_numpy(), "hqtr_qv_density", figure_path, 'mean_values')
            metric_dd = dd.read_parquet(f'{spoqc_tmp_folder}/hqtr_output_ac_prob', columns=["ac_density"], engine="pyarrow")
            missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"ac_density"].compute().to_numpy(), "hqtr_ac_density", figure_path, 'mean_values')
            metric_dd = dd.read_parquet(
                f'{spoqc_tmp_folder}/hqtr_output_mask_raw',
                columns=["intensity", "hqtr_beliefs", "hqtr_mask"],
                engine="pyarrow"
            )
            intensity = metric_dd[f"intensity"].compute().to_numpy()
            mask = metric_dd[f"hqtr_mask"].compute().to_numpy()
            beliefs = metric_dd[f"hqtr_beliefs"].compute().to_numpy()
            missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, intensity, "hqtr_intensity", figure_path, 'mean_values')
            missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, mask, f'hqtr_mask', figure_path, 'markov_labels', true_false_binary=True)
            missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, mask, f'hqtr_mask_mean', figure_path, 'mean_values')
            missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, beliefs, f'hqtr_beliefs', figure_path, 'mean_values_nonzero')
            missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, beliefs, f'hqtr_beliefs_mean_informative', figure_path, 'mean_values_informative')
            umap_cats.extend([
                'hqtr_mask_smoothed',
                'hqtr_mask_mean_smoothed',
                'hqtr_beliefs_smoothed',
                'hqtr_beliefs_mean_informative_smoothed',
                'hqtr_intensity',
                'hqtr_qv_density',
                'hqtr_ac_density',
                'hqtr_beliefs',
                'hqtr_beliefs_mean_informative',
                'hqtr_mask',
                'hqtr_mask_mean',
            ])

            metrices = ['edge_strength', 'energy', 'relevance', 'entropy', 'homogenity', 'uniformity']
            for metric in metrices:
                parquet_folder = f'{spoqc_tmp_folder}/metrices/{modality}'
                metric_dd = dd.read_parquet(f'{parquet_folder}/{metric}_output_{modality}.parquet')
                missions.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[metric].compute().to_numpy(), f'{metric}_{modality}', figure_path, 'mean_values')
                umap_cats.append(f'{metric}_{modality}')

    return umap_cats

def write_out_anndata(sdata, rna, figure_path, subdir):

    if ( subdir == 'overview' ):
        # Remove columns that are not useful for inspection.
        if ( 'nuclei_idxs' in sdata['table'].obs.columns ):
            sdata['table'].obs.drop(columns=['nuclei_idxs'], inplace=True)

        sdata['table'].write_h5ad(
            f"{figure_path}/analysis/rna_qc_annotated.h5ad", 
            compression="gzip", 
            compression_opts=9
        )

    if ( subdir == 'cluster' ):
        if ( 'nuclei_idxs' in rna.obs.columns ):
            rna.obs.drop(columns=['nuclei_idxs'], inplace=True)
        rna.write_h5ad(
            f"{figure_path}/analysis/rna_cluster.h5ad", 
            compression="gzip", 
            compression_opts=9
        )
