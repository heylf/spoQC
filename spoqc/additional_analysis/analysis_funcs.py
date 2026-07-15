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
from .. import subworkflows

def create_celltype_fraction_df(x, label, rna):
        
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

    # Sort x numerically
    fractions_df['x'] = fractions_df['x'].astype(int)
    fractions_df = fractions_df.sort_values('x').reset_index(drop=True)
    fractions_df['x'] = fractions_df['x'].astype(str)

    return(fractions_df)


def leiden_silhouette(adata, resolution, res_index, ninits=5):
    '''
    Returns average silhouette score and average number of clusters based on a given resolution and random state 
    with ninits=random initializations.
    '''

    scores = []
    num_clusters = []
    adata_test = adata.copy()

    for rs in range(0, ninits): #since the random seed will change result - multiple samples per resolution
        sc.tl.leiden(adata_test, resolution=resolution, key_added='temp_leiden', random_state=rs)

        if ( len(set(adata_test.obs['temp_leiden'])) > 1 ):
            scores.append(silhouette_score(adata_test.obsm['X_umap'], adata_test.obs['temp_leiden']))
            num_clusters.append(len(set(adata_test.obs['temp_leiden'])))
            adata_test.obs.drop(columns=['temp_leiden'], inplace=True)
        else:
            scores.append(0.0)
            num_clusters.append(0.0)
            adata_test.obs.drop(columns=['temp_leiden'], inplace=True)

    return [res_index, [resolution]*ninits, scores, list(range(0, ninits)), num_clusters]


# This is for checking which leiden cluster resoltuion would work the best.
# Pick the one with the highest silhouette score but not the one from the beginning.
def test_resolutions_leiden(
        rna, 
        figure_path,
        threads,
        annotation_key=None,
        k=None,
        steps=None,
        end=2.0,
        start=0.0,
        resolutions=None,
    ):
    
    if ( resolutions == None ):
        resolutions = np.linspace(0, 3, num = 21)[1:]
        if ( steps ):
            resolutions = np.linspace(start, end, num = steps)[1:]

    out = [-1] * len(resolutions)
    win_res = 0.5
    diff_clusters = 100

    # Parallel testing for resolutions.
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(leiden_silhouette, rna, res, res_index) for res_index, res in enumerate(resolutions)]
        for future in concurrent.futures.as_completed(futures):
            results = future.result()
            out[results[0]] = results[1:]

    for res in range(0, len(resolutions)):
        average_num_clusters = int(np.mean(out[res][-1]))

        if ( annotation_key ):
            diff_clusters_new = np.abs(average_num_clusters-( len(set(rna.obs[annotation_key])) + 3 ) )
            if ( diff_clusters_new < diff_clusters ):
                diff_clusters = diff_clusters_new
                win_res = resolutions[res]
        if ( k ):
            diff_clusters_new = np.abs(average_num_clusters-k)
            if ( diff_clusters_new < diff_clusters ):
                diff_clusters = diff_clusters_new
                win_res = resolutions[res]

    # Unpack data into dataframe.    
    rows = []
    for block in out:
        rows.extend(zip(*block))
    ss = pd.DataFrame(rows, columns=['res','ss', 'rs', 'num_clusters'])

    plt.figure(figsize=(15, 5))
    ax = sns.lineplot(data = ss, x = 'res', y = 'ss')
    plt.xlabel('resolutions')
    plt.ylabel('silhouettescore')
    for res_value in ss['res'].unique():  # Assuming 'res' contains the breakpoints
        plt.axvline(x=res_value, color='grey', linestyle='--', alpha=0.7)  # Adding vertical lines
    plt.xticks(ss['res'].unique())  # Ensure all 'res' values are shown on the x-axis
    plt.savefig(f'{figure_path}/test_resolutions_leiden_clustering_ss.png')
    plt.savefig(f'{figure_path}/test_resolutions_leiden_clustering_ss.pdf')
    plt.close()

    if ( annotation_key or k):
        title = ''
        if ( annotation_key ):
            title = len(set(rna.obs[annotation_key]))
        else:
            title = str(k)

        plt.figure(figsize=(15, 5))
        ax = sns.lineplot(data=ss, x='res', y='num_clusters')
        plt.xlabel('resolutions')
        plt.ylabel('number of clusters')
        plt.ylim([0, 40])
        for res_value in ss['res'].unique():
            plt.axvline(x=res_value, color='grey', linestyle='--', alpha=0.7)
        for y in [5, 10, 15, 20, 25, 30, 35, 40]:
            plt.axhline(y=y, color='grey', linestyle='--', alpha=0.7)
        plt.axvline(x=win_res, color='black', linestyle='-', alpha=0.7)
        plt.title(f'Annotation had {title} celltypes')
        plt.xticks(ss['res'].unique())  # ensure all 'res' values appear
        plt.tight_layout()
        plt.savefig(f'{figure_path}/test_resolutions_leiden_clustering_num_clusters.png')
        plt.savefig(f'{figure_path}/test_resolutions_leiden_clustering_num_clusters.pdf')
        plt.close()

    return win_res


def load_cell_metrices(sdata, spoqc_tmp_folder, CONST, *, include_nucleus_free=False):
    counts = 'transcript_counts'
    if ( CONST.CANORM ):
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
    polys = subworkflows.hqcr.create_polygon_dataframe(sdata, imagedim, 'cell_boundaries')

    for modality in ['hqcr', 'hqpr', 'hqtr']:
        if ( modality == 'hqcr' ):
            metric_df = pd.read_parquet(
                f'{spoqc_tmp_folder}/hqcr_output_mask_smoothed_{suffix}.parquet',
                columns=["hqcr_beliefs_smoothed", "hqcr_mask_smoothed"], engine="pyarrow"
            )
            metric_df['hqcr_beliefs_smoothed'] = np.array(metric_df['hqcr_beliefs_smoothed']).reshape(dim_x, dim_y).flatten()
            metric_df['hqcr_mask_smoothed'] = np.array(metric_df['hqcr_mask_smoothed']).reshape(dim_x, dim_y).flatten()
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_df["hqcr_mask_smoothed"], "hqcr_mask_smoothed", figure_path, 'markov_labels', true_false_binary=True)
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_df["hqcr_mask_smoothed"], "hqcr_mask_mean_smoothed", figure_path, 'mean_values')
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_df["hqcr_beliefs_smoothed"], "hqcr_beliefs_smoothed", figure_path, 'mean_values')
            metric_df = pd.read_parquet(
                f'{spoqc_tmp_folder}/hqcr_output_mask_{suffix}.parquet',
                columns=["hqcr_beliefs", "hqcr_mask"], engine="pyarrow"
            )
            metric_df['hqcr_beliefs'] = np.array(metric_df['hqcr_beliefs']).reshape(dim_x, dim_y).flatten()
            metric_df['hqcr_mask'] = np.array(metric_df['hqcr_mask']).reshape(dim_x, dim_y).flatten()
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_df["hqcr_beliefs"], "hqcr_beliefs", figure_path, 'mean_values')
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_df["hqcr_mask"], "hqcr_mask", figure_path, 'markov_labels', true_false_binary=True)
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_df["hqcr_mask"], "hqcr_mask_mean", figure_path, 'mean_values')
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
                subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"hqpr_{staining}_mask_smoothed"].compute().to_numpy(), f"hqpr_{staining}_mask_smoothed", figure_path, 'markov_labels', true_false_binary=True)
                subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"hqpr_{staining}_mask_smoothed"].compute().to_numpy(), f"hqpr_{staining}_mask_mean_smoothed", figure_path, 'mean_values')
                subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"hqpr_{staining}_beliefs_smoothed"].compute().to_numpy(), f"hqpr_{staining}_beliefs_smoothed", figure_path, 'mean_values_nonzero')
                subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"hqpr_{staining}_beliefs_smoothed"].compute().to_numpy(), f"hqpr_{staining}_beliefs_mean_informative_smoothed", figure_path, 'mean_values_informative')
                metric_dd = dd.read_parquet(
                    f'{spoqc_tmp_folder}/hqpr_{staining}_output_mask_raw',
                    columns=[f"intensity", f"hqpr_{staining}_beliefs", f"hqpr_{staining}_mask"],
                    engine="pyarrow"
                )
                subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"intensity"].compute().to_numpy(), f'hqpr_{staining}_intensity', figure_path, 'mean_values')
                subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"hqpr_{staining}_mask"].compute().to_numpy(), f'hqpr_{staining}_mask', figure_path, 'markov_labels', true_false_binary=True)
                subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"hqpr_{staining}_mask"].compute().to_numpy(), f'hqpr_{staining}_mask_mean', figure_path, 'mean_values')
                subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"hqpr_{staining}_beliefs"].compute().to_numpy(), f'hqpr_{staining}_beliefs', figure_path, 'mean_values_nonzero')
                subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"hqpr_{staining}_beliefs"].compute().to_numpy(), f'hqpr_{staining}_beliefs_mean_informative', figure_path, 'mean_values_informative')
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
                    subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[metric].compute().to_numpy(), f'{metric}_{modality}_{staining}', figure_path, 'mean_values')
                    umap_cats.append(f'{metric}_{modality}_{staining}')

        if ( modality == 'hqtr' ):
            metric_dd = dd.read_parquet(f'{spoqc_tmp_folder}/hqtr_output_mask_smoothed_{suffix}', columns=["hqtr_beliefs_smoothed", "hqtr_mask_smoothed"], engine="pyarrow")
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd["hqtr_mask_smoothed"].compute().to_numpy(), "hqtr_mask_smoothed", figure_path, 'markov_labels', true_false_binary=True)
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd["hqtr_mask_smoothed"].compute().to_numpy(), "hqtr_mask_mean_smoothed", figure_path, 'mean_values')
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd["hqtr_beliefs_smoothed"].compute().to_numpy(), "hqtr_beliefs_smoothed", figure_path, 'mean_values_nonzero')
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd["hqtr_beliefs_smoothed"].compute().to_numpy(), "hqtr_beliefs_mean_informative_smoothed", figure_path, 'mean_values_informative')
            metric_dd = dd.read_parquet(f'{spoqc_tmp_folder}/hqtr_output_qv_prob', columns=["qv_density"], engine="pyarrow")
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"qv_density"].compute().to_numpy(), "hqtr_qv_density", figure_path, 'mean_values')
            metric_dd = dd.read_parquet(f'{spoqc_tmp_folder}/hqtr_output_ac_prob', columns=["ac_density"], engine="pyarrow")
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"ac_density"].compute().to_numpy(), "hqtr_ac_density", figure_path, 'mean_values')
            metric_dd = dd.read_parquet(
                f'{spoqc_tmp_folder}/hqtr_output_mask_raw',
                columns=["intensity", "hqtr_beliefs", "hqtr_mask"],
                engine="pyarrow"
            )
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"intensity"].compute().to_numpy(), "hqtr_intensity", figure_path, 'mean_values')
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"hqtr_mask"].compute().to_numpy(), f'hqtr_mask', figure_path, 'markov_labels', true_false_binary=True)
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"hqtr_mask"].compute().to_numpy(), f'hqtr_mask_mean', figure_path, 'mean_values')
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"hqtr_beliefs"].compute().to_numpy(), f'hqtr_beliefs', figure_path, 'mean_values_nonzero')
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"hqtr_beliefs"].compute().to_numpy(), f'hqtr_beliefs_mean_informative', figure_path, 'mean_values_informative')
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
                subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[metric].compute().to_numpy(), f'{metric}_{modality}', figure_path, 'mean_values')
                umap_cats.append(f'{metric}_{modality}')

    return umap_cats