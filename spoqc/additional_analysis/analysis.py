# In[]
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px   # plotly
import nbformat               # plotly related
import plotly.graph_objects as go
import os
import concurrent.futures
import dask.dataframe as dd
import sys
import ast
import json

from sklearn.metrics import silhouette_score
from pandas.api.types import is_numeric_dtype

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
def test_resolutions_leiden(rna, figure_path, threads, annotation_key=None, k=None, steps=None, end=1.0, start=0.0):
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


def cell_category_analysis(
        sdata,
        subdir,
        CONST,
        seed,
        suffix,
        dim_x,
        dim_y,
        imagedim,
        stainings,
    ):

    spoqc_tmp_folder = CONST.TMP_PATH
    image_type = CONST.IMAGE_TYPE
    resolution = CONST.RESOLUTION

    umap_cats = []

    # Load cell metrices
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
    umap_cats.extend(cell_metrices)

    figure_path = f'{CONST.FIGURE_PATH}/analysis/{subdir}'
    rna = sdata.table
    rna.X = rna.layers['normlog']
    sc.pp.neighbors(rna, n_neighbors=20, random_state=seed)
    sc.tl.umap(rna, min_dist=0.1, spread=1.2, random_state=seed)

    ####################################################################################################################    
    # Do the actual analysis
    ####################################################################################################################
    object = 'cell'
    polys = subworkflows.hqcr.create_polygon_dataframe(sdata, imagedim, f'{object}_boundaries')

    for modality in ['hqcr', 'hqpr', 'hqtr']:
        if ( modality == 'hqcr' ):
            metric_df = pd.read_parquet(f'{spoqc_tmp_folder}/hqcr_output_mask_{suffix}.parquet', columns=["hqcr_beliefs", "hqcr_mask"], engine="pyarrow")
            metric_df['hqcr_beliefs'] = np.array(metric_df['hqcr_beliefs']).reshape(dim_x, dim_y).flatten()
            metric_df['hqcr_mask'] = np.array(metric_df['hqcr_mask']).reshape(dim_x, dim_y).flatten()
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_df["hqcr_mask"], "hqcr_mask", figure_path, 'markov_labels', true_false_binary=True)
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_df["hqcr_beliefs"], "hqcr_beliefs", figure_path, 'mean_values')
            umap_cats.extend(['hqcr_mask', 'hqcr_beliefs'])

        if ( modality == 'hqpr' ):
            for staining in stainings:
                # mask and beliefs
                metric_dd = dd.read_parquet(f'{spoqc_tmp_folder}/hqpr_{staining}_output_mask_{suffix}', columns=[f"hqpr_{staining}_beliefs", f"hqpr_{staining}_mask"], engine="pyarrow")
                subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"hqpr_{staining}_mask"].compute().to_numpy(), f"hqpr_{staining}_mask", figure_path, 'markov_labels', true_false_binary=True)
                subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"hqpr_{staining}_beliefs"].compute().to_numpy(), f"hqpr_{staining}_beliefs", figure_path, 'mean_values')
                metric_dd = dd.read_parquet(f'{spoqc_tmp_folder}/hqpr_{staining}_output_mask_prob', columns=[f"intensity"], engine="pyarrow")
                subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"intensity"].compute().to_numpy(), f'hqpr_{staining}_intensity', figure_path, 'mean_values')
                umap_cats.extend([f'hqpr_{staining}_mask', f'hqpr_{staining}_beliefs', f'hqpr_{staining}_intensity'])

                metrices = ['edge_strength', 'energy', 'relevance', 'entropy', 'homogenity', 'uniformity']
                for metric in metrices:
                    parquet_folder=f'{spoqc_tmp_folder}/metrices/{modality}/{staining}'
                    metric_dd = dd.read_parquet(f'{parquet_folder}/{metric}_output_{modality}_{staining}.parquet')
                    subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[metric].compute().to_numpy(), f'{metric}_{modality}_{staining}', figure_path, 'mean_values')
                    umap_cats.append(f'{metric}_{modality}_{staining}')


        if ( modality == 'hqtr' ):
            metric_dd = dd.read_parquet(f'{spoqc_tmp_folder}/hqtr_output_mask_{suffix}', columns=["hqtr_beliefs", "hqtr_mask"], engine="pyarrow")
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd["hqtr_mask"].compute().to_numpy(), "hqtr_mask", figure_path, 'markov_labels', true_false_binary=True)
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd["hqtr_beliefs"].compute().to_numpy(), "hqtr_beliefs", figure_path, 'mean_values')
            metric_dd = dd.read_parquet(f'{spoqc_tmp_folder}/hqtr_output_qv_prob', columns=["qv_density"], engine="pyarrow")
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"qv_density"].compute().to_numpy(), "hqtr_qv_density", figure_path, 'mean_values')
            metric_dd = dd.read_parquet(f'{spoqc_tmp_folder}/hqtr_output_ac_prob', columns=["ac_density"], engine="pyarrow")
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"ac_density"].compute().to_numpy(), "hqtr_ac_density", figure_path, 'mean_values')
            metric_dd = dd.read_parquet(f'{spoqc_tmp_folder}/hqtr_output_mask_prob', columns=["intensity"], engine="pyarrow")
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"intensity"].compute().to_numpy(), "hqtr_intensity", figure_path, 'mean_values')
            umap_cats.extend(['hqtr_mask', 'hqtr_beliefs', 'hqtr_intensity', 'hqtr_qv_density', 'hqtr_ac_density'])

            metrices = ['edge_strength', 'energy', 'relevance', 'entropy', 'homogenity', 'uniformity']
            for metric in metrices:
                parquet_folder=f'{spoqc_tmp_folder}/metrices/{modality}'
                metric_dd = dd.read_parquet(f'{parquet_folder}/{metric}_output_{modality}.parquet')
                subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[metric].compute().to_numpy(), f'{metric}_{modality}', figure_path, 'mean_values')
                umap_cats.append(f'{metric}_{modality}')
    
    bool_colors = {False: "#4C78A8", True: "#E45756"}
    log1p_umap_cat = ['convexhull_outside_trnascripts']

    figures = []
    for cat in ['border_cell', 'doublet', 'nucleus_free']:
        if ( not os.path.exists(f"{figure_path}/{cat}") ):
            os.makedirs(f"{figure_path}/{cat}")

        df = sdata.table.obs.copy()

        # Check if I have enough cells for the celltype for the cat=True/False, else remove cells because
        # I cannot make a good estimation for my distribution.
        n_points = 20

        filtered_indices = []
        for c in list(set(df[CONST.ANNOTATION_KEY])):
            for b in [False, True]:
                rows_celltype = df[CONST.ANNOTATION_KEY] == c
                rows_cat = df[cat] == b
                subset = df[rows_celltype & rows_cat]
                if ( len(subset) > n_points ):
                    filtered_indices.extend(list(subset.index))
        filtered_indices.sort()
        df = df.iloc[filtered_indices]

        for umap_cat in umap_cats:
            # same condition as before
            if len(set(rna.obs[umap_cat])) > 2:

                if ( umap_cat in log1p_umap_cat ):
                    df[f'log101p_{umap_cat}'] = np.log10( df[umap_cat] + 1 )
                    umap_cat = f'log101p_{umap_cat}'

                # ---- split violin plot -------
                fig = go.Figure()
                df_false = df[df[cat] == False]
                fig.add_trace(
                    go.Violin(
                        x=df_false[umap_cat],                      # numeric axis
                        y=df_false[CONST.ANNOTATION_KEY],          # categorical axis
                        name=str(False),
                        legendgroup=str(False),
                        scalegroup=str(False),
                        side="negative",                           # left side
                        orientation="h",                           # horizontal violins
                        line_color=bool_colors[False],
                        fillcolor=bool_colors[False],
                        opacity=0.7,
                        points=False,                              # or "outliers"/"all" if you like
                        meanline_visible=False,
                    )
                )

                df_true = df[df[cat] == True]
                fig.add_trace(
                    go.Violin(
                        x=df_true[umap_cat],
                        y=df_true[CONST.ANNOTATION_KEY],
                        name=str(True),
                        legendgroup=str(True),
                        scalegroup=str(True),
                        side="positive",                           # right side
                        orientation="h",
                        line_color=bool_colors[True],
                        fillcolor=bool_colors[True],
                        opacity=0.7,
                        points=False,
                        meanline_visible=False,
                    )
                )

                fig.update_layout(
                    title=f"{umap_cat} of {cat}",
                    width=800,
                    height=2500,
                    violinmode="overlay",  # needed for split
                    xaxis_title=umap_cat,
                    yaxis_title=CONST.ANNOTATION_KEY,
                    legend_title=cat
                )

                # optional: keep stable order of boolean in legend
                fig.update_layout(
                    legend=dict(
                        traceorder="normal"
                    )
                )

                figures.append(fig)
                fig.write_image(
                    f"{figure_path}/{cat}/split_violinplot_{umap_cat}.png",
                    scale=3,
                )
                fig.write_image(
                    f"{figure_path}/{cat}/split_violinplot_{umap_cat}.pdf",
                    scale=3,
                )

                # ---- split boxplot plot -------
                fig = go.Figure()
                fig.add_trace(
                    go.Box(
                        x=df_false[umap_cat],
                        y=df_false[CONST.ANNOTATION_KEY],
                        name=str(False),
                        legendgroup=str(False),
                        orientation="h",
                        marker_color=bool_colors[False],
                        boxpoints=False,      # no points
                        # You can use these two to keep False/True paired
                        offsetgroup="False",
                        # width=0.5  # tweak if you want more separation
                    )
                )

                fig.add_trace(
                    go.Box(
                        x=df_true[umap_cat],
                        y=df_true[CONST.ANNOTATION_KEY],
                        name=str(True),
                        legendgroup=str(True),
                        orientation="h",
                        marker_color=bool_colors[True],
                        boxpoints=False,
                        offsetgroup="True",
                    )
                )

                fig.update_layout(
                    title=f"{umap_cat} of {cat}",
                    width=800,
                    height=2500,
                    boxmode="group",  # group False/True per category
                    xaxis_title=umap_cat,
                    yaxis_title=CONST.ANNOTATION_KEY,
                    legend_title=cat,
                    legend=dict(traceorder="normal")
                )

                figures.append(fig)
                fig.write_image(
                    f"{figure_path}/{cat}/split_boxplot_{umap_cat}.png",
                    scale=3,
                )
                fig.write_image(
                    f"{figure_path}/{cat}/split_boxplot_{umap_cat}.pdf",
                    scale=3,
                )


                # --- boxplots ---
                boxplot_trace = go.Box(
                    x=df[cat],
                    y=df[umap_cat]
                )

                # Create the layout
                layout = go.Layout(
                    xaxis=dict(
                        tickangle=-45,
                        tickfont=dict(size=18),
                        title="leiden cluster"
                    ),
                    yaxis=dict(
                        title=umap_cat,
                        tickfont=dict(size=18)
                    ),
                    height=500,
                    width=800
                )

                # Create the figure
                fig = go.Figure(data=[boxplot_trace], layout=layout)
                helperfuncs.apply_general_plotly_layout(fig, False)
                fig.write_html(f"{figure_path}/{cat}/boxplot_{umap_cat}.html")
                fig.write_image(f"{figure_path}/{cat}/boxplot_{umap_cat}.png", scale=3)
                fig.write_image(f"{figure_path}/{cat}/boxplot_{umap_cat}.pdf", scale=3)


                # --- violin plots ---
                violin_trace = go.Violin(
                    x=df[cat],
                    y=df[umap_cat],
                    box_visible=True,        # embedded boxplot
                    box=dict(fillcolor="white", line=dict(color="black")),
                    meanline_visible=True,   # show mean line
                    points=False,             # change to 'all' to show individual points
                    spanmode='hard'
                )

                # Create the layout
                layout = go.Layout(
                    xaxis=dict(
                        tickangle=-45,
                        tickfont=dict(size=18),
                        title="leiden cluster"
                    ),
                    yaxis=dict(
                        title=umap_cat,
                        tickfont=dict(size=18)
                    ),
                    height=500,
                    width=800
                )

                # Create the figure
                fig = go.Figure(data=[violin_trace], layout=layout)

                helperfuncs.apply_general_plotly_layout(fig, False)

                fig.write_html(f"{figure_path}/{cat}/violin_{umap_cat}.html")
                fig.write_image(f"{figure_path}/{cat}/violin_{umap_cat}.png", scale=3)
                fig.write_image(f"{figure_path}/{cat}/violin_{umap_cat}.pdf", scale=3)


        # Generate plotly HTML
        html_content = ''.join(fig.to_html(full_html=False) for fig in figures)
        with open(f"{figure_path}/{cat}/qc_analysis.html", "w") as f:
            f.write(html_content)

    done_file = open(f"{figure_path}/done.txt", "w")
    done_file.write("its done")
    done_file.close()


def celltype_cluster_analysis(
        sdata,
        subdir,
        CONST,
        seed,
        suffix,
        dim_x,
        dim_y,
        imagedim,
        stainings,
        annotation,
    ):

    spoqc_tmp_folder = CONST.TMP_PATH
    image_type = CONST.IMAGE_TYPE
    resolution = CONST.RESOLUTION

    umap_cats = []

    # Load cell metrices
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
        'nucleus_free',
        'cell_overlap_area',
        'convexhull_outside_trnascripts',
        'num_low_qc_transcript'
    ]
    umap_cats.extend(cell_metrices)

    figure_path = f'{CONST.FIGURE_PATH}/analysis/{subdir}'
    rna = sdata.table
    rna.X = rna.layers['normlog']

    ####################################################################################################################
    # Subsetting data for different analyses
    ####################################################################################################################
    if ( subdir == 'cluster' ):
        largest_cluster = ''
        if ( CONST.CLUSTER_CELLTYPE ):
            largest_cluster = CONST.CLUSTER_CELLTYPE
        else:
            num_largest_cluster = 0
            for g in rna.obs.groupby(CONST.ANNOTATION_KEY):
                if ( len(g[1]) > num_largest_cluster ):
                    num_largest_cluster = len(g[1])
                    largest_cluster = g[0]
        print(f"[NOTE] Picking cluster {largest_cluster}")
        rna = rna[rna.obs[CONST.ANNOTATION_KEY] == largest_cluster]

    print(rna.obs[CONST.ANNOTATION_KEY])

    ####################################################################################################################
    # Madatory steps
    ####################################################################################################################

    sc.pp.neighbors(rna, n_neighbors=20, random_state=seed)
    sc.tl.umap(rna, min_dist=0.1, spread=1.2, random_state=seed)

    ####################################################################################################################    
    # Testing for resolution
    ####################################################################################################################
    res_file_name = f"{figure_path}/res.txt"
    win_res = -1
    if ( not os.path.exists(res_file_name) ):
        if ( subdir == 'overview' ):
            win_res = test_resolutions_leiden(rna, figure_path, 
                                              CONST.THREADS, annotation_key=CONST.ANNOTATION_KEY, steps=10)
        else:
            win_res = test_resolutions_leiden(rna, figure_path, CONST.THREADS, k=15, steps=10)

        res_file = open(f"{figure_path}/res.txt", "w")
        res_file.write(f"{win_res}\nIs the winning leiden resolution\n")
        res_file.close()
    else:
        res_file = open(f"{figure_path}/res.txt", "r")
        win_res = float(res_file.readline().strip("\n"))

    # Pick the correct solution after you have inspected the testing plots.
    sc.tl.leiden(rna, resolution=win_res, key_added='leiden', random_state=seed)

    if ( ( len(set(rna.obs['leiden'])) > 30 ) and ( not os.path.exists(res_file_name) ) ):
        rna.obs.drop(columns=['leiden'], inplace=True)
        win_res = test_resolutions_leiden(
            rna,
            figure_path,
            CONST.THREADS,
            steps=30,
            end=0.1,
            start=0.000001
        )
        sc.tl.leiden(rna, resolution=win_res, key_added='leiden', random_state=seed)

    ####################################################################################################################    
    # Do the actual analysis
    ####################################################################################################################

    object = 'cell'
    polys = subworkflows.hqcr.create_polygon_dataframe(sdata, imagedim, f'{object}_boundaries')

    for modality in ['hqcr', 'hqpr', 'hqtr']:
        if ( modality == 'hqcr' ):
            metric_df = pd.read_parquet(f'{spoqc_tmp_folder}/hqcr_output_mask_{suffix}.parquet', columns=["hqcr_beliefs", "hqcr_mask"], engine="pyarrow")
            metric_df['hqcr_beliefs'] = np.array(metric_df['hqcr_beliefs']).reshape(dim_x, dim_y).flatten()
            metric_df['hqcr_mask'] = np.array(metric_df['hqcr_mask']).reshape(dim_x, dim_y).flatten()
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_df["hqcr_mask"], "hqcr_mask", figure_path, 'markov_labels', true_false_binary=True)
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_df["hqcr_beliefs"], "hqcr_beliefs", figure_path, 'mean_values')
            umap_cats.extend(['hqcr_mask', 'hqcr_beliefs'])

        if ( modality == 'hqpr' ):
            print(stainings)
            for staining in stainings:
                print(staining)
                
                # mask and beliefs
                metric_dd = dd.read_parquet(f'{spoqc_tmp_folder}/hqpr_{staining}_output_mask_{suffix}', columns=[f"hqpr_{staining}_beliefs", f"hqpr_{staining}_mask"], engine="pyarrow")
                subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"hqpr_{staining}_mask"].compute().to_numpy(), f"hqpr_{staining}_mask", figure_path, 'markov_labels', true_false_binary=True)
                subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"hqpr_{staining}_beliefs"].compute().to_numpy(), f"hqpr_{staining}_beliefs", figure_path, 'mean_values')
                metric_dd = dd.read_parquet(f'{spoqc_tmp_folder}/hqpr_{staining}_output_mask_prob', columns=[f"intensity"], engine="pyarrow")
                subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"intensity"].compute().to_numpy(), f'hqpr_{staining}_intensity', figure_path, 'mean_values')
                umap_cats.extend([f'hqpr_{staining}_mask', f'hqpr_{staining}_beliefs', f'hqpr_{staining}_intensity'])

                metrices = ['edge_strength', 'energy', 'relevance', 'entropy', 'homogenity', 'uniformity']
                for metric in metrices:
                    parquet_folder=f'{spoqc_tmp_folder}/metrices/{modality}/{staining}'
                    metric_dd = dd.read_parquet(f'{parquet_folder}/{metric}_output_{modality}_{staining}.parquet')
                    subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[metric].compute().to_numpy(), f'{metric}_{modality}_{staining}', figure_path, 'mean_values')
                    umap_cats.append(f'{metric}_{modality}_{staining}')


        if ( modality == 'hqtr' ):
            metric_dd = dd.read_parquet(f'{spoqc_tmp_folder}/hqtr_output_mask_{suffix}', columns=["hqtr_beliefs", "hqtr_mask"], engine="pyarrow")
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd["hqtr_mask"].compute().to_numpy(), "hqtr_mask", figure_path, 'markov_labels', true_false_binary=True)
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd["hqtr_beliefs"].compute().to_numpy(), "hqtr_beliefs", figure_path, 'mean_values')
            metric_dd = dd.read_parquet(f'{spoqc_tmp_folder}/hqtr_output_qv_prob', columns=["qv_density"], engine="pyarrow")
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"qv_density"].compute().to_numpy(), "hqtr_qv_density", figure_path, 'mean_values')
            metric_dd = dd.read_parquet(f'{spoqc_tmp_folder}/hqtr_output_ac_prob', columns=["ac_density"], engine="pyarrow")
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"ac_density"].compute().to_numpy(), "hqtr_ac_density", figure_path, 'mean_values')
            metric_dd = dd.read_parquet(f'{spoqc_tmp_folder}/hqtr_output_mask_prob', columns=["intensity"], engine="pyarrow")
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[f"intensity"].compute().to_numpy(), "hqtr_intensity", figure_path, 'mean_values')
            umap_cats.extend(['hqtr_mask', 'hqtr_beliefs', 'hqtr_intensity', 'hqtr_qv_density', 'hqtr_ac_density'])

            metrices = ['edge_strength', 'energy', 'relevance', 'entropy', 'homogenity', 'uniformity']
            for metric in metrices:
                parquet_folder=f'{spoqc_tmp_folder}/metrices/{modality}'
                metric_dd = dd.read_parquet(f'{parquet_folder}/{metric}_output_{modality}.parquet')
                subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, metric_dd[metric].compute().to_numpy(), f'{metric}_{modality}', figure_path, 'mean_values')
                umap_cats.append(f'{metric}_{modality}')

    # This had to be done if I analyse a specific clsuter because I subset the data.
    if ( subdir == 'cluster' ):
        add_cols = [x for x in list(sdata.table.obs.columns) if x not in list(rna.obs.columns)]
        for col in add_cols:
            rna.obs[col] = np.array(sdata.table.obs.loc[rna.obs.index, col])

    # Add general stuff to check
    umap_cats.extend([CONST.ANNOTATION_KEY, 'leiden'])

    X = rna.obsm["X_umap"]

    finite = np.isfinite(X).all(axis=1)
    xmin, xmax = np.nanmin(X[finite, 0]), np.nanmax(X[finite, 0])
    ymin, ymax = np.nanmin(X[finite, 1]), np.nanmax(X[finite, 1])

    pad_x = 0.03 * (xmax - xmin if xmax > xmin else 1.0)
    pad_y = 0.03 * (ymax - ymin if ymax > ymin else 1.0)

    xrange = [xmin - pad_x, xmax + pad_x]
    yrange = [ymin - pad_y, ymax + pad_y]

    cluster_centroid_coords = []
    clusters = list(set(rna.obs['leiden']))
    clusters.sort()
    for cluster in clusters:
        centroid_coords = [np.mean(X[rna.obs['leiden'] == cluster,0]), np.mean(X[rna.obs['leiden'] == cluster,1])]
        cluster_centroid_coords.append(centroid_coords) 
    cluster_centroid_coords = np.array(cluster_centroid_coords)

    leiden_clusters = list(set(rna.obs['leiden']))
    leiden_colors = helperfuncs.generate_distinct_colors(len(leiden_clusters))

    # Generate spatial plots for leiden clsuters
    rna.obs['spatial_leiden_cluster'] = [True] * rna.n_obs
    for c in leiden_clusters:
        rna.obs['spatial_leiden_cluster'] = rna.obs['leiden'] == c
        helperfuncs.plot_scatter(
            rna,
            figure_path,
            f'leiden_cluster_{c}',
            None,
            'spatial_leiden_cluster',
            ['lightblue', 'black'], 
            f'leiden_cluster_{c}'
        )

    # Generate spatial plots for annotation clusters
    rna.obs['spatial_leiden_cluster'] = [True] * rna.n_obs
    for c in list(set(rna.obs[CONST.ANNOTATION_KEY])):
        rna.obs['spatial_leiden_cluster'] = rna.obs[CONST.ANNOTATION_KEY] == c
        helperfuncs.plot_scatter(
            rna,
            figure_path,
            f'annotation_{c}',
            None,
            'spatial_leiden_cluster',
            ['lightblue', 'black'], 
            f'annotation_{c}'
        )

    celltype_colors = []
    for umap_cat in umap_cats:
        fig = None
        col_color = 'leiden'

        # --- prepare labels/colors (your original logic) ---
        if not is_numeric_dtype(rna.obs[umap_cat]) and umap_cat != 'leiden':
            labels = None
            col_color = umap_cat
            if (umap_cat == CONST.ANNOTATION_KEY and annotation):
                labels = annotation.celltypes
                labels.sort()
            else: 
                labels = list(set(rna.obs[umap_cat]))

                if ( labels[0] in ['True', 'False'] ):
                    labels = ['True', 'False']

            if len(labels) == 2:
                colors = ['black', 'yellow']
            else:
                colors = helperfuncs.generate_distinct_colors(len(labels))
            
        else:
            labels = [int(x) for x in list(set(rna.obs[col_color]))]
            labels.sort()
            labels = [str(x) for x in labels]
            colors = leiden_colors

        # Save color to use for later
        if ( umap_cat == CONST.ANNOTATION_KEY ):
            celltype_colors = colors
            rna.uns['spoqc_celltype_colors'] = list(zip(labels, colors))
        if ( umap_cat == 'leiden' ):
            rna.uns['spoqc_leiden_colors'] = list(zip(labels, colors))

        # ==========================
        # Overlay: cluster centroids
        # ==========================
        # Compute per-cluster means of UMAP coords and umap_cat (numeric only)
        
        if ( rna.obs[umap_cat][0] in ['True', 'False'] ):
            rna.obs[umap_cat] = [1 if x == "True" else 0 for x in rna.obs[umap_cat]]

        if is_numeric_dtype(rna.obs[umap_cat]):

            for mode in ['mean', 'median']:

                # --- base scatter ---
                # Build a DataFrame with everything we need
                df = pd.DataFrame({
                    'leiden': rna.obs['leiden'].values,
                    'UMAP1': X[:, 0],
                    'UMAP2': X[:, 1],
                    umap_cat: pd.to_numeric(rna.obs[umap_cat].values, errors='coerce')
                })
                
                fig = px.scatter(
                    df, 
                    x="UMAP1", 
                    y="UMAP2",
                    labels={"color": umap_cat},
                    category_orders={"color": labels} if labels is not None else None,
                    color_discrete_sequence=colors if colors is not None else None,
                    color=rna.obs[col_color].tolist() if labels is not None else None,
                    custom_data=[umap_cat]
                )

                # Attach umap_cat per point and format hover
                if ( len(labels) > 2 and umap_cat != CONST.ANNOTATION_KEY ):
                    fig.update_traces(hovertemplate=
                        "UMAP1=%{x:.3f}<br>"
                        "UMAP2=%{y:.3f}<br>"
                        f"{umap_cat}=%{{customdata[0]:.4g}}"
                        "<extra></extra>"
                    )
                if ( len(labels) <= 2 ):
                    fig.update_traces(hovertemplate=
                        "UMAP1=%{x:.3f}<br>"
                        "UMAP2=%{y:.3f}<br>"
                        "<extra></extra>"
                    )

                fig.update_layout(legend={'itemsizing': 'constant'})
                fig.update_traces(marker_size=2)
                fig.update_xaxes(range=xrange, autorange=False)
                fig.update_yaxes(range=yrange, autorange=False, scaleanchor="x", scaleratio=1)

                cent = df.groupby('leiden', as_index=False).agg({
                    'UMAP1': 'mean',
                    'UMAP2': 'mean',
                    umap_cat: mode
                }).rename(columns={'UMAP1': 'centroid_x', 'UMAP2': 'centroid_y'})

                # (Optional) keep/update a running table
                # if 'cluster_centroid_df' not in globals():
                #     cluster_centroid_df = cent.copy()
                # else:
                #     cluster_centroid_df = cluster_centroid_df.drop(columns=[umap_cat], errors='ignore') \
                #                                              .merge(cent[['leiden', umap_cat]], on='leiden', how='outer')

                # Size markers by the centroid value (linear mapping)
                vals = cent[umap_cat].to_numpy()
                vmin = np.nanmin(vals)
                vmax = np.nanmax(vals)
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
                    sizes = np.full_like(vals, 14, dtype=float)  # fallback
                else:
                    # map to, e.g., 8..40 px
                    sizes = 8 + (vals - vmin) * (40 - 8) / (vmax - vmin)

                # Build label -> color mapping that matches px.scatter's mapping
                # (labels and colors already defined above for the base plot)
                label_to_color = {str(lbl): colors[i] for i, lbl in enumerate(labels)}

                # Colors for each centroid in the same order as 'cent' rows
                centroid_colors = [ label_to_color.get(str(lv), colors[0]) for lv in cent['leiden'] ]

                # --- centroid overlay (match renderer) ---
                centroid_trace = go.Scattergl(                 # <— match WebGL
                    x=cent['centroid_x'],
                    y=cent['centroid_y'],
                    name="__centroids__",                      # give it a name so we can reorder if needed
                    mode='markers+text',
                    showlegend=False,
                    marker=dict(
                        size=sizes,
                        color=centroid_colors,
                        line=dict(width=1, color='white')
                    ),
                    text=cent['leiden'].astype(str),
                    textposition='middle center',
                    textfont=dict(color='white'),
                    textfont_size=14,
                    customdata=np.stack([cent['leiden'].astype(str).values, vals], axis=1),
                    hovertemplate=(
                        "cluster=%{customdata[0]}<br>"
                        "UMAP1=%{x:.3f}<br>"
                        "UMAP2=%{y:.3f}<br>"
                        f"{umap_cat}=%{{customdata[1]:.4g}}"
                        "<extra></extra>"
                    ),
                )
                fig.add_trace(centroid_trace)

                # (Optional but safe) ensure centroids are the LAST traces (topmost)
                # Rebuild fig.data with centroid traces moved to the end.
                centroid_idxs = [i for i, t in enumerate(fig.data) if getattr(t, "name", "") == "__centroids__"]
                other_idxs = [i for i in range(len(fig.data)) if i not in centroid_idxs]
                fig.data = tuple([fig.data[i] for i in other_idxs + centroid_idxs])

                # --- save ---
                plot_suffix = ''
                if ( umap_cat in cell_metrices ):
                    plot_suffix = '_hqcr' # this helps me later to sort plots
                
                fig.write_html(f"{figure_path}/umap_plot_{mode}_{umap_cat + plot_suffix}.html")
                helperfuncs.plotly_save_as_png(fig, f"{figure_path}/umap_plot_{mode}_{umap_cat + plot_suffix}.png")
                helperfuncs.plotly_save_as_png(fig, f"{figure_path}/umap_plot_{mode}_{umap_cat + plot_suffix}.pdf")
        

            if ( umap_cat in ['nuceli_count', 'control_probe_counts']):

                # --- barplot ---
                t = 0.0
                if ( umap_cat == 'nuceli_count' ):
                    t = 1.0

                pct_umap_cat = [-1] * len(set(rna.obs['leiden']))
                leiden_cluster_list = []

                for g in rna.obs.groupby('leiden'):
                    pct_umap_cat[int(g[0])] = (g[1][g[1][umap_cat] > t].size / g[1].size) * 100
                    leiden_cluster_list.append(int(g[0]))

                df = pd.DataFrame({'leiden': leiden_cluster_list, f'pct': pct_umap_cat})
                df = df.sort_values('leiden').reset_index(drop=True)
                df['leiden'] = df['leiden'].astype(str)

                fig_pct = px.bar(
                    df,
                    x="leiden",
                    y="pct",
                    title=f"Percentage of cells having {umap_cat} > {t}"
                )
                fig_pct.update_yaxes(range=[0, 100])
                fig_pct.update_layout(xaxis_title="leiden cluster")
                fig_pct.write_html(f"{figure_path}/barplot_pct_{umap_cat}.html")
                fig_pct.write_image(f"{figure_path}/barplot_pct_{umap_cat}.png", scale=3)
                fig_pct.write_image(f"{figure_path}/barplot_pct_{umap_cat}.pdf", scale=3)

            else:

                if ( umap_cat in ['convexhull_outside_trnascripts']):
                    rna.obs['log10p1_convexhull_outside_trnascripts'] =  np.log10(( np.array(rna.obs[umap_cat]) + 1) )
                    umap_cat = 'log10p1_convexhull_outside_trnascripts'

                # --- boxplots ---
                boxplot_trace = go.Box(
                    x=rna.obs['leiden'],
                    y=rna.obs[umap_cat]
                )

                # Create the layout
                layout = go.Layout(
                    xaxis=dict(
                        tickangle=-45,
                        tickfont=dict(size=18),
                        title="leiden cluster"
                    ),
                    yaxis=dict(
                        title=umap_cat,
                        tickfont=dict(size=18)
                    ),
                    height=500,
                    width=800
                )

                layout.xaxis.update(
                    categoryorder='array',
                    categoryarray=sorted(rna.obs['leiden'].unique(), key=int)
                )

                # Create the figure
                fig = go.Figure(data=[boxplot_trace], layout=layout)

                helperfuncs.apply_general_plotly_layout(fig, False)

                fig.write_html(f"{figure_path}/boxplot_{umap_cat}.html")
                fig.write_image(f"{figure_path}/boxplot_{umap_cat}.png", scale=3)
                fig.write_image(f"{figure_path}/boxplot_{umap_cat}.pdf", scale=3)


                # --- violin plots ---
                violin_trace = go.Violin(
                    x=rna.obs['leiden'],
                    y=rna.obs[umap_cat],
                    box_visible=True,        # show embedded boxplot
                    box=dict(fillcolor="white", line=dict(color="black")),
                    meanline_visible=True,   # show mean line
                    points=False,            # set to 'all' if you want scatter points
                    spanmode='hard'          # truncate violin at actual data min/max
                )

                # Create the layout
                layout = go.Layout(
                    xaxis=dict(
                        tickangle=-45,
                        tickfont=dict(size=18),
                        title="leiden cluster"
                    ),
                    yaxis=dict(
                        title=umap_cat,
                        tickfont=dict(size=18)
                    ),
                    height=500,
                    width=800
                )

                layout.xaxis.update(
                    categoryorder='array',
                    categoryarray=sorted(rna.obs['leiden'].unique(), key=int)
                )

                # Create the figure
                fig = go.Figure(data=[violin_trace], layout=layout)

                helperfuncs.apply_general_plotly_layout(fig, False)

                fig.write_html(f"{figure_path}/violin_{umap_cat}.html")
                fig.write_image(f"{figure_path}/violin_{umap_cat}.png", scale=3)
                fig.write_image(f"{figure_path}/violin_{umap_cat}.pdf", scale=3)

        else:

            # Build a DataFrame with everything we need
            df = pd.DataFrame({
                'UMAP1': X[:, 0],
                'UMAP2': X[:, 1],
                umap_cat: rna.obs[umap_cat].values
            })

            fig = px.scatter(
                df,
                x="UMAP1",
                y="UMAP2",
                color=umap_cat,
                labels={"color": umap_cat},
                category_orders={umap_cat: labels} if labels is not None else None,
                color_discrete_sequence=colors if colors is not None else None,
                custom_data=[umap_cat]
            )

            # Attach umap_cat per point and format hover
            if ( len(labels) > 2 and umap_cat != CONST.ANNOTATION_KEY ):
                fig.update_traces(hovertemplate=
                    "UMAP1=%{x:.3f}<br>"
                    "UMAP2=%{y:.3f}<br>"
                    f"{umap_cat}=%{{customdata[0]}}"
                    "<extra></extra>"
                )
            if ( len(labels) <= 2 ):
                fig.update_traces(hovertemplate=
                    "UMAP1=%{x:.3f}<br>"
                    "UMAP2=%{y:.3f}<br>"
                    "<extra></extra>"
                )

            fig.update_layout(legend={'itemsizing': 'constant'})
            fig.update_traces(marker_size=2)
            fig.update_xaxes(range=xrange, autorange=False)
            fig.update_yaxes(range=yrange, autorange=False, scaleanchor="x", scaleratio=1)

            plot_suffix = ''
            if ( umap_cat in cell_metrices ):
                plot_suffix = '_hqcr' # this helps me later to sort plots
            
            fig.write_html(f"{figure_path}/umap_plot_{umap_cat + plot_suffix}.html")
            helperfuncs.plotly_save_as_png(fig, f"{figure_path}/umap_plot_{umap_cat + plot_suffix}.png")
            helperfuncs.plotly_save_as_png(fig, f"{figure_path}/umap_plot_{umap_cat + plot_suffix}.pdf")

        if ( umap_cat in ['doublet', 'nucleus_free', 'border_cell'] ):

            for x in ['leiden', CONST.ANNOTATION_KEY]:

                bar_plot_pct = pd.DataFrame(
                    rna.obs
                    .groupby(x)[umap_cat]
                    .mean() * 100
                )
                bar_plot_pct.columns = ['pct']
                bar_plot_pct[x] = list(bar_plot_pct.index)

                fig_pct = px.bar(
                    bar_plot_pct,
                    x=x,
                    y="pct",
                    title=f"Percentage of cells classified as {umap_cat}"
                )
                fig_pct.update_yaxes(range=[0, 100])
                fig_pct.update_layout(xaxis_title="leiden cluster")
                fig_pct.write_html(f"{figure_path}/barplot_pct_{umap_cat}_{x}.html")
                fig_pct.write_image(f"{figure_path}/barplot_pct_{umap_cat}_{x}.png", scale=3)
                fig_pct.write_image(f"{figure_path}/barplot_pct_{umap_cat}_{x}.pdf", scale=3)

    # --- cell composition plot -----
    ctf_df = create_celltype_fraction_df('leiden', CONST.ANNOTATION_KEY, rna)
    n_labels = len(set(ctf_df['label']))
    n_x = len(set(ctf_df['x']))
    fig = px.bar(
        ctf_df, x="x", y="fractions", 
        labels={"x": 'leiden'},
        color="label", 
        color_discrete_sequence=celltype_colors
    )
    fig.update_layout(
        width=max(1000, n_x * 20),
        height=max(600, n_labels * 30)
    )
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(range=[0, 1.0])
    fig.write_html(f"{figure_path}/fractions_celltype_leiden.html")
    fig.write_image(f"{figure_path}/fractions_celltype_leiden.png", scale=3)
    fig.write_image(f"{figure_path}/fractions_celltype_leiden.pdf", scale=3)

    done_file = open(f"{figure_path}/done.txt", "w")
    done_file.write("its done")
    done_file.close()

    # --- beliefs filtering plot -----
    filter_cols = ['hqcr_beliefs', 'hqtr_beliefs']
    for staining in stainings:
        filter_cols.append(f'hqpr_{staining}_beliefs')
    for org_c in filter_cols:

        t = 0.3
        if ( org_c.startswith('hqpr') ):
            t = 0.2
        if ( org_c.startswith('hqtr') ):
            t = 0.1

        c = f'{org_c}_filtered'
        rna.obs[c] = [True] * rna.n_obs
        rna.obs[c] = rna.obs[org_c] < t

        helperfuncs.plot_scatter(
            rna,
            figure_path,
            c,
            None,
            c,
            ['lightblue', 'black'], 
            c
        )

        # Build a DataFrame with everything we need
        df = pd.DataFrame({
            'UMAP1': X[:, 0],
            'UMAP2': X[:, 1],
            c: rna.obs[c].values
        })

        # Decide your desired order once:
        bool_order = [False, True]
        color_map = {False: 'lightblue', True: 'black'}

        fig = px.scatter(
            df,
            x="UMAP1",
            y="UMAP2",
            color=c,
            category_orders={c: bool_order},
            color_discrete_map=color_map
        )

        fig.update_layout(legend={'itemsizing': 'constant'})
        fig.update_traces(marker_size=2)
        fig.update_xaxes(range=xrange, autorange=False)
        fig.update_yaxes(range=yrange, autorange=False, scaleanchor="x", scaleratio=1)
        fig.write_html(f"{figure_path}/umap_plot_{c}.html")
        helperfuncs.plotly_save_as_png(fig, f"{figure_path}/umap_plot_{c}.png")
        helperfuncs.plotly_save_as_png(fig, f"{figure_path}/umap_plot_{c}.pdf")

    # Read in sdata.attrs (bounding boxes) and write it into the adata.uns as metadata
    if ( subdir == 'overview' ):
        hqr_metadata_dic = dict()
        for modality in ['hqcr', 'hqpr', 'hqtr']:
            metadata_file = ""

            if ( modality == 'hqcr' ):
                metadata_path = f"{CONST.FIGURE_PATH}/{modality}/hqcr_ident/"
                metadata_file = f"{metadata_path}/{modality}s.json"
                if ( os.path.exists(metadata_file)):
                    with open(metadata_file, "r") as f:
                        metadata_json = json.load(f)
                    hqr_metadata_dic['hqcr'] = metadata_json

            if ( modality == 'hqpr' ):
                for staining in stainings:
                    metadata_path = f"{CONST.FIGURE_PATH}/{modality}/{staining}/hqpr_bounding_box/"
                    metadata_file = f"{metadata_path}/{modality}s_{staining}.txt"
                    if ( os.path.exists(metadata_file)):
                        with open(metadata_file, "r") as f:
                            content = f.read()
                            metadata_list = ast.literal_eval(content)
                        hqr_metadata_dic[f'{modality}_{staining}'] = metadata_list

            if ( modality == 'hqtr' ):
                metadata_path = f"{CONST.FIGURE_PATH}/{modality}/hqtr_bounding_box/"
                metadata_file = f"{metadata_path}/{modality}s.txt"
                if ( os.path.exists(metadata_file)):
                    with open(metadata_file, "r") as f:
                        content = f.read()
                        metadata_list = ast.literal_eval(content)
                    hqr_metadata_dic[f'{modality}'] = metadata_list
            
            sdata.table.uns = hqr_metadata_dic

        # Write out annotated .h5ad
        # Remove columns that are not useful for inspection.
        if ( 'nuclei_idxs' in sdata.table.obs.columns ):
            sdata.table.obs.drop(columns=['nuclei_idxs'], inplace=True)
        sdata.table.write_h5ad(
            f"{CONST.FIGURE_PATH}/analysis/rna_qc_annotated.h5ad", 
            compression="gzip", 
            compression_opts=9
        )
        
    if ( subdir == 'cluster' ):
        if ( 'nuclei_idxs' in rna.obs.columns ):
            rna.obs.drop(columns=['nuclei_idxs'], inplace=True)
        rna.write_h5ad(
            f"{CONST.FIGURE_PATH}/analysis/rna_cluster.h5ad", 
            compression="gzip", 
            compression_opts=9
        )

# %%
