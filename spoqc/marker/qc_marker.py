import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import concurrent.futures
import scanpy as sc

from plotly.subplots import make_subplots

from .. import helperfuncs

def plot_marker_density_and_scatter(sdata, figure_path, markers, name):

    df = pd.DataFrame({
        'x': sdata['table'].obsm['spatial'][:,0],
        'y': sdata['table'].obsm['spatial'][:,1]
    })

    # Get expression values for each marker
    for celltype_markers in markers.values():
        for marker in celltype_markers:
            idx = list(sdata['table'].var.index).index(marker)
            df[marker] = sdata['table'].X.todense()[:,idx].flatten().tolist()[0]

    for celltype in markers:

        celltype_markers = markers[celltype]

        # Set up the plot grid
        plt.figure(figsize=(20, 5))

        for i, marker in enumerate(celltype_markers):
            plt.subplot(1, len(celltype_markers), i + 1)
            
            try:
                # Create a weighed kernel density estimate (KDE) plot. 
                # The kenrel is weighted by the expression of the marker.
                sns.kdeplot(data=df, x='x', y='y', weights=marker, cmap="viridis", 
                            shade=True, bw_adjust=0.5, cbar=True)

            except Exception as e:
                 # Handle errors by displaying an error message and creating an empty plot
                print(f"[ERROR] Not enough data to create KDE plot for marker '{marker}': {e}")

            plt.title(f'{marker}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim(np.min(df['x']), np.max(df['x']))
            plt.ylim(np.min(df['y']), np.max(df['y']))

            plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.savefig(f'{figure_path}/densityplot_{name}_{celltype}_markers.png', bbox_inches='tight', dpi=300)
        plt.close()

        # Set up the plot grid
        plt.figure(figsize=(20, 5))

        for i, marker in enumerate(celltype_markers):
            plt.subplot(1, len(celltype_markers), i + 1)
            
            sns.scatterplot(data=df, x='x', y='y', hue=marker, s=20, legend=False)

            plt.title(f'{marker}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim(np.min(df['x']), np.max(df['x']))
            plt.ylim(np.min(df['y']), np.max(df['y']))

            plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.savefig(f'{figure_path}/scatterplot_{name}_{celltype}_markers.png', bbox_inches='tight', dpi=300)
        plt.close()


        # Set up the plot grid
        plt.figure(figsize=(20, 5))

        for i, marker in enumerate(celltype_markers):
            plt.subplot(1, len(celltype_markers), i + 1)

            plt.title(f'{marker}')

            try:
                sns.scatterplot(data=df, x='x', y='y', s=20, hue=marker, legend=False)
                sns.kdeplot(
                    data=df, x='x', y='y', cmap="viridis", weights=marker,
                    shade=True, alpha=0.5, bw_adjust=0.5, cbar=True
                )
                
            except Exception as e:
                 # Handle errors by displaying an error message and creating an empty plot
                print(f"[ERROR] Not enough data to create KDE plot for marker '{marker}': {e}")


            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim(np.min(df['x']), np.max(df['x']))
            plt.ylim(np.min(df['y']), np.max(df['y']))
            plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.savefig(f'{figure_path}/scatterplot_densityplot_{name}_{celltype}_markers.png', bbox_inches='tight', dpi=300)
        plt.close()



def plot_marker_boxplot(sdata, figure_path, markers, annotation_key, name):

    log2fc_list = []
    celltype_list = []

    rna_adata = sdata['table']
    rna_adata.X = sdata['table'].X

    # Marker expression
    for celltype in markers.keys():

        # Subset data based on the markers
        idx_var = [True if v in markers[celltype] else False for v in rna_adata.var.index]

        num_features = len(markers[celltype])

        celltype_list = celltype_list + [celltype] * num_features

        # Just take cells of celltype
        idx_obs = rna_adata.obs[annotation_key] == celltype
        subset_rna_adata = rna_adata[idx_obs,idx_var]

        mtx = subset_rna_adata.X.todense()
        celltype_mean_list = [np.mean(mtx[:,i]) for i in range(0,num_features)]
            
        # For background take all cells that are not celltype                
        idx_obs = rna_adata.obs[annotation_key] != celltype
        subset_rna_adata = rna_adata[idx_obs,idx_var]

        mtx = subset_rna_adata.X.todense()
        background_mean_list = [np.mean(mtx[:,i]) for i in range(0,num_features)]

        for i in range(0, num_features):
            log2fc_list.append(np.log2(celltype_mean_list[i] / background_mean_list[i]))
        
    df = pd.DataFrame({'celltype': celltype_list,
                    'log2fc': log2fc_list})

    boxplot_trace = go.Box(
            x=df['celltype'],
            y=df['log2fc']
        )

    # Create the layout
    layout = go.Layout(
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=18),
            title='Celltype'
        ),
        yaxis=dict(
            title=f"Log2 foldchange {name.replace('_',' ')} expression",
            tickfont=dict(size=18)
        ),
        height=500,
        width=800,
        title="Celltype vs Non-Celltype"
    )

    # Create the figure
    fig = go.Figure(data=[boxplot_trace], layout=layout)

    helperfuncs.apply_general_plotly_layout(fig, False)

    fig.write_html(f"{figure_path}/boxplot_{name}_plot.html")
    fig.write_image(f"{figure_path}/boxplot_{name}_plot.png", scale=3)


def compute_radius_lists(rna_adata, radius, annotation_key, markers, figure_path, name):

    cell_spatial_coords = pd.DataFrame({
                            'x': rna_adata.obsm['spatial'][:, 0],
                            'y': rna_adata.obsm['spatial'][:, 1]
                        })

    cells_lists = helperfuncs.points_within_radius(cell_spatial_coords, radius, False)

    # These are used for plotting (dataframe) later
    celltype_mean_list = []
    celltype_list = []
    celltype_maker_list = []

    if ( len(cells_lists) == 0 ):
        sys.exit(f"[ERROR] The radius {radius} you have chosen is too small.")

    for celltype in markers.keys():

        num_features = len(markers[celltype])

        celltype_list = celltype_list + [celltype] * num_features

        celltype_maker_list = celltype_maker_list + markers[celltype]

        # This will hold the mean marker expression of the cells in the radius of the cell.
        cells_mean_list = []

        # Just take cells that are annotated with the selected celltype.
        idx_obs = rna_adata.obs[annotation_key] == celltype
        cells_to_look_at = rna_adata[idx_obs,:].obs.index

        map_barcode_to_index = list(rna_adata[idx_obs,:].obs.index)

        # Go over each cell and get the cells that are in the radius of the cell.
        for j in cells_to_look_at:

            # Get the markers.
            idx_var = [True if v in markers[celltype] else False for v in rna_adata.var.index]

            # Just check if there are any cells in the radius of the cell.
            if ( len(cells_lists[map_barcode_to_index.index(j)]) > 0 ):

                subset_rna_adata = rna_adata[cells_lists[map_barcode_to_index.index(j)], idx_var]

                mtx = subset_rna_adata.X.todense()

                # Calculate the mean expression of each marker gene for the cells in the radius of the cell.
                cells_mean_list.append([np.mean(mtx[:,i]) for i in range(0,num_features)])

        cells_mean_list = np.array(cells_mean_list)

        # This is now the mean of the means of the cells in the radius of each cell.
        if ( len(cells_mean_list) != 0 ):
            celltype_mean_list = celltype_mean_list + [np.mean(cells_mean_list[:,i]) for i in range(0,num_features)]
        else:
            sys.exit(f"[ERROR] The radius {radius} you have chosen was too small for celltype {celltype}.") 

    df = pd.DataFrame({'celltype': celltype_list,
                    'celltype_means': celltype_mean_list})

    boxplot_trace = go.Box(
            x=df['celltype'],
            y=df['celltype_means']
        )

    # Create the layout
    layout = go.Layout(
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=10),
            title='Celltype'
        ),
        yaxis=dict(
            title=f"Mean {name.replace('_',' ')} expression"
        ),
        height=500,
        width=800
    )

    # Create the figure
    fig = go.Figure(data=[boxplot_trace], layout=layout)

    fig.write_html(f"{figure_path}/boxplot_{name}_{radius}.html")

    return celltype_list, celltype_mean_list, celltype_maker_list, [radius] * len(celltype_mean_list)


def plot_marker_radius_line(sdata, figure_path, markers, name, threads, annotation_key, radi):
    
    rna_adata = sdata['table']
    rna_adata.X = sdata['table'].X

    radius_celltype_mean_list = []
    radius_celltype_list = []
    radius_celltype_maker_list = []
    radius_list = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(compute_radius_lists, rna_adata, radius, annotation_key, 
                                   markers, figure_path, name) for radius in radi]
        for future in concurrent.futures.as_completed(futures):
            results = future.result()
            radius_celltype_list = radius_celltype_list + results[0]
            radius_celltype_mean_list = radius_celltype_mean_list + results[1]
            radius_celltype_maker_list = radius_celltype_maker_list + results[2]
            radius_list = radius_list + results[3]

    df = pd.DataFrame({
        'radius': radius_list,
        'celltype_marker': [f'{radius_celltype_list[i]}_{radius_celltype_maker_list[i]}' \
                            for i in range(0, len(radius_celltype_list))],
        'celltype_means': radius_celltype_mean_list,
        'celltype': radius_celltype_list
    })

    fig = make_subplots()

    for marker in np.unique(df['celltype_marker']):
        subset_df = df[df['celltype_marker'] == marker]
        # Add each marker as a trace.
        fig.add_trace(
            go.Scatter(x=subset_df['radius'], y=subset_df['celltype_means'], name=marker)
        )

    # Add titles and labels
    fig.update_layout(
        width = 1000,
        height = 500,
        xaxis_title="Radius",
        yaxis_title=f"Mean {name.replace('_',' ')} expression",
         xaxis=dict(range=[min(radius_list), max(radius_list)])
    )

    helperfuncs.apply_general_plotly_layout(fig, True)

    fig.update_traces(textfont=dict(size=18))
    fig.write_html(f"{figure_path}/lineplot_{name}_all_radiuses.html")
    fig.write_image(f"{figure_path}/lineplot_{name}_all_radiuses.png", scale=3)


def plot_sanpy_score_genes(sdata, figure_path, markers, name):

    rna_adata = sdata['table']

    df = pd.DataFrame({
            'x': sdata['table'].obsm['spatial'][:,0],
            'y': sdata['table'].obsm['spatial'][:,1]
        })

    celltype_list = []
    scores_list = []
    for i, celltype in enumerate(markers.keys()):
        sc.tl.score_genes(rna_adata, markers[celltype], ctrl_size=50, gene_pool=list(rna_adata.var.index), 
                        n_bins=25, score_name=f'{name}_{celltype}_score', random_state=0, copy=False, use_raw=False)
        scores = list(rna_adata.obs[f'{name}_{celltype}_score'])
        df[f'{name}_{celltype}_score'] = scores
        df[f'{name}_{celltype}_score_shifted'] = helperfuncs.min_value_shift(df[f'{name}_{celltype}_score'])
        celltype_list = celltype_list + [celltype]*len(scores)
        scores_list = scores_list + scores

    # Set up the plot grid
    plt.figure(figsize=(20, 5))

    for i, celltype in enumerate(markers.keys()):
        plt.subplot(1, len(markers.keys()), i + 1)
        
        plt.title(f'{celltype}')

        # Overlay a scatter plot
        sns.scatterplot(data=df, x='x', y='y', s=5, hue=f'{name}_{celltype}_score', legend=False)
        
        sns.kdeplot(
            data=df, x='x', y='y', cmap="viridis", weights=f'{name}_{celltype}_score_shifted',
            shade=True, alpha=0.5, bw_adjust=0.5, cbar=True
        )
            
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(np.min(df['x']), np.max(df['x']))
        plt.ylim(np.min(df['y']), np.max(df['y']))
        plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'{figure_path}/scatterplot_densityplot_{name}_scanpy_gene_scores.png', bbox_inches='tight', dpi=300)
    plt.close()

    df = pd.DataFrame({'celltype': celltype_list,
                    'score': scores_list})

    boxplot_trace = go.Box(
            x=df['celltype'],
            y=df['score']
        )

    # Create the layout
    layout = go.Layout(
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=18),
            title='Celltype'
        ),
        yaxis=dict(
            title=f"Scanpy expression scores",
            tickfont=dict(size=18)
        ),
        height=500,
        width=800
    )

    # Create the figure
    fig = go.Figure(data=[boxplot_trace], layout=layout)

    helperfuncs.apply_general_plotly_layout(fig, False)

    fig.write_html(f"{figure_path}/boxplot_{name}_scanpy_gene_scores_plot.html")
    fig.write_image(f"{figure_path}/boxplot_{name}_scanpy_gene_scores_plot.png", scale=3)
