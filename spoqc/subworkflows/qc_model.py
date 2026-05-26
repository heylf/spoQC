#In[]
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import geopandas as gpd
import scanpy as sc
import shutil

from plotly.subplots import make_subplots
from esda.moran import Moran
from libpysal.weights import Queen

from .. import helperfuncs



def plot_pca_scatter(df, figure_path, nPCs, flip=False):
    nplots = 5
    rows = int(np.ceil(nPCs/nplots))

    plt.figure(figsize=(70, 50))
    for i in range(0, nPCs):
        plt.subplot(rows, nplots, i + 1)
        ax = plt.gca()
        
        # Create a scatter plot with Seaborn
        sns.scatterplot(data=df, x='x', y='y', hue=f'PC{i}', s=10, palette='grey')

        # Add labels and title
        plt.title(f'PC{i+1}')
        plt.xlabel('X')
        plt.ylabel('Y')
        ax.set_aspect('equal', adjustable='box')

        if ( flip ):
            plt.gca().invert_yaxis()

        # Move the legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., markerscale=1)

    plt.tight_layout()
    plt.savefig(f'{figure_path}/scatterplot_PCs.png', bbox_inches='tight')
    plt.savefig(f'{figure_path}/scatterplot_PCs.pdf', bbox_inches='tight')
    plt.close()

    # Individual PC plots
    for i in range(0, nPCs):
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

        # Create a scatter plot with Seaborn
        sns.scatterplot(data=df, x='x', y='y', hue=f'PC{i}', s=1, palette='grey')

        # Add labels and title
        plt.title(f'PC{i+1}')
        plt.xlabel('X')
        plt.ylabel('Y')
        ax.set_aspect('equal', adjustable='box')

        if ( flip ):
            plt.gca().invert_yaxis()

        # Move the legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., markerscale=1)

        plt.tight_layout()
        plt.savefig(f'{figure_path}/scatterplot_PC{i+1}.png', bbox_inches='tight')
        plt.savefig(f'{figure_path}/scatterplot_PC{i+1}.pdf', bbox_inches='tight')
        plt.close()


def plot_spatial_vs_exression_variance(sdata, figure_path, df, nPCs):

    moran_variances = [-1] * nPCs
    moran_Is = [-1] * nPCs

    rna_adata = sdata['table']

    for i in range(0, nPCs):

        print(i)

        # Convert to GeoDataFrame which is needed to take sparsity of spatial data into account.
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))

        # Create spatial-neighbor weights using queen contiguity
        w = Queen.from_dataframe(gdf)

        # Calculate Moran's I with spatial weights.
        # P-value of 0.01 with 99 permutations is not necessarily more significant than a result with 
        # a p-value of 0.001 with 999 permutations. Is is recommended to do 999 permutation. 9999 for more precision.
        moran = Moran(df['PC' + str(i)], w, permutations=999)

        # p_norm = This is the p-value based on the assumption that the statistic follows a normal distribution.
        # p_sim = This is the p-value based on the permutation test, which is a non-parametric method.
        # print(f"Moran's I: {moran.I}")
        # print(f"p-value: {moran.p_sim}")
        # print(f"Variance: {moran.VI_sim}")
        moran_variances[i] = moran.VI_sim
        moran_Is[i] = moran.I


    data = pd.DataFrame({
        'pc': ['PC' + str(i+1) for i in range(0, nPCs)],
        'variance_explained': np.cumsum(rna_adata.uns['pca']['variance'])[:nPCs],
        'spatial_variance': moran_variances,
        'moran_I': moran_Is,
    })

    # Variance Explained (Expression Data)

    # Create a subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add the first trace (for the primary y-axis)
    fig.add_trace(
        go.Scatter(x=data['pc'], y=data['variance_explained'], name="Variance Explained (Expression Data)"),
        secondary_y=False,  # Set to False for primary y-axis
    )

    # Add the second trace (for the secondary y-axis)
    fig.add_trace(
        go.Scatter(x=data['pc'], y=data['spatial_variance'], name="Spatial Variance", line=dict(dash='dash')),
        secondary_y=True,  # Set to True for secondary y-axis
    )

    # Add titles and labels
    fig.update_layout(
        xaxis_title="PC",
        width=2000,
        height=500,
        yaxis_title="Variance Explained (Expression Data)",
    )

    # Set the title for the secondary y-axis
    fig.update_yaxes(title_text="Spatial Variance", secondary_y=True)

    helperfuncs.apply_general_plotly_layout(fig, True)

    fig.write_html(f"{figure_path}/pca_evaluation_spatial_variance.html")
    fig.write_image(f"{figure_path}/pca_evaluation_spatial_variance.png", scale=3)
    fig.write_image(f"{figure_path}/pca_evaluation_spatial_variance.pdf", scale=3)

    # Create a subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add the first trace (for the primary y-axis)
    fig.add_trace(
        go.Scatter(x=data['pc'], y=data['variance_explained'], name="Variance Explained (Expression Data)"),
        secondary_y=False,  # Set to False for primary y-axis
    )

    # Add the second trace (for the secondary y-axis)
    fig.add_trace(
        go.Scatter(x=data['pc'], y=data['moran_I'], name="Moran's I", line=dict(dash='dash')),
        secondary_y=True,  # Set to True for secondary y-axis
    )

    # Add titles and labels
    fig.update_layout(
        xaxis_title="PC",
        width=2000,
        height=500,
        yaxis_title="Variance Explained (Expression Data)",
    )

    # Set the title for the secondary y-axis
    fig.update_yaxes(title_text="Moran's I", secondary_y=True)

    helperfuncs.apply_general_plotly_layout(fig, True)

    fig.write_html(f"{figure_path}/pca_evaluation_moransi.html")
    fig.write_image(f"{figure_path}/pca_evaluation_moransi.png", scale=3)
    fig.write_image(f"{figure_path}/pca_evaluation_moransi.pdf", scale=3)


def run_qc_model(sdata, figure_path, CONST):
    # For model QC we need to get the normalized data
    sdata['table'].X = sdata['table'].layers['normlogscale']
    rna_adata = sdata.table

    sc.tl.pca(rna_adata, n_comps=100)

    df = pd.DataFrame({
            'x': rna_adata.obsm['spatial'][:,0],
            'y': rna_adata.obsm['spatial'][:,1]
        })

    X_pca = rna_adata.obsm['X_pca']

    for i in range(0, CONST.nPCs):
        df[f'PC{i}'] = X_pca[:,i]

    sc.pl.pca_variance_ratio(rna_adata, n_pcs=100, log=True, save='.png')
    shutil.move("figures/pca_variance_ratio.png", f"{figure_path}/pca_variance_ratio.png")
    sc.pl.pca_variance_ratio(rna_adata, n_pcs=100, log=True, save='.pdf')
    shutil.move("figures/pca_variance_ratio.pdf", f"{figure_path}/pca_variance_ratio.pdf")

    plot_pca_scatter(df, figure_path, CONST.nPCs)
    plot_spatial_vs_exression_variance(sdata, figure_path, df, CONST.nPCs)