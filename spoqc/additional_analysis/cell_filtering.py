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
from . import analysis_funcs

def do_filtering(
        sdata,
        subdir,
        CONST,
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

    cell_metrices = analysis_funcs.load_cell_metrices(sdata, spoqc_tmp_folder, CONST, include_nucleus_free=True)
    umap_cats.extend(cell_metrices)

    figure_path = f'{CONST.FIGURE_PATH}/analysis/{subdir}'
    rna = sdata.table
    rna.X = rna.layers['normlog']

    ####################################################################################################################
    # Do the actual analysis
    ####################################################################################################################
    umap_cats.extend(analysis_funcs.map_modality_metrics_to_cells(
        sdata, imagedim, image_type, resolution, spoqc_tmp_folder, suffix, dim_x, dim_y, stainings, figure_path
    ))

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

    # --- beliefs filtering plot -----
    filter_cols = ['hqcr_beliefs', 'hqtr_mask_mean']
    for staining in stainings:
        filter_cols.append(f'hqpr_{staining}_mask_mean')
    for org_c in filter_cols:

        t = 0.0

        # with hqcr we clearly see a bimodal distribution
        if ( org_c.startswith('hqcr') ):
            t = 0.5
        if ( org_c.startswith('hqpr') ):
            t = 0.2
        if ( org_c.startswith('hqtr') ):
            t = 0.2

        # for the rest we take gaussian now
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

    c = f'hqr_filtered'
    rna.obs[c] = rna.n_obs['hqcr_filtered'] & rna.n_obs['hqpr_filtered'] & rna.n_obs['hqtr_filtered']
    
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

    ####################################################################################################################
    # Write out annotated .h5ad
    ####################################################################################################################

    # Remove columns that are not useful for inspection.
    if ( 'nuclei_idxs' in sdata.table.obs.columns ):
        sdata.table.obs.drop(columns=['nuclei_idxs'], inplace=True)

    sdata.table.write_h5ad(
        f"{CONST.FIGURE_PATH}/analysis/rna_qc_annotated.h5ad", 
        compression="gzip", 
        compression_opts=9
    )


# %%
