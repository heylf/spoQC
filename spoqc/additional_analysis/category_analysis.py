# In[]
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px   # plotly
import nbformat               # plotly related
import plotly.graph_objects as go
import os
import dask.dataframe as dd

from sklearn.metrics import silhouette_score
from pandas.api.types import is_numeric_dtype

from .. import helperfuncs
from . import analysis_funcs

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

    cell_metrices = analysis_funcs.load_cell_metrices(sdata, spoqc_tmp_folder, CONST)
    umap_cats.extend(cell_metrices)

    figure_path = f'{CONST.FIGURE_PATH}/analysis/{subdir}'
    rna = sdata['table']
    rna.X = rna.layers['normlog']

    ####################################################################################################################
    # Do the actual analysis
    ####################################################################################################################
    umap_cats.extend(analysis_funcs.map_modality_metrics_to_cells(
        sdata, imagedim, image_type, resolution, spoqc_tmp_folder, suffix, dim_x, dim_y, stainings, figure_path
    ))

    bool_colors = {False: "#4C78A8", True: "#E45756"}
    log1p_umap_cat = ['convexhull_outside_trnascripts']

    figures = []
    for cat in ['border_cell', 'doublet', 'nucleus_free']:
        if ( not os.path.exists(f"{figure_path}/{cat}") ):
            os.makedirs(f"{figure_path}/{cat}")

        df = sdata['table'].obs.copy()

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
# %%
