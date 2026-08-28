import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import plotly.express as px
import nbformat
import plotly.io as pio
import plotly.graph_objects as go

from funkyheatmappy import funky_heatmap
from IPython.display import Image, display

def plot_funkyheatmap(rna, figure_path):

    ####################################################################################################################
    # Setup
    ####################################################################################################################

    ids = list(set(rna.obs['leiden']))
    ids = [int(x) for x in ids]
    ids.sort()
    ids = [str(x) for x in ids]

    # Initial funkyheamap variable
    funky_heatmap_df = pd.DataFrame({'id': ids})
    funky_heatmap_df.index = ids
    column_lists = [
    ["id", "group", "name", "geom", "options", "palette"],
    ["id", np.nan, "leiden cluster", "text", {"ha": 0, "width": 2}, np.nan],
    ]

    # Add beliefs
    cols = ['hqcr_beliefs']
    for col in rna.obs.columns:
        if col.startswith('hqpr') and col.endswith('beliefs'):
            cols.append(col)
    cols.append('hqtr_beliefs')
    
    for col in cols:
        funky_heatmap_df = funky_heatmap_df.join(rna.obs.groupby('leiden')[col].median(), how='left')
        column_lists.append([col, "beliefs", col, "bar", {"width": 2, "legend": False, "scale": False}, "beliefs"])

    # Add cell metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    counts = 'transcript_counts'
    # TODO if ( '' ):
    counts = 'canorm_transcript_counts'
    cell_metrices = [
        counts,
        'n_genes_by_counts',
        'convexity_metric_cell',
        'convexity_min_nuceli',
        'border_scores',
        'thinness_score',
        'island_score',
        'cell_overlap_area',
        'log10p1_convexhull_outside_trnascripts',
        'num_low_qc_transcript'
    ]
    for col in cell_metrices:
        funky_heatmap_df = funky_heatmap_df.join(rna.obs.groupby('leiden')[col].median(), how='left')
        if col == 'log10p1_convexhull_outside_trnascripts':
            column_lists.append([col, "hqcr", 'log10p1_ch_uRNAs', "circle", {"width": 1, "legend": False}, "hqcr"])
        else:
            column_lists.append([col, "hqcr", col, "circle", {"width": 1, "legend": False}, "hqcr"])

    # Add pixel metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    pixel_metrices_s = []
    for col in rna.obs.columns:
        if 'hqpr' in col:
            for m in ['edge_strength', 'energy', 'relevance', 'entropy']:
                if col.startswith(m):
                    pixel_metrices_s.append(col)

    for col in pixel_metrices_s:
        funky_heatmap_df = funky_heatmap_df.join(rna.obs.groupby('leiden')[col].median(), how='left')
        column_lists.append([col, "hqpr_s", col, "circle", {"width": 1, "legend": False}, "hqpr_s"])

    pixel_metrices_as = []
    for col in rna.obs.columns:
        if 'hqpr' in col:
            for m in ['homogenity', 'uniformity']:
                if col.startswith(m):
                    pixel_metrices_as.append(col)

    for col in pixel_metrices_as:
        funky_heatmap_df = funky_heatmap_df.join(rna.obs.groupby('leiden')[col].median(), how='left')
        column_lists.append([col, "hqpr_as", col, "circle", {"width": 1, "legend": False}, "hqpr_as"])

    pixel_metrices_others = []
    for col in rna.obs.columns:
        if 'hqpr' in col:
            for m in ['intensity']:
                if col.endswith(m):
                    pixel_metrices_others.append(col)

    for col in pixel_metrices_others:
        funky_heatmap_df = funky_heatmap_df.join(rna.obs.groupby('leiden')[col].median(), how='left')
        column_lists.append([col, "hqpr_others", col, "circle", {"width": 1, "legend": False}, "hqpr_others"])

    # Add transcripts metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for col in ['edge_strength_hqtr', 'energy_hqtr', 'relevance_hqtr', 'entropy_hqtr']:
        funky_heatmap_df = funky_heatmap_df.join(rna.obs.groupby('leiden')[col].median(), how='left')
        column_lists.append([col, "hqtr_s", col, "circle", {"width": 1, "legend": False}, "hqtr_s"])

    for col in ['homogenity_hqtr', 'uniformity_hqtr']:
        funky_heatmap_df = funky_heatmap_df.join(rna.obs.groupby('leiden')[col].median(), how='left')
        column_lists.append([col, "hqtr_as", col, "circle", {"width": 1, "legend": False}, "hqtr_as"])

    for col in ['hqtr_intensity', 'hqtr_ac_density', 'hqtr_qv_density']:
        funky_heatmap_df = funky_heatmap_df.join(rna.obs.groupby('leiden')[col].median(), how='left')
        column_lists.append([col, "hqtr_others", col, "circle", {"width": 1, "legend": False}, "hqtr_others"])

    # Create column info ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # funkyheatmappy min-max scales each column's value/size/color via. A metric that's identical across every leiden 
    # cluster (e.g. all-NaN after groupby, or genuinely constant) makes that 0/0 = NaN, giving every circle/bar a NaN 
    # size and crashing matplotlib's bbox_inches='tight' computation at savefig time. Keep such columns in the plot but
    # disable their per-column min-max scaling and substitute a fixed neutral value, since the real (constant) value 
    # carries no differentiating signal across clusters anyway.
    CIRCLE_SIZE_FLOOR = 0.05

    for row in column_lists[1:]:
        col = row[0]
        geom = row[3]
        if col == "id":
            continue
        if funky_heatmap_df[col].nunique(dropna=True) <= 1:
            funky_heatmap_df[col] = 0.5
            row[4] = {**row[4], "scale": False}
        # funkyheatmappy's min-max scaling always maps a column's minimum value to exactly 0. For
        # "circle" geoms this becomes the circle radius (r = row_height / 2 * size_value), so the
        # cluster with the lowest value for a metric always gets an invisible, zero-radius circle.
        # Pre-scale those columns into [CIRCLE_SIZE_FLOOR, 1.0] ourselves and disable the library's
        # own scaling so the smallest circle in each column stays visibly non-zero.
        elif geom == "circle":
            cmin = funky_heatmap_df[col].min()
            cmax = funky_heatmap_df[col].max()
            funky_heatmap_df[col] = CIRCLE_SIZE_FLOOR + (1 - CIRCLE_SIZE_FLOOR) * (
                (funky_heatmap_df[col] - cmin) / (cmax - cmin)
            )
            row[4] = {**row[4], "scale": False}

    column_info = pd.DataFrame(column_lists[1:], columns=column_lists[0])
    column_info.index = column_info["id"]

    column_groups = pd.DataFrame(columns=["Category", "group", "subgroup", "palette"],
                                data = [["Beliefs", "beliefs", "\n", "beliefs"],
                                        ["HQCR Metrics", "hqcr", "\n", "hqcr"],
                                        ["HQPR Metrics", "hqpr_s", "S", "hqpr_s"],
                                        ["HQPR Metrics", "hqpr_as", "AS", "hqpr_s"],
                                        ["HQPR Metrics", "hqpr_others", "Others", "hqpr_s"],
                                        ["HQTR Metrics", "hqtr_s", "S", "hqtr_s"],
                                        ["HQTR Metrics", "hqtr_as", "AS", "hqtr_s"],
                                        ["HQTR Metrics", "hqtr_others", "Others", "hqtr_s"],
                                        ]
                                )
    
    ####################################################################################################################
    # Plot
    ####################################################################################################################

    funky_heatmap(
        funky_heatmap_df, 
        column_info = column_info, 
        column_groups = column_groups,
        add_abc=False
    )

    fig = plt.gcf()

    positions = {ax: ax.get_position() for ax in fig.axes}
    main_ax = max(fig.axes, key=lambda ax: positions[ax].width * positions[ax].height)

    # collect non-main small axes = legends
    legend_axes = []
    for ax in fig.axes:
        if ax is main_ax:
            continue
        p = positions[ax]
        if p.width < 0.25 or p.height < 0.25:
            legend_axes.append(ax)

    def ax_text(ax):
        texts = [t.get_text() for t in ax.texts if t.get_text().strip()]
        title = ax.get_title()
        return " ".join(([title] if title else []) + texts).lower()

    # desired palette order
    desired_order = {
        "beliefs": 0,
        "hqcr": 1,
        "hqpr_s": 2,
        "hqpr_as": 3,
        "hqpr_others": 4,
        "hqtr_s": 5,
        "hqtr_as": 6,
        "hqtr_others": 7,
    }

    def legend_key(ax):
        txt = ax_text(ax)
        for k, v in desired_order.items():
            if k in txt:
                return v
        return 999  # unknown legends go last

    legend_axes = sorted(legend_axes, key=legend_key)

    n = len(legend_axes)
    top = 0.80
    bottom = 0.08
    gap = 0
    h = (top - bottom - gap * (n - 1)) / n

    for i, ax in enumerate(legend_axes):
        y = top - (i + 1) * h - i * gap
        ax.set_position([0.92, y, 0.1, h])


    fig.canvas.draw()  # needed to compute text/layout sizes

    renderer = fig.canvas.get_renderer()

    heights = []
    for ax in legend_axes:
        bbox = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        heights.append(bbox.height)

    y = top

    for ax, h in zip(legend_axes, heights):
        y -= h + 0.01
        ax.set_position([0.92, y, 0.1, h])

    for ax in fig.axes:
        # Title
        ax.title.set_fontsize(16)

        # Axis labels
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)

        # Tick labels
        ax.tick_params(axis='both', labelsize=12)

        # Text inside axes (important for legends in funkyheatmap!)
        for text in ax.texts:
            text.set_fontsize(12)

    fig.subplots_adjust(right=0.82)
    plt.savefig(f'{figure_path}/funkyheatmap_1.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{figure_path}/funkyheatmap_1.pdf', bbox_inches='tight', dpi=300)
    plt.close()