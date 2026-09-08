import pandas as pd
import numpy as np
import plotly.express as px

from .. import helperfuncs

# Function to print all HQCRs
def plot_hqcr(sdata, figure_path, min_number_good_cells_hqcr, minimum_number_of_total_cells):
    islands = list(set(sdata['table'].obs['island_index']))
    sdata['table'].obs['cell_region'] = np.array(['undefined'] * len(sdata['table']))

    for island in islands:
        island_idxs = sdata['table'].obs['island_index'] == island
        island_adata = sdata['table'][island_idxs]

        num_good_qc_cells = list(island_adata.obs['refined_qc_class']).count(1) # good
        num_bad_qc_cells = list(island_adata.obs['refined_qc_class']).count(0) # bad

        good_bad_ration = 0.0
        if ( num_bad_qc_cells == 0 ):
            good_bad_ration = 100
        else:
            good_bad_ration = num_good_qc_cells / num_bad_qc_cells

        title = ''
        island_suffix = ''
        if ( good_bad_ration > 1.0 ):
            if ( num_good_qc_cells > min_number_good_cells_hqcr ):
                title = f'HQCR {island+1}'
                island_suffix = 'hqcr'
                sdata['table'].obs.loc[island_idxs, 'cell_region'] = 'hqcr'
            else:
                title = f'Small HQCR {island+1}'
                island_suffix = 'small_hqcr'
                sdata['table'].obs.loc[island_idxs, 'cell_region'] = 'small_hqcr'
        else:
            title = f'LQCR {island+1}'
            island_suffix = 'lqcr'
            sdata['table'].obs.loc[island_idxs, 'cell_region'] = 'lqcr'

        # Here I just setelect all the cells that are part of the island to mark them later in the plot.
        island_select = np.array([0] * sdata['table'].n_obs)
        island_select[island_idxs] = 1
        sdata['table'].obs['island_select'] = island_select

        if ( num_good_qc_cells + num_bad_qc_cells > minimum_number_of_total_cells ):
            helperfuncs.plot_scatter(
                island_adata,
                f'{figure_path}/{island_suffix}/',
                f'zoomed_{island_suffix}_{island+1}',
                None,
                None,
                None,
                title
            )
            helperfuncs.plot_scatter(
                sdata['table'],
                f'{figure_path}/{island_suffix}/',
                f'{island_suffix}_{island+1}',
                None,
                'island_select',
                ['lightblue', 'red'],
                title
            )

    hqcr_df = pd.DataFrame({
        'islands': sdata['table'].obs['island_index'], 
        'cell_region': sdata['table'].obs['cell_region']
    })

    hqcr_df.to_json(f"{figure_path}/hqcr.json", orient="columns")


def generate_hqcr_html(figure_path, df_plot, cat, ncat, catnames, qc_metrics):

    figures = []

    qc_metrics = [x for x in qc_metrics if x not in ['celltype']]

    for level in qc_metrics:
        plotname = 'violinplot'
        title = 'Distribution of'
        if ( level == 'doublet' or level == 'nucleus_free' ):
            num_doublets_qc_cluster = [-1] * ncat
            for c in range(0, ncat):
                ndoublets = len([True for x in df_plot[df_plot[cat] == catnames[c]][level] if x == 1])
                num_doublets_qc_cluster[c] = ndoublets

            plot_doublet_df = pd.DataFrame({
                cat: [str(x) for x in catnames],
                level: num_doublets_qc_cluster
            })
            fig = px.bar(
                plot_doublet_df,
                x=level,
                y=cat,
                width=800,
                height=800
            )
            plotname = 'barplot'
            title = 'Count of'
        else:
            if ( level == 'island_score' ):
                fig = px.violin(
                    x=helperfuncs.min_max_normalize(df_plot['island_score']), 
                    y=df_plot[cat],
                    width=800,
                    height=800
                )
                fig.update_layout(
                    xaxis=dict(range=[0, 1.1], title='min-max normalized island score'),
                    yaxis=dict(title=cat)
                )
            else:
                fig = px.violin(
                    df_plot,
                    x=level,
                    y=cat,
                    width=800,
                    height=800
                )

        fig.update_layout(title=f"{title} {level} for all {cat}", showlegend=True)
        helperfuncs.apply_general_plotly_layout(fig, True)

        figures.append(fig)
        fig.write_image(f"{figure_path}/{plotname}_{level}.png", scale=3)
        fig.write_image(f"{figure_path}/{plotname}_{level}.pdf", scale=3)

    with open(f'{figure_path}/hqcr_{cat}.html', 'w') as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))