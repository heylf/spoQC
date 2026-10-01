import pandas as pd
import numpy as np
import plotly.express as px
from .. import helperfuncs
from .. import subworkflows
from spoqc.core import raster, threads
from spoqc.core.figures import save_figure

def start_image_celltype_analysis(
        sdata,
        figure_path,
        spoqc_tmp_folder,
        modality,
        image_type,
        resolution,
        imagedim,
        dim_x,
        dim_y,
        annotation_key,
        canorm,
        *,
        staining=None
    ):

    prefix = modality
    if ( staining ):
        figure_path = f'{figure_path}/{modality}/{modality}_celltype/{staining}/'
        prefix = f'{modality}_{staining}'
    else:
        figure_path = f'{figure_path}/{modality}/{modality}_celltype/'

    # Load data
    # Image df
    qc_metrics = ['as_score', 's_score', 'intensity']

    # One read of mask_raw for all four columns. The frames carry the pixel index 0..n-1 that
    # dask read back from the parts ('__null_dask_index__').
    n_pixels = dim_x * dim_y
    columns = raster.read_pixel_columns(
        f'{spoqc_tmp_folder}/{prefix}_output_mask_raw', [*qc_metrics, f'{prefix}_mask'], n_pixels, threads.budget()
    )
    image_df = pd.DataFrame({metric: columns[metric] for metric in qc_metrics},
                            index=pd.Index(np.arange(n_pixels), name='index'))
    image_df['intensity'] = np.log10( image_df['intensity'] + 1 )

    mask_df = pd.DataFrame({f'{prefix}_mask': columns[f'{prefix}_mask']}, index=pd.Index(np.arange(n_pixels)))
    del columns

    for col in image_df.columns:
        image_df[col] = np.flipud(np.array(image_df[col]).reshape(dim_x, dim_y)).flatten()

    # Just sanity check for image orientation
    helperfuncs.plot_pixels(
        figure_path,
        np.array(image_df[f'intensity']).reshape(dim_x, dim_y),
        imagedim,
        'intensity_check_rotation', 
        'Log10 Pixel Energy', 
        'hot',
        False,
        False
    )

    counts = 'transcript_counts'
    if ( canorm ):
        counts = 'canorm_transcript_counts'

    # Cell df
    helperfuncs.read_sdata_parquet_tmp_files(sdata, spoqc_tmp_folder, 'hqcr')
    cell_df = subworkflows.hqcr.load_cell_df(counts, sdata)
    cell_df[annotation_key] = sdata['table'].obs[annotation_key]
    cell_df['nucleus_free'] = sdata['table'].obs['wnucleus_free']
    subworkflows.hqcr.cell_artefact_assignment(cell_df, sdata)

    figures = []
    for object in ['cell']:
        polys = subworkflows.hqcr.create_polygon_dataframe(sdata, imagedim, f'{object}_boundaries')

        for qc_metric in qc_metrics:
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, 
                                    image_df[qc_metric], qc_metric, figure_path, 'mean_values')

            fig = px.violin(
                sdata['table'].obs,
                x=qc_metric,
                y=annotation_key,
                color='artefact',
                box=False,
                title=f'{qc_metric} of {object}'
            )
            fig.update_layout(width=800, height=2500, violinmode='overlay')
            figures.append(fig)
            save_figure(fig, f"{figure_path}/split_violinplot_{qc_metric}_{object}.png", f"{figure_path}/split_violinplot_{qc_metric}_{object}.pdf", scale=3)

        if ( object == 'cell' ):
            subworkflows.hqcr.map_values_to_cells(sdata, polys, image_type, resolution, 
                                    mask_df[f'{prefix}_mask'], f'{prefix}_class', figure_path, 'markov_labels')
            
            bar_plot_df_1 = (
                sdata['table'].obs
                .groupby(annotation_key)[f'{prefix}_class']
                .apply(lambda x: (x == 1).sum())
                .reset_index(name=f'num_class')
            )
            bar_plot_df_1['class'] = [modality] * len(bar_plot_df_1)

            bar_plot_df_2 = (
                sdata['table'].obs
                .groupby(annotation_key)[f'{prefix}_class']
                .apply(lambda x: (x == 0).sum())
                .reset_index(name=f'num_class')
            )
            bar_plot_df_2['class'] = [modality.replace('h','l')] * len(bar_plot_df_1)

            bar_plot_df = pd.concat([bar_plot_df_1, bar_plot_df_2])

            # Bar plot of artefact scores
            fig_bar = px.bar(
                bar_plot_df,
                x=f'num_class',
                y=annotation_key,
                color='class',
                orientation='h',
                title=f'Number of cells classified as {modality} per celltype'
            )
            fig_bar.update_layout(barmode="stack", height=2500)
            figures.append(fig_bar)
            save_figure(fig_bar, f"{figure_path}/barplot_celltypes_{modality}.png", f"{figure_path}/barplot_celltypes_{modality}.pdf", scale=3)

            bar_plot_pct = (
                bar_plot_df
                .groupby("celltype")
                .apply(lambda g: g.assign(pct=100 * g["num_class"] / g["num_class"].sum()))
                .reset_index(drop=True)
            )
            bar_plot_pct["pct_rounded"] = bar_plot_pct["pct"].round(2)

            fig_pct = px.bar(
                bar_plot_pct,
                x="pct",
                y="celltype",
                color="class",
                orientation="h",
                title=f"Percentage of cells classified as {modality} per celltype",
                text="pct_rounded"
            )

            fig_pct.update_layout(barmode="stack", xaxis_title="Percentage (%)", height=2500)
            figures.append(fig_pct)
            save_figure(fig_pct, f"{figure_path}/barplot_celltypes_pct_{modality}.png", f"{figure_path}/barplot_celltypes_pct_{modality}.pdf", scale=3)


    # Generate plotly HTML
    html_content = ''.join(fig.to_html(full_html=False) for fig in figures)
    with open(f"{figure_path}/celltype_qc_analysis.html", "w") as f:
        f.write(html_content)
