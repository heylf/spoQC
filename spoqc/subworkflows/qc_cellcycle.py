#In[]
import scanpy as sc
import plotly.express as px   # plotly
import pandas as pd
import anndata

from typing import List, Any

from .. import helperfuncs

def cellcycle_qc(
        rna_adata: anndata.AnnData,
        figure_path: str,
        cell_cycle_genes: List[str],
        s_genes: List[str],
        g2m_genes: List[str],
        plot_colors_phase: List[str] 
    ) -> anndata.AnnData:
    """
    Perform cell cycle quality control (QC) on single-cell RNA sequencing data.

    This function scores cells for S and G2M phases using predefined gene lists,
    calculates PCA on cell cycle genes, and generates visualizations of the cell cycle phases
    and their distributions.

    Args:
        figure_path (str): Path to save the generated figures.
        rna_adata (anndata.AnnData): Annotated data object containing RNA expression data.
        cell_cycle_genes (List[str]): List of cell cycle-related genes to use for PCA calculation.
        s_genes (List[str]): List of genes associated with the S phase of the cell cycle.
        g2m_genes (List[str]): List of genes associated with the G2/M phase of the cell cycle.
        plot_colors_phase (List[str]): Colors to use for plotting cell cycle phases.

    Returns:
        anndata.AnnData: Updated annotated data object with cell cycle scores and phase information.

    Notes:
        - This function uses Scanpy's `score_genes_cell_cycle` and `pca` functions.
        - Visualizations are saved as interactive HTML files using Plotly.
    """

    # cell cycle scoring. 
    # (wrapper to sc.tl.score_gene_list, which is launched twice, to score separately S and G2M phases.) 
    # Both sc.tl.score_gene_list and sc.tl.score_cell_cycle_genes are a port from Seurat. 
    # To score a gene list, the algorithm calculates the difference of mean expression of the given list,
    # and the mean expression of reference genes. 
    # To build the reference, the function randomly chooses a bunch of genes matching the distribution of the expression 
    # of the given list. 
    # Cell cycle scoring adds three slots in data, a score for S phase, 
    # a score for G2M phase and the predicted cell cycle phase.
    sc.tl.score_genes_cell_cycle(rna_adata, s_genes=s_genes, g2m_genes=g2m_genes)

    # Difference from Seurat. 
    # The R package stores raw data, scaled data and variable genes information in separate slots, 
    # Scanpy instead keeps only one snapshot of the data. 
    # This implies that PCA is always calculated on the entire dataset. 
    # In order to calculate PCA reduction using only a subset of genes (like cell_cycle_genes), a trick should be used. 
    # Basically we create a dummy object to store information of PCA projection, 
    # which is then reincorporated into original dataset.
    rna_adata_cc_genes = rna_adata[:, cell_cycle_genes]
    sc.tl.pca(rna_adata_cc_genes, use_highly_variable=False)

    for o in ['phase']:

        colors=[]

        if o == 'phase':
            colors=plot_colors_phase

        fig = px.scatter(
            rna_adata_cc_genes.obsm["X_pca"], x=0, y=1, labels={"0":"PC1", "1":"PC2", "color": "donor"},
            category_orders={"color": sorted(set(list(rna_adata.obs[o])))},
            color_discrete_sequence=colors,
            color=list(rna_adata.obs[o])
        )
        
        fig.update_layout(legend= {'itemsizing': 'constant'}) # keeps marker size bigger
        fig.update_traces(marker_size=10)
        
        fig.write_html(f"{figure_path}/scatter_cellcylce_{o}.html")
        fig.write_image(f"{figure_path}/scatter_cellcylce_{o}.png", scale=3)

    for x in ['sample']:

        fig = px.bar(
            helperfuncs.create_fraction_df(rna_adata, x, 'phase'), x='fractions', y='x', 
            labels={"y": x},
            color='label', 
            color_discrete_sequence=plot_colors_phase
        )
        
        fig.write_html(f"{figure_path}/barplot_{x}_cellcycle_fractions.html")
        fig.write_image(f"{figure_path}/barplot_{x}_cellcycle_fractions.png", scale=3)

    return rna_adata


def spatial_cellcycle_qc(figure_path: str, sdata: Any) -> None:
    """
    Perform spatial cell cycle quality control (QC) by visualizing cell cycle phases 
    in spatial transcriptomics data.

    This function filters the spatial data by its global coordinate system, extracts spatial 
    coordinates and cell cycle phase information, and generates density and scatter plots.

    Args:
        figure_path (str): Path to save the generated figures.
        sdata (Any): Spatial transcriptomics data object with cell cycle phase and spatial information.

    Returns:
        None: The function saves plots to the specified `figure_path` and does not return a value.

    Notes:
        - The spatial data is filtered using a "global" coordinate system.
        - Visualization functions (`plot_density_by_category` and `plot_scatter_by_category`) 
          are assumed to be part of the `helperfuncs` module.
    """

    sdata_filtered_cs = sdata.filter_by_coordinate_system("global")

    df = pd.DataFrame({
        'x': sdata_filtered_cs['table'].obsm['spatial'][:,0],
        'y': sdata_filtered_cs['table'].obsm['spatial'][:,1],
        'phase': sdata_filtered_cs['table'].obs['phase']
    })

    helperfuncs.plot_scatter_density_by_category_df(df, 'phase', figure_path, '1', ['yellow'], None)


def run_qc_cellcycle(sdata, figure_path, CONST):
    rna_adata = sdata.table

    # Get cell cylce genes
    cell_cycle_genes = [x.strip() for x in open(f'{CONST.CELLCYCLE_GENE_FILE}')]
    cell_cycle_genes = list(set(cell_cycle_genes))
    s_genes = cell_cycle_genes[:43]
    g2m_genes = cell_cycle_genes[43:]

    # Filter for genes that are in the sdata
    cell_cycle_genes = list(set(cell_cycle_genes) & set(rna_adata.var_names))
    s_genes = list(set(s_genes) & set(rna_adata.var_names))
    g2m_genes = list(set(g2m_genes) & set(rna_adata.var_names))

    # Just do cellcycle QC if genes are available
    if ( len(s_genes) == 0 ):
        print("[WARN] Sorry it seems your data has no S phase genes")
    elif ( len(g2m_genes) == 0 ):
        print("[WARN] Sorry it seems your data has no G2M phase genes")
    elif ( len(cell_cycle_genes) > 0 ):
        rna_adata = cellcycle_qc(
            rna_adata,
            figure_path,
            cell_cycle_genes,
            s_genes,
            g2m_genes,
            ['red', 'blue', 'yellow']
        )
        spatial_cellcycle_qc(figure_path, sdata)
    else:
        print("[WARN] Something else went wrong")