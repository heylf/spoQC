import pandas as pd
import plotly.graph_objects as go
import concurrent.futures
import geopandas as gpd
import concurrent.futures

from esda.moran import Moran
from libpysal.weights import Queen

from ... import helperfuncs

# Calculate Moran's I for each gene
def compute_for_gene(gene, rna_adata):
    data = pd.DataFrame({
        'x': rna_adata.obsm['spatial'][:, 0],
        'y': rna_adata.obsm['spatial'][:, 1]
    })
    
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.x, data.y))
    
    # Create spatial-neighbor weights using queen contiguity
    w = Queen.from_dataframe(gdf)

    moran = Moran(rna_adata.X[:, list(rna_adata.var_names).index(gene)].todense(), w, permutations=999)
    
    return [gene, moran.VI_sim, moran.I]


def calculate_global_moran_I_values(sdata, figure_path, spoqc_tmp_folder, threads):

    rna_adata = sdata['table']

    gene_svariance_moransI_list = []

    # Create Moran's I variances and values
    genes_list = list(rna_adata.var_names)

    # TODO 
    # Be careful with the order because of threading the list is filled based on how fast the indidivudal thread is.
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(compute_for_gene, gene, rna_adata) for gene in genes_list]
        for future in concurrent.futures.as_completed(futures):
            gene_svariance_moransI_list.append(future.result())

    data = pd.DataFrame(gene_svariance_moransI_list)
    data.columns = ['genes', 'spatial_variance', 'morans_I']

    # Sort the DataFrame by Moran's I in descending order and select the top x genes
    data_sorted = data.sort_values(by='morans_I', ascending=False)

    # Create the bar plot with flipped axes
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data_sorted['morans_I'], y=data_sorted['genes'], orientation='h'))
    fig.update_layout(
        xaxis_title="Moran's I",
        yaxis_title="Genes",
        title=f"Autocorrelation for all Genes"
    )
    helperfuncs.apply_general_plotly_layout(fig, True)
    fig.write_html(f"{figure_path}/contamination_global_morans_I.html")
    fig.write_image(f"{figure_path}/contamination_global_morans_I.png", scale=3)
    
    helperfuncs.df_to_parquet(data_sorted, 'ambient', spoqc_tmp_folder, [], 'genes')
    return data_sorted