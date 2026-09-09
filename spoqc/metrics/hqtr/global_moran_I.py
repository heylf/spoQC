import numpy as np
import pandas as pd
import plotly.graph_objects as go
import geopandas as gpd
import scipy.sparse as sp

from libpysal.weights import Queen

from ... import helperfuncs

# Vectorized Moran's I for all genes at once, given a shared weights matrix.
# Degenerate genes (zero variance) are filled with NaN, matching what
# ac_image.py expects when it zeroes out bad genes via np.isnan(...).
def moran_I_all_genes(X_dense: np.ndarray, weights) -> np.ndarray:
    n = X_dense.shape[0]
    S0 = weights.sum()

    z = X_dense - X_dense.mean(axis=0, keepdims=True)
    z_weights = weights @ z

    num = np.einsum("ij,ij->j", z, z_weights)
    den = np.einsum("ij,ij->j", z, z)

    morans_I = np.full(X_dense.shape[1], np.nan, dtype=np.float64)
    ok = den > 0
    morans_I[ok] = (n / S0) * (num[ok] / den[ok])
    return morans_I


def calculate_global_moran_I_values(sdata, figure_path, spoqc_tmp_folder):

    rna_adata = sdata['table']

    genes_list = np.array(rna_adata.var_names)

    coords = rna_adata.obsm['spatial']
    gdf = gpd.GeoDataFrame({'x': coords[:, 0], 'y': coords[:, 1]},
                            geometry=gpd.points_from_xy(coords[:, 0], coords[:, 1]))

    # Create spatial-neighbor weights using queen contiguity, once for all genes
    # (coordinates, and therefore the weights matrix, don't depend on the gene).
    w = Queen.from_dataframe(gdf)
    w.transform = "r"
    weights = w.sparse

    X = rna_adata.X
    X_dense = X.toarray() if sp.issparse(X) else np.asarray(X)

    morans_I = moran_I_all_genes(X_dense, weights)

    data = pd.DataFrame({'genes': genes_list, 'morans_I': morans_I})

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
    fig.write_image(f"{figure_path}/contamination_global_morans_I.pdf", scale=3)

    helperfuncs.df_to_parquet(data_sorted, 'ambient', spoqc_tmp_folder, [], 'genes')
    return data_sorted
