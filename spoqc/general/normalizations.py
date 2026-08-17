import scanpy as sc
import anndata as ad

from typing import Any

def transform_normalize_sc_data(sdata, n_variable_genes, span):
    rna_adata = sdata['table']
    sc.pp.normalize_total(rna_adata)
    sc.pp.log1p(rna_adata)
    rna_adata.layers['normlog'] = rna_adata.X.copy()

    try:
        sc.pp.highly_variable_genes(rna_adata, flavor="seurat_v3", 
                                    batch_key="sample", span=span, n_top_genes=n_variable_genes)
    except Exception as e:
        print(f"[WARN] Could not perform HVG analysis because of: {e}")

    sc.pp.scale(rna_adata, zero_center=True)  # Scale data to unit variance and zero mean
    rna_adata.layers['normlogscale'] = rna_adata.X.copy()
    rna_adata.X = rna_adata.layers['raw'] # raw = counts


def cell_area_normalization(sdata):
    for x in ['transcript_counts', 'n_genes_by_counts']:
        sdata['table'].obs[f'canorm_{x}'] = sdata['table'].obs[f'{x}'] / sdata['table'].obs['cell_area']

# The .obs table has NaNs for cells that have 0 transcript counts.
def fill_nans_for_0_transcript_cells(data: Any): # anndata or spatialdata
    if ( isinstance(data, ad.AnnData) ):
        cols = data.obs.columns
        for col in cols:
            if ( type(data.obs[col][0]) != str ):
                data.obs[col] = data.obs[col].fillna(0)
    else:
        cols = data['table'].obs.columns
        for col in cols:
            if ( type(data['table'].obs[col][0]) != str ):
                data['table'].obs[col] = data['table'].obs[col].fillna(0)