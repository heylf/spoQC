import scanpy as sc
import pandas as pd

from . import additional_analysis

def process_sdata(dataset, sdata):

    if ( dataset in ['Xenium_FFPE_Human_Breast_Cancer_Rep1', 'Xenium_FFPE_Human_Breast_Cancer_Rep2'] ):
        print('[NOTE] Apply extra Xenium_FFPE_Human_Breast_Cancer processing')
        sdata['table'].obs.index = sdata['table'].obs['cell_id']

    if ( dataset == 'Xenium_Prime_Mouse_Brain_Coronal_FF' ):
        print('[NOTE] Apply extra Xenium_Prime_Mouse_Brain_Coronal_FF processing')
        # Cellid mapping for the transripts because somestime cellids are string and have UNASSIGNED or other keywords for
        # beeing unassigned to a cell.
        # Thus I give all cell_ids just a int ID
        # sdata['table'].obs.index = [str(i) for i in range(len(sdata['table'].obs.index))]
        # mapping = sdata['table'].obs.index.to_series().set_axis(sdata['table'].obs["cell_id"].values)
        # sdata.shapes['cell_boundaries'].index = sdata.shapes['cell_boundaries'].index.map(mapping)
        # sdata.shapes['cell_circles'].index = sdata.shapes['cell_circles'].index.map(mapping)
        # sdata.shapes['nucleus_boundaries'].index = sdata.shapes['nucleus_boundaries'].index.map(mapping)

        # # Mapping of transcript table
        # mapping = dict(zip(sdata['table'].obs["cell_id"], sdata['table'].obs.index))
        # sdata.points['transcripts']['cell_id'] = (
        #     sdata.points['transcripts']['cell_id']
        #         .map(mapping, meta=('cell_id', 'str'))
        #         .fillna('-1')
        #         .astype('str')
        # )

    if ( dataset in ['Xenium_V1_FF_Mouse_Brain_MultiSection_1', 
                     'Xenium_V1_FF_Mouse_Brain_MultiSection_2',
                     'Xenium_V1_FF_Mouse_Brain_MultiSection_3'] ):
        print('[NOTE] Apply extra Xenium_V1_FF_Mouse_Brain_MultiSection processing')
        sdata['table'].obs.index = sdata['table'].obs['cell_id']

    if ( dataset in ['Xenium_V1_hLiver_nondiseased_section_FFPE'] ):
        print('[NOTE] Apply extra Xenium_V1_hLiver_nondiseased_section_FFPE processing')
        sdata['table'].obs.index = sdata['table'].obs['cell_id']


def unsupervised_celltype_annotation(sdata, CONST, seed):
    figure_path = f'{CONST.FIGURE_PATH}/annotation/'
    rna = sdata['table']
    rna.X = rna.layers['normlog']

    nn = 20
    if ( rna.n_obs < 100 ):
        nn = 10
        n_pcs=2
    print(f"[NOTE] Using {nn} neighbours")

    sc.pp.neighbors(rna, n_neighbors=nn, n_pcs=n_pcs, random_state=seed)
    sc.tl.umap(rna, min_dist=0.1, spread=1.2, random_state=seed)
    win_res = additional_analysis.analysis_funcs.test_resolutions_leiden(
        sdata['table'],
        figure_path,
        CONST.THREADS,
        k=20,
        steps=30,
        end=2.0,
        start=0.0
    )
    sc.tl.leiden(rna, resolution=win_res, key_added='leiden', random_state=seed)

    # In case it is really bad data, I have to do something else.
    # Quite often leiden clustering finds then a resolution which is overfitting with too many subclusters.
    # What might help is then a thorough seach in the range of 0-0.1 resolution.
    if ( len(set(rna.obs['leiden'])) > 30 ):
        rna.obs.drop(columns=['leiden'], inplace=True)
        win_res = additional_analysis.analysis_funcs.test_resolutions_leiden(
            sdata['table'],
            figure_path,
            CONST.THREADS,
            k=20,
            steps=30,
            end=0.1,
            start=0.000001
        )
        sc.tl.leiden(rna, resolution=win_res, key_added='leiden', random_state=seed)

    # If that does not help then iut is assumed that all the data points are bad.
    if ( len(set(rna.obs['leiden'])) > 30 ):
        rna.obs['leiden'] = ['bad'] * rna.n_obs

    annotation_df = pd.DataFrame({
        'Barcode': list(rna.obs.index),
        'Cluster': [f'leiden_{str(x)}' for x in rna.obs['leiden']]
    })
    annotation_df.to_csv(f'{figure_path}/unsupervised_cell_annotation.tsv', sep='\t', index=False)

