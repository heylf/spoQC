import spatialdata as sd
import pandas as pd

def _process_dataset(dataset, sdata):

    if ( dataset in ['Xenium_FFPE_Human_Breast_Cancer_Rep1', 'Xenium_FFPE_Human_Breast_Cancer_Rep2'] ):
        print('[NOTE] Apply extra Xenium_FFPE_Human_Breast_Cancer processing')
        sdata['table'].obs.index = sdata['table'].obs['cell_id']

    if ( dataset in ['Xenium_V1_FF_Mouse_Brain_MultiSection_1', 
                     'Xenium_V1_FF_Mouse_Brain_MultiSection_2',
                     'Xenium_V1_FF_Mouse_Brain_MultiSection_3'] ):
        print('[NOTE] Apply extra Xenium_V1_FF_Mouse_Brain_MultiSection processing')
        sdata['table'].obs.index = sdata['table'].obs['cell_id']

    if ( dataset in ['Xenium_V1_hLiver_nondiseased_section_FFPE'] ):
        print('[NOTE] Apply extra Xenium_V1_hLiver_nondiseased_section_FFPE processing')
        sdata['table'].obs.index = sdata['table'].obs['cell_id']


def get_data_xenium(input_path, dataset=None):
    print(f'[NOTE] Load data {input_path}')
    sdata = sd.read_zarr(f"{input_path}")
    sdata['table'].obs['sample'] = ['sampleone'] * sdata['table'].n_obs
    if dataset:
        _process_dataset(dataset, sdata)
    return sdata


def correct_indexing(sdata):
    # Apply Integer indexing
    sdata['table'].obs.index = [int(i) for i in range(len(sdata['table'].obs.index))]
    mapping = sdata['table'].obs.index.to_series().set_axis(sdata['table'].obs["cell_id"].values)
    sdata.shapes['cell_boundaries'].index = sdata.shapes['cell_boundaries'].index.map(mapping)
    sdata.shapes['cell_circles'].index = sdata.shapes['cell_circles'].index.map(mapping)
    sdata.shapes['nucleus_boundaries'].index = sdata.shapes['nucleus_boundaries'].index.map(mapping)

    # Mapping of transcript table
    mapping = dict(zip(sdata['table'].obs["cell_id"], sdata['table'].obs.index))
    sdata.points['transcripts']['cell_id'] = (
        sdata.points['transcripts']['cell_id']
            .map(mapping, meta=('cell_id', int))
            .fillna(-1)
            .astype(int)
    )

    # Check for nan's in transcripts feature names
    sdata.points['transcripts']['feature_name'] = (
        sdata.points['transcripts']['feature_name']
        .astype('string')
        .fillna('NaN')
        .astype('category')
    )

    # Mapping of nucleus gemoetires
    if 'cell_id' in list(sdata.shapes['nucleus_boundaries'].columns):
        sdata.shapes['nucleus_boundaries']['cell_id'] = (
            sdata.shapes['nucleus_boundaries']['cell_id']
                .map(mapping)
                .fillna(-1)
                .astype(int)
        )

        # Check for nan's in sdata.shapes['nucleus_boundaries'].index
        if sdata.shapes['nucleus_boundaries'].index.hasnans:
            sdata.shapes['nucleus_boundaries'].index = sdata.shapes['nucleus_boundaries']['cell_id']
        
    # make index unqiue for multinulcei cells
    sdata.shapes["nucleus_boundaries"].index = pd.RangeIndex(len(sdata.shapes["nucleus_boundaries"]))

    # I need string indexes for anndata else code breaks
    sdata['table'].obs.index = sdata['table'].obs.index.astype(str)
    sdata['table'].obs.index.name = 'index'