import numpy as np
import pandas as pd
import re
import spatialdata as sd
import sys
import scanpy as sc

from spatialdata.models import PointsModel
from typing import NamedTuple

from . import dataloaders
from .. import helperfuncs
from .. import general

class CargoSpatialData:
    def __init__(
            self,
            input_path,
            datatype,
            dataset,
            annotation,
            annotation_key,
            image_type,
            resolution,
    ):

        if datatype == "xenium":
            print(f"[NOTE] Load {datatype} data")
            self.sdata = dataloaders.xenium.get_data_xenium(input_path, dataset)
        self._apply_monotonic_indexing()

        if annotation:
            self.celltype_annotation = CelltypeAnnotation(self.sdata, annotation, annotation_key)
        else:
            self.celltype_annotation = CelltypeAnnotation(self.sdata, None, annotation_key)
        self.cols_already_written = list(self.sdata['table'].obs.columns)

        self.stainings = list(self.sdata[image_type][resolution].image.c.values)
        self.imagedim = None
        self.dim_x = None
        self.dim_y = None

        self.set_sdata_dimentsion_attributes(image_type, resolution)


    def set_sdata_dimentsion_attributes(self, image_type, resolution):
        img_extent = sd.get_extent(self.sdata[image_type], coordinate_system='global')
        self.imagedim = ImageDimStruct(img_extent['x'][0], img_extent['y'][0], img_extent['x'][1], img_extent['y'][1])
        self.dim_x = len(self.sdata[image_type][resolution].image.y.values)
        self.dim_y = len(self.sdata[image_type][resolution].image.x.values)


    # Return a copy of a dask DataFrame with a globally unique, monotonically increasing RangeIndex.
    # Some readers (e.g. the Xenium zarr reader) build points partitions that each carry their own locally-scoped 
    # 0..n index, so the same index value repeats across partitions. `ddf.reset_index(drop=True)` does not fix this
    # because dask resets the index independently per partition. Here we compute the (cheap) per-partition lengths
    # and offset each partition's index by the cumulative length of the partitions before it.
    def _apply_monotonic_indexing(self):
        print("[NOTE] Apply monotonic indexing to data")

        def _assign_partition_index(df, offsets, partition_info=None):
            start = offsets[partition_info["number"]]
            df = df.copy()
            df.index = pd.RangeIndex(start, start + len(df))
            return df

        ddf = self.sdata.points['transcripts']
        lengths = ddf.map_partitions(len).compute().to_numpy()
        offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        ddf = ddf.map_partitions(_assign_partition_index, offsets, meta=ddf._meta)
        self.sdata.points['transcripts'] = PointsModel.parse(ddf)


    def correct_indexing(self, datatype):
        if datatype == "xenium":
            print(f"[NOTE] Correct indexing for {datatype} data")
            dataloaders.xenium.correct_indexing(self.sdata)


    def perform_standard_data_processing(self, step, nhvg, span):
        print(f'[NOTE] Perform mandaory steps')
        if ( step != 'generalqc' ):
            general.valid_geometries.correct_for_valid_geometries(self.sdata)

            # Sanity Check
            for obj_type in ['cell', 'nucleus']:
                geometries = np.array(self.sdata[f'{obj_type}_boundaries']['geometry'])
                for i, obj in enumerate(geometries):
                    if( not obj.is_valid ):
                        sys.exit("[ERROR] Found invalid geometries")

        general.normalizations.transform_normalize_sc_data(self.sdata, nhvg, span)
        general.normalizations.fill_nans_for_0_transcript_cells(self.sdata)
        print("[finish]")


class CelltypeAnnotation:
    def __init__(self, sdata, input_path, annotation_key):
        if input_path:
            print(f"[NOTE] Adding annotation {input_path}")

            adata = sdata['table']

            df_labels = pd.read_csv(f'{input_path}', sep=None, engine='python')
            df_labels = df_labels[['Barcode', 'Cluster']]
            df_labels.index = df_labels['Barcode']
            df_labels = df_labels.drop(columns='Barcode')
            df_labels.columns = [annotation_key]

            # Check if annotation and anndata have the same number of cells
            if adata.n_obs != len(df_labels):
                warn_text = f"""
                [WARN]: The annotation has a different number of cells {len(df_labels)} than your sdata "
                {adata.n_obs}. Please Check your annotation.
                """
                print(warn_text)
            
            # I have to map here if that is the case.
            if "cell_id" in adata.obs.columns:
                mapping = dict(zip(adata.obs["cell_id"], adata.obs.index))
                if df_labels.index[0] in list(adata.obs["cell_id"]) and df_labels.index[0] not in list(adata.obs.index):
                    df_labels.index = df_labels.index.map(mapping)
            
            if ( type(adata.obs.index[0]) == str ):
                df_labels.index = df_labels.index.map(str)

            # Sometimes annoation does not contain all cells.
            adata.obs = adata.obs.join(df_labels[annotation_key], how='left')
            adata.obs[annotation_key] = adata.obs[annotation_key].fillna('unkown')
            
            # Clean up celltype names, else you will always run in potential code breaks.
            adata.obs[annotation_key] = [re.sub(r'[^A-Za-z0-9]', '', x) for x in adata.obs[annotation_key]]

            # Get and order celltypes
            celltypes = list(set(adata.obs[annotation_key]))
            celltypes.sort()

            # Assign attributes
            self.ncelltypes = len(set(adata.obs[annotation_key]))
            self.annotation_key = annotation_key
            self.celltypes = celltypes
            self.colors = helperfuncs.generate_distinct_colors(self.ncelltypes)
        else:
            self.annotation_key = annotation_key
            self.ncelltypes = None
            self.celltypes = None
            self.colors = None


    def perform_unsupervised_celltype_annotation(self, sdata, args):
        figure_path = f'{args.output_dir}/annotation/'
        rna = sdata['table']
        rna.X = rna.layers['normlog']

        nn = 20
        n_pcs = None
        if ( rna.n_obs < 100 ):
            nn = 10
            n_pcs=2
        print(f"[NOTE] Using {nn} neighbours")

        sc.pp.neighbors(rna, n_neighbors=nn, n_pcs=n_pcs, random_state=args.seed)
        sc.tl.umap(rna, min_dist=0.1, spread=1.2, random_state=args.seed)
        win_res = helperfuncs.test_resolutions_leiden(
            sdata['table'],
            figure_path,
            args.nthreads,
            k=20,
            steps=30,
            end=2.0,
            start=0.0
        )
        sc.tl.leiden(rna, resolution=win_res, key_added='leiden', random_state=args.seed)

        # In case it is really bad data, I have to do something else.
        # Quite often leiden clustering finds then a resolution which is overfitting with too many subclusters.
        # What might help is then a thorough seach in the range of 0-0.1 resolution.
        if ( len(set(rna.obs['leiden'])) > 30 ):
            rna.obs.drop(columns=['leiden'], inplace=True)
            win_res = helperfuncs.test_resolutions_leiden(
                sdata['table'],
                figure_path,
                args.nthreads,
                k=20,
                steps=30,
                end=0.1,
                start=0.000001
            )
            sc.tl.leiden(rna, resolution=win_res, key_added='leiden', random_state=args.seed)

        # If that does not help then iut is assumed that all the data points are bad.
        if ( len(set(rna.obs['leiden'])) > 30 ):
            rna.obs['leiden'] = ['bad'] * rna.n_obs

        annotation_df = pd.DataFrame({
            'Barcode': list(rna.obs.index),
            'Cluster': [f'leiden_{str(x)}' for x in rna.obs['leiden']]
        })
        annotation_df.to_csv(f'{figure_path}/unsupervised_cell_annotation.tsv', sep='\t', index=False)



class ImageDimStruct(NamedTuple):
    bb_xmin: int
    bb_ymin: int
    bb_xmax: int
    bb_ymax: int