import numpy as np
import pandas as pd

from spatialdata.models import PointsModel

from . import dataloaders

class CargoSpatialData:
    def __init__(self, input_path, datatype, dataset):
        if datatype == "xenium":
            print(f"[NOTE] Load {datatype} data")
            self.sdata = dataloaders.xenium.get_data_xenium(input_path, dataset)
        self._apply_monotonic_indexing()

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


    def _correct_indexing(sdata, datatype):
        if datatype == "xenium":
            print(f"[NOTE] Correct indexing for {datatype} data")
            dataloaders.xenium.correct_indexing(sdata)



