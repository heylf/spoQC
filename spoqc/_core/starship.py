import os
import sys

from typing import Dict, Any, Tuple

from . import _output_structure
from . import _config
from . import _data
from .. import process_datasets

class Enterpise:
    def __init__(self, kwargs):
        self.args = _config.Args(kwargs)
        _output_structure.create_output_structure(self.args)

    def load_cargo_data(self):
        self.cargo = _data.CargoSpatialData(
            self.args.input_file,
            self.args.datatype,
            self.args.dataset,
            self.args.annotation_file,
            self.args.annotation_key,
            self.args.image_type,
            self.args.resoltion,
        )
        print(self.cargo.sdata)

        # Crop data
        if self.args.crop_size > 0:
            print('[NOTE] Crop for testing')
            start = 10500
            end = self.args.crop_size
            self.cargo.sdata = self._crop_data(self.cargo.sdata, start, start, start+end, start+end+500, 'global')

        # Correct indexing
        self.cargo.correct_indexing(self.args.datatype)

        # Get RNA data and set raw data layer
        adata = self.cargo.sdata['table']
        adata.layers['raw'] = adata.X

        # Apply standard data processing to cargo
        self.cargo.perform_standard_data_processing(
            self.args.step,
            self.args.num_variable_genes,
            self.args.span
        )

        
    def _crop_data(
            self: Any,
            sdata: Any,
            bb_xmin: float,
            bb_ymin: float,
            bb_xmax: float,
            bb_ymax: float,
            coordsystem: str
        ) -> None:
        """
        Crop a spatial dataset to a specified bounding box within a given coordinate system.

        Parameters:
        sdata (SpatialData): The spatial dataset to crop.
        bb_xmin (float): Minimum x-coordinate of the bounding box.
        bb_ymin (float): Minimum y-coordinate of the bounding box.
        bb_xmax (float): Maximum x-coordinate of the bounding box.
        bb_ymax (float): Maximum y-coordinate of the bounding box.
        coordsystem (str): The coordinate system used for cropping.

        Returns:
        Tuple[SpatialData, float, float]: A tuple containing:
            - The cropped spatial dataset.
            - The minimum x-coordinate of the bounding box.
            - The minimum y-coordinate of the bounding box.
        """
        sdata_filtered_cs = sdata.filter_by_coordinate_system(coordsystem)

        cropped_sdata = None
        try:
            cropped_sdata = sdata_filtered_cs.query.bounding_box(
                axes=["x", "y"],
                min_coordinate=[bb_xmin, bb_ymin],
                max_coordinate=[bb_xmax, bb_ymax],
                target_coordinate_system=coordsystem,
            )
        except: 
            # This erorr sometimes happen - ValueError: Number of partitions do not match (1 != 8)
            sys.exit(f"""
                [Error] Cropping failed with {bb_xmin}, {bb_ymin}, {bb_xmax}, {bb_ymax}.
                Please check the coordinates and try again.
                """
            )

        if ( 'table' in cropped_sdata._shared_keys ):
            # This has to be done because else those levels have different cell_ids captures.
            # I think this happends because the cropping does not capture polygons on the cropping border.
            ids = cropped_sdata['table'].obs.index
            ids = ids.astype(type(sdata['cell_boundaries'].index[0])).tolist()
                
            for id in ids:
                if ( id not in sdata['cell_boundaries'].index ):
                    sys.exit(f"""
                        [Error] Please check your indexing of sdata['table'].obs.index
                        and sdata['cell_boundaries'].index the index {id} is not in the latter index.
                        """
                    )
            cropped_sdata['cell_boundaries'] = sdata['cell_boundaries'].loc[ids]
            cropped_sdata['nucleus_boundaries'] = sdata['nucleus_boundaries'].loc[ids]
            return cropped_sdata
        else:
            sys.exit("[NOTE] No table in sdata so returning None")


    def generate_unsupervised_annotation(self):
        if self.args.step in ['annotation']:
            print(f'[NOTE] Perform unsuperivsed cell annotation')
            self.cargo.celltype_annotation.perform_unsupervised_celltype_annotation(self.cargo.sdata, self.args)
        print("[finish]")

