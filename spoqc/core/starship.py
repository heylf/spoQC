import os
import sys
import pkgutil
import importlib
import numpy as np

from typing import Dict, Any, Tuple

from . import _output_structure
from . import _config
from . import _data
from . import metric
from . import prior
from . import hqr
from .. import helperfuncs
from .. import metrics
from .. import priors

class Enterpise:
    def __init__(self, kwargs):
        self.args = _config.Args(kwargs)
        self.cargo = None

        self.hqcr_metricset = None
        self.hqpr_metricset = None
        self.hqtr_metricset = None

        self.hqcr_priorset = None
        self.hqpr_priorset = None
        self.hqtr_priorset = None

        self.hqcr_set = None
        self.hqpr_set = None
        self.hqtr_set = None

        _output_structure.create_output_structure(self.args)

    def load_cargo_data(self):
        self.cargo = _data.CargoSpatialData(
            self.args.input_file,
            self.args.datatype,
            self.args.dataset,
            self.args.annotation_file,
            self.args.annotation_key,
            self.args.image_type,
            self.args.resolution,
        )
        print(self.cargo.sdata)

        # Crop data
        if self.args.crop_size > 0:
            print('[NOTE] Crop for testing')
            start = 10500
            end = self.args.crop_size
            self.cargo.sdata = self._crop_data(self.cargo.sdata, start, start, start+end, start+end+500, 'global')
            self.cargo.set_sdata_dimentsion_attributes(self.args.image_type, self.args.resolution)
            
        # Correct indexing
        self.cargo.correct_indexing(self.args.datatype)

        # Get RNA data and set raw data layer
        adata = self.cargo.sdata['table']
        adata.layers['raw'] = adata.X

        # Apply standard data processing to cargo
        self.cargo.perform_standard_data_processing(
            self.args.output_dir,
            self.args.num_variable_genes,
            self.args.span,
            self.args.check_geometries,
        )

        if self.args.step in ['all', 'unittest', 'hqpr', 'hqpr_metrices']:
            self.cargo.get_pixel_intensities_hqpr(
                self.args.tmp_dir,
                self.args.image_type,
                self.args.resolution,
                self.args.staining,
            )
            
        if self.args.step in ['all', 'unittest', 'hqtr', 'hqtr_metrices']:
            self.cargo.get_pixel_intensities_hqtr(
                self.args.output_dir,
                self.args.tmp_dir,
                self.args.overwrite,
                self.args.chunk_size,
            )


        # Some initial plots done during data loading
        self._data_loading_plots()

        print("[finish]")

        
    def _data_loading_plots(self):
        print("[NOTE] Plot some data loading plots")

        if self.args.step in ['all', 'unittest', 'hqpr', 'hqpr_metrices']:

            modality = "hqpr"
            helperfuncs.plot_pixels(
                f'{self.args.output_dir}/{modality}/{modality}_metrices/{self.args.staining}/',
                np.log10(self.cargo.xy_intensities_hqpr + 1),
                self.cargo.imagedim,
                'input_pixel_intensities',
                'input_pixel_intensities',
                'gray',
                False,
                False
            )

        if self.args.step in ['all', 'unittest', 'hqtr', 'hqtr_metrices']:

            modality = "hqtr"
            figure_path = f'{self.args.output_dir}/{modality}/{modality}_metrices/'

            # Plot transcript point plot
            helperfuncs.plot_scatter_by_category(
                self.cargo.sdata.points['transcripts'].compute(),
                None, 
                figure_path, 
                'transcript_points',
                'transcript_points',
                None,
                pointsize=0.5
            )

            helperfuncs.plot_pixels(
                figure_path,
                np.log10(self.cargo.xy_intensities_hqtr + 1),
                self.cargo.imagedim,
                'input_transcript_densities',
                'input_transcript_densities',
                'gray',
                False,
                False
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


    def _load_metricset(self, name, modality):
        print(f"[NOTE] Load metric set for {modality}")
        metricset_list = []
        metrics_module = getattr(metrics, modality)
        for module_info in pkgutil.iter_modules(metrics_module.__path__):
            module_name = module_info.name
            full_name = f"{metrics_module.__name__}.{module_name}"
            module = importlib.import_module(full_name)

            if hasattr(module, "init_metric"):
                metricset_list.append(module.init_metric(self))
                print(f"Loaded metric: {module_name}")
            else:
                print(f"WARNING: {module_name} has no init_metric() function")

        metricset = metric.MetricSet(name, metricset_list)
        print('[finish]')
        return metricset


    def load_metric_sets(self):
        self.hqcr_metricset = self._load_metricset("hqcr", "hqcr")
        self.hqpr_metricset = self._load_metricset("hqpr", "hqpr")
        self.hqtr_metricset = self._load_metricset("hqtr", "hqtr")


    def _load_priorset(self, name, modality):
        priorset_list = []
        priors_module = getattr(priors, modality)
        for module_info in pkgutil.iter_modules(priors_module.__path__):
            module_name = module_info.name
            full_name = f"{priors_module.__name__}.{module_name}"
            module = importlib.import_module(full_name)

            if hasattr(module, "init_prior"):
                priorset_list.append(module.init_prior(self))
                print(f"Loaded prior: {module_name}")
            else:
                print(f"WARNING: {module_name} has no init_prior() function")

        priorset = prior.PriorSet(name, priorset_list)
        return priorset


    def _check_prior_metric_match(self, metricset, priorset):
        metric_names = [metric.name for metric in metricset]
        for prior in priorset:
            for metric in prior.needs_metrics:
                if metric not in metric_names:
                    sys.exit(f"""
                        [ERROR] You defined a prior for a metric that is not defined. 
                        The prior for {prior.name} is missing as metric.
                    """)


    def initialize_hqcr_set(self):
        hqcr_set = hqr.HqcrSet(self)
        hqcr_set.load_cell_clustering_df(self)
        hqcr_set.load_cell_clustering_adata(self)
        self.hqcr_set = hqcr_set


    def _initialize_hqpr_set(self):
        hqpr_set = hqr.HqprSet(self)
        self.hqpr_set = hqpr_set


    def _initialize_hqtr_set(self):
        hqtr_set = hqr.HqtrSet(self)
        self.hqtr_set = hqtr_set


    def load_prior_sets(self):
        if self.args.step in ['all', 'unittest', 'hqcr_ident', 'hqcr_celltype', 'hqpr_celltype', 'hqtr_celltype']:
            self.initialize_hqcr_set()
            self.hqcr_priorset = self._load_priorset("hqcr", "hqcr")
            self._check_prior_metric_match(self.hqcr_metricset.metricset, self.hqcr_priorset.priorset)
        if self.args.step in ['all', 'unittest', 'hqpr', 'hqpr_metrices']:
            self._initialize_hqpr_set()
            self.hqpr_priorset = self._load_priorset("hqpr", "hqpr")
            self._check_prior_metric_match(self.hqpr_metricset.metricset, self.hqpr_priorset.priorset)
        if self.args.step in ['all', 'unittest', 'hqtr', 'hqtr_metrices']:
            self._initialize_hqtr_set()
            self.hqtr_priorset = self._load_priorset("hqtr", "hqtr")
            self._check_prior_metric_match(self.hqtr_metricset.metricset, self.hqtr_priorset.priorset)




