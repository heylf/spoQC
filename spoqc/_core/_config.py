import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
import random
import numpy as np
import numba
import os

from typing import Dict

class Args:
    def __init__(self, kwargs: Dict):

        # Files or paths
        self.input_file = kwargs['input_file']
        self.output_dir = f"{kwargs['output_dir']}/report/"
        self.tmp_dir = kwargs['tmp_dir']
        self.annotation_file = kwargs['annotation_file'] if 'annotation_file' in kwargs else None
        self.reference_file = kwargs['reference_file'] if 'reference_file' in kwargs else None
        self.cellcycle_gene_file = kwargs['cellcycle_gene_file'] if 'cellcycle_gene_file' in kwargs else None

        # Options
        self.datatype = kwargs['datatype']
        self.step = kwargs['step']
        self.overwrite = kwargs['overwrite']
        self.staining = kwargs['staining']
        self.pixel_qc_chunk_size = kwargs['pixel_qc_chunk_size']
        self.kmeans_sample_size = kwargs['kmeans_sample_size']
        self.dataset = kwargs['dataset'] if 'dataset' in kwargs else None
        self.cluster_celltype = kwargs['cluster_celltype'] if 'cluster_celltype' in kwargs else None
        self.annotation_key = "celltype"
        self.canorm = True
        self.image_type = 'morphology_focus'
        self.resoltion = 'scale0'
        self.n_celltypes = 20 if kwargs['dev_test'] else None

        if kwargs['dev_test'] or kwargs['step'] == 'unittest':
            self.nthreads = 8
        else:
            self.nthreads = kwargs['nthreads']

        # Parameters
        self.thresh_prior_pixel = kwargs['thresh_prior_pixel'] if 'thresh_prior_pixel' in kwargs else None
        self.nstds_prior_pixel = kwargs['nstds_prior_pixel']
        self.doublet_prior_std = kwargs['doublet_prior_std']
        self.num_variable_genes = 5000
        self.npcs = 60
        self.span = 1.0 # Increase if you run into error like ValueError: b'There are other near singularities as well.
        self.radi = [20, 30, 40, 80, 100]
        self.seed = 123 # (!!! DO NOT CHANGE THIS SEED !!!)

        # Testing options
        self.dev_test = kwargs['dev_test']
        self.dev_report = kwargs['dev_report']

        if kwargs['dev_test'] or kwargs['step'] == 'unittest':
            self.crop_size = 2000
        else:
            self.crop_size = 0

        # Plot parameters
        self.point_size = 1

        # Init calls
        self._print_args()
        self._set_pub_style_plots()
        self._set_seed()
        self._set_env()

    def _print_args(self) -> None:
        attrs = vars(self)
        for attr_name, attr_value in attrs.items():
            print(f'Attribute Name: {attr_name}')
            print(f'Attribute Value: {attr_value}')


    def _set_pub_style_plots(self) -> None:
        print("[NOTE] Setting pubication ready plot style.")
        plt.rcParams.update({
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12
        })

        font_template = go.layout.Template(
            layout=dict(
                font=dict(size=14),
                title=dict(font=dict(size=18)),
                xaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)),
                yaxis=dict(title_font=dict(size=16), tickfont=dict(size=14)),
                legend=dict(font=dict(size=14))
            )
        )

        pio.templates["publication_fonts"] = font_template
        pio.templates.default = "plotly_white+publication_fonts"

    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        print(f"[NOTE] seed {self.seed}")


    def _set_env(sef):
        # Numba threads
        print(f"[NOTE] Setting numba threads to {sef.nthreads}")
        requested_threads = sef.nthreads
        maximum_threads = numba.config.NUMBA_NUM_THREADS
        active_threads = min(requested_threads, maximum_threads)
        print(
            f"[NOTE] Numba thread pool maximum: {maximum_threads}; "
            f"using: {active_threads}"
        )
        numba.set_num_threads(active_threads)

        # Blosc threads (for the Zarr datasets we still write)
        os.environ["BLOSC_NTHREADS"] = str(sef.nthreads)