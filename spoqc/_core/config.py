from typing import Dict

class Args:
    def __init__(self, kwargs: Dict):

        # Files or paths
        self.input_file = kwargs['input_file']
        self.output_dir = f"{kwargs['output_dir']}/report/"
        self.tmp_dir = kwargs['tmp_dir']
        self.annotation_file = kwargs['annotation_file']
        if 'reference_file' in kwargs:
            self.reference_file = kwargs['reference_file']
        if 'cellcycle_gene_file' in kwargs:
            self.cellcycle_gene_file = kwargs['cellcycle_gene_file']

        # Options
        self.step = kwargs['step'] 
        self.overwrite = kwargs['overwrite']
        self.staining = kwargs['staining']
        self.pixel_qc_chunk_size = kwargs['pixel_qc_chunk_size']
        self.kmeans_sample_size = kwargs['kmeans_sample_size']
        if 'dataset' in kwargs:
            self.dataset = kwargs['dataset']
        if 'cluster_celltype' in kwargs:
            self.cluster_celltype = kwargs['cluster_celltype']
        self.annotation_key = "celltype"
        self.canorm = True
        self.image_type = 'morphology_focus'
        self.resoltion = 'scale0'

        if kwargs['dev_test']:
            self.n_celltypes = 20
        else:
            self.n_celltypes = None

        if kwargs['dev_test'] or kwargs['step'] == 'unittest':
            self.nthreads = 8
        else:
            self.nthreads = kwargs['nthreads']

        # Parameters
        if 'thresh_prior_pixel' in kwargs:
            self.thresh_prior_pixel = kwargs['thresh_prior_pixel']
        self.nstds_prior_pixel = kwargs['nstds_prior_pixel']
        self.doublet_prior_std = kwargs['doublet_prior_std']
        self.variable_genes = 5000
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


    def print_args(self) -> None:
        attrs = vars(self)
        for attr_name, attr_value in attrs.items():
            print(f'Attribute Name: {attr_name}')
            print(f'Attribute Value: {attr_value}')




