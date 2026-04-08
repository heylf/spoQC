#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

# TODO create under helperfunc a general plot layout
# TODO check the orientation of the plots again (if it makes sense to invert y axis).
# TODO check topology methods for more stuff.
# TODO maybe for all spatial plots remove x and y axsis and just call the labels spatialdim1, spatialdim2, spatialdim3 ...
# TODO remove packages that are not used anymroe
# TODO put every metric into their own script, so it is easier for people to find them and add new metrics
# TODO why do I have negative values for canorm_transcript counts?
# TODO change x and y axis for the correct once from the coordinate system. For all plots.

# In[]

import sys

# Utility imports
import os
import random
import shutil
import argparse
import importlib
import numpy as np
import pandas as pd
import re

# Tool imports
import spatialdata as sd
import spatialdata_plot
import matplotlib.pyplot as plt
import scanpy as sc

# Tool froms
from spatialdata_io import xenium
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
from scipy.stats import median_abs_deviation

# Own scripts
from spoqc import general
from spoqc import whole_slide
from spoqc import marker
from spoqc import model_preparation
from spoqc import cell_analysis
from spoqc import multiplet
from spoqc import hqr
from spoqc import void
from spoqc import helperfuncs
from spoqc import ambient
from spoqc import process_datasets
from spoqc import folder_structure
from spoqc import additional_analysis
from spoqc import plot_config

def build_parser() -> argparse.ArgumentParser:

    print("[NOTE] Use arguments")

    tool_description = """
    """

    # parse command line arguments
    parser = argparse.ArgumentParser(description=tool_description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # version
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.1.0")

    # mandatory
    parser.add_argument(
        "-i", "--input",
        dest="input",
        type=str, 
        help="Path to the input directory containing Xenium data.",
        required=True
    )
    parser.add_argument(
        "-o", "--output",
        dest="output",
        type=str, 
        help="Path to the output directory containing the report.",
        required=True
    )
    parser.add_argument(
        "-t",
        dest="tmp",
        type=str, 
        help="Path to the tmp directory where spoQC saves tmp files.",
        required=True
    )

    # optional
    parser.add_argument(
        "-n", "--threads",
        dest="threads",
        type=int,
        default=1, 
        help="Number of cores to be used.",
        required=False
    )
    parser.add_argument(
        "-a", "--annotation",
        dest="annotation",
        type=str,
        help="Path to the annotation file.",
        required=False
    )
    parser.add_argument(
        "--reference",
        dest="reference",
        type=str,
        help="Path to a transcript reference file for the transcript QC.",
        required=False
    )
    parser.add_argument(
        "--cellcycle_gene_file",
        dest="cellcycle_gene_file",
        type=str,
        help="Path to a file containing gene names for the cell cycles.",
        required=False
    )
    parser.add_argument(    
        "-s", "--step",
        dest="step",
        type=str,
        default="all",
        help="Steps to run for QC.",
        required=False
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action='store_true',
        help="Overwriting temporary files.",
        required=False
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
        help="This is used for to apply standardization to spatial data for the data used in the publication.",
        required=False
    )
    parser.add_argument(
        "--staining",
        dest="staining",
        type=str,
        help="Number of cores to be used.",
        default='0',
        required=False
    )
    parser.add_argument(
        "--thresh_prior_pixel",
        dest="thresh_prior_pixel",
        type=float,
        default=None,
        help="You can set a prior threshold for the pixel prior distribution. Please read the documentation to understand what this threshold does before you set it.",
        required=False
    )
    parser.add_argument(
        "--nstds_prior_pixel",
        dest="nstds_prior_pixel",
        type=float,
        default=None,
        help="You can set the number of stds for the pixel prior distribution. Please read the documentation to understand what this does before you set it.",
        required=False
    )
    parser.add_argument(
        "--cluster_celltype",
        dest="cluster_celltype",
        type=str,
        default=None,
        help="Name of the cluster cell type you want to specifically analyse.",
        required=False
    )
    parser.add_argument(
        "--dev_test",
        dest="dev_test",
        action="store_true",
        help="This is just for developing and testing the tool.",
        required=False
    )

    return parser

# In[]
def main(argv: list[str] | None = None) -> None:
    print("[START]")

# In[]

    # Setting matplot styles
    plot_config.set_pub_style()

    parser = build_parser()
    args_ns = parser.parse_args(argv)
    args = vars(args_ns)

    print(f"[NOTE] Turn on mode testing: {args['dev_test']}")

    def constant(f):
        def fset(self, value):
            raise TypeError('your are not allowed to change constant values')
        def fget(self):
            return f()
        return property(fget, fset)

    class _Const(object):
        @constant
        def TESTING(): # set to > 100 to turn on testing case
            if args['dev_test'] or args['step'] == 'unittest' :
                return 2000
            else:
                return 0
        @constant
        def THREADS():
            if args['dev_test'] or args['step'] == 'unittest':
                return 8
            else:
                return int(args['threads'])
        @constant
        def OVERWRITE():
            return args['overwrite']
        @constant
        def INPUT_PATH():
            return args['input']
        @constant
        def FIGURE_PATH():
            return f"{args['output']}/report/"
        @constant
        def TMP_PATH():
            return f"{args['tmp']}"
        @constant
        def DATASET():
            return args['dataset']
        @constant
        def TRANSCRIPT_REFERENCE():
            return f"{args['reference']}"
        @constant
        def CELLCYCLE_GENE_FILE():
            return f"{args['cellcycle_gene_file']}"
        @constant
        def ANNOTATION_FILE():
            return args['annotation']
        @constant
        def VARIABLE_GENES():
            return 5000
        @constant
        def nPCs():
            return 60
        @constant
        def SPAN():
            # scanpy.pp.highly_variable_genes
            # span : Optional[float] (default: 0.3)
            # Increase if you run into error like ValueError: b'There are other near singularities as well. 0.031008'
            # The fraction of the data (cells) used when estimating the variance in the loess model fit if 
            # flavor='seurat_v3'.
            return 1.0
        @constant
        def ANNOTATION_KEY():
            return 'celltype'
        @constant
        def RADI():
            return [20, 30, 40, 80, 100]
        @constant    # threshold for total UMIs
        def THRESH_UMI():   
            return 100
        @constant    # threshold number of genes 
        def THRESH_N_GENES():   
            return 20
        @constant   # threshold p-value for unimodality of the UMI count and number of genes
        def THESH_UNIMODALITY():
            return 0.01
        @constant
        def STEP():
            return args['step']
        @constant   # threshold p-value for unimodality of the UMI count and number of genes
        def CANORM(): # turn on to select cell area normlaized counts for HQCR
            return True
        @constant
        def N_CELLTYPES():
            if CONST.TESTING == 0:
                return None
            else:
                return 20
        @constant
        def POINT_SIZE():       
            return 1
        @constant
        def IMAGE_TYPE():
            return 'morphology_focus'
        @constant
        def RESOLUTION():
            return 'scale0'
        @constant
        def STAINING():
            return args['staining']
        @constant
        def THRESHOLD_PRIOR_PIXEL():
            return args['thresh_prior_pixel']
        @constant
        def NSTDS_PRIOR_PIXEL():
            return args['nstds_prior_pixel']
        @constant
        def CLUSTER_CELLTYPE():
            return args['cluster_celltype']

    # Initialize constant variables
    CONST = _Const()
    constant_members = [attr_name for attr_name in vars(_Const) if isinstance(getattr(_Const, attr_name), property)]
    for attr_name in constant_members:
        attr_value = getattr(_Const, attr_name).fget(_Const)
        print(f'Attribute Name: {attr_name}')
        print(f'Attribute Value: {attr_value}')

    # Seeds (!!! DO NOT CHANGE THIS SEED !!!)
    seed=123
    random.seed(seed)
    np.random.seed(seed)
    print(f"[NOTE] seed {seed}")

    # ---------------- Environment ----------------
    # Numba threads
    os.environ["NUMBA_NUM_THREADS"] = str(CONST.THREADS)
    # Blosc threads (for the Zarr datasets we still write)
    os.environ["BLOSC_NTHREADS"] = str(CONST.THREADS)
    # Timer
    timer = helperfuncs.Timer()

    # ---------------- Folder Structure ------------
    importlib.reload(folder_structure)
    folder_structure.create_folder_structure(CONST)

    # In[]
    #######################
    ###### LOAD DATA ######
    #######################
    input_path = f'{CONST.INPUT_PATH}'
    print(f'[NOTE] Load data {input_path}')
    sdata = sd.read_zarr(f"{input_path}")
    sdata['table'].obs['sample'] = ['sampleone'] * sdata['table'].n_obs
    if ( CONST.DATASET ):
        process_datasets.process_sdata(CONST.DATASET, sdata)
    print(sdata)

    # In[]
    # Cropping for testing
    if ( CONST.TESTING > 0 ):
        print('[NOTE] Cropping for testing')
        importlib.reload(helperfuncs)
        start = 10500
        end = CONST.TESTING
        cropped_sdata, _, _ = helperfuncs.image_crop(sdata, start, start, start+end, start+end+500, 'global')
        sdata = cropped_sdata

    # In[]
    # Apply Integer indexing
    sdata.table.obs.index = [int(i) for i in range(len(sdata.table.obs.index))]
    mapping = sdata.table.obs.index.to_series().set_axis(sdata.table.obs["cell_id"].values)
    sdata.shapes['cell_boundaries'].index = sdata.shapes['cell_boundaries'].index.map(mapping)
    sdata.shapes['cell_circles'].index = sdata.shapes['cell_circles'].index.map(mapping)
    sdata.shapes['nucleus_boundaries'].index = sdata.shapes['nucleus_boundaries'].index.map(mapping)

    # Mapping of transcript table
    mapping = dict(zip(sdata.table.obs["cell_id"], sdata.table.obs.index))
    sdata.points['transcripts']['cell_id'] = (
        sdata.points['transcripts']['cell_id']
            .map(mapping, meta=('cell_id', int))
            .fillna(-1)
            .astype(int)
    )

    # TODO need a plot for that
    # Check for nan's in transcripts feature names
    transcripts = sdata.points['transcripts'].assign(
        feature_name=sdata.points['transcripts']['feature_name'].astype('string').fillna('NaN').astype('category')
    )
    sdata.points['transcripts'] = transcripts

    # I need string indexes for anndata else code breaks
    sdata.table.obs.index = sdata.table.obs.index.astype(str)
    sdata['table'].obs.index.name = 'index'

    # In[]
    # Get RNA data and set raw data layer
    rna_adata = sdata.tables["table"]
    rna_adata.layers['raw'] = rna_adata.X

    # Add annotation
    # TODO check adding the annotation again. I found during hitchhikersguide a potential error.
    annotation = helperfuncs.AnnotationStruct(0, [""])
    if ( CONST.ANNOTATION_FILE ):
        print(f"[NOTE] Adding annotation {CONST.ANNOTATION_FILE}")
        df_labels = pd.read_csv(f'{CONST.ANNOTATION_FILE}', sep=None, engine='python')
        df_labels.index = df_labels['Barcode']
        df_labels = df_labels.drop(columns='Barcode')

        if ( type(rna_adata.obs.index[0]) == str ):
            rna_adata.obs[CONST.ANNOTATION_KEY] = list(df_labels.iloc[rna_adata.obs.index]['Cluster'])
        else:
            rna_adata.obs[CONST.ANNOTATION_KEY] = list(df_labels.loc[rna_adata.obs.index]['Cluster'])

        # Clean up celltype names, else you will always run in potential code breaks.
        rna_adata.obs[CONST.ANNOTATION_KEY] = [re.sub(r'[^A-Za-z0-9]', '', x) for x in rna_adata.obs[CONST.ANNOTATION_KEY]]

        # Save number of celltypes and the celltypes names.
        annotation = helperfuncs.AnnotationStruct(len(set(rna_adata.obs[CONST.ANNOTATION_KEY])),
                                                list(set(rna_adata.obs[CONST.ANNOTATION_KEY])))

    # General variables from data
    obs_columns = list(sdata['table'].obs.columns)
    y_list = []
    x_list = []
    for i in sdata.images[CONST.IMAGE_TYPE]:
        y_list.append(sdata.images[CONST.IMAGE_TYPE][i]['image'].shape[1])
        x_list.append(sdata.images[CONST.IMAGE_TYPE][i]['image'].shape[2])
    img_extent = sd.get_extent(sdata[CONST.IMAGE_TYPE], coordinate_system='global')
    imagedim = helperfuncs.ImageDimStruct(img_extent['x'][0], img_extent['y'][0], img_extent['x'][1], img_extent['y'][1])
    dim_x = len(sdata[CONST.IMAGE_TYPE][CONST.RESOLUTION].image.y.values)
    dim_y = len(sdata[CONST.IMAGE_TYPE][CONST.RESOLUTION].image.x.values)
    stainings = list(sdata[CONST.IMAGE_TYPE][CONST.RESOLUTION].image.c.values)

    # This file is useful to later figure out which folder stands for which staining.
    # Staining names can be weird and would disrupt the code, thus I have to use the indices.
    staining_log = open(f'{CONST.FIGURE_PATH}/staining_log.txt', 'w')
    for i, staining in enumerate(stainings):
        staining_log.write(f'{i} = {staining} \n')
    staining_log.close()

    print("[finish]")

    # In[]
    ############################
    ###### ALWAYS PERFORM ######
    ############################
    print(f'[NOTE] Perform mandaory steps')
    if ( CONST.STEP != 'generalqc' ):
        general.valid_geometries.correct_for_valid_geometries(sdata)

        # Sanity Check
        for obj_type in ['cell', 'nucleus']:
            geometries = np.array(sdata[f'{obj_type}_boundaries']['geometry'])
            for i, obj in enumerate(geometries):
                if( not obj.is_valid ):
                    sys.exit("[ERROR] Found invalid geometries")

    general.normalizations.transform_normalize_sc_data(sdata, CONST.VARIABLE_GENES, CONST.SPAN)
    general.normalizations.fill_nans_for_0_transcript_cells(sdata)
    print("[finish]")

    # In[]
    ########################
    ###### ANNOTATION ######
    ########################
    print(f'[NOTE] Perform annotation')
    if ( CONST.STEP in ['annotation'] ):
        process_datasets.unsupervised_celltype_annotation(sdata, CONST, seed)
    print("[finish]")

    # In[]
    importlib.reload(helperfuncs)
    importlib.reload(general.qc_adata)
    # Low resources and quick
    ########################
    ###### GENERAL QC ######
    ########################
    if ( CONST.STEP in ['all', 'unittest', 'generalqc'] ):
        print('[NOTE] General QC')
        figure_path = f'{CONST.FIGURE_PATH}/generalqc/'

        helperfuncs.plot_original_image_cell_circles(sdata, figure_path, '1')

        general.qc_adata.quick_viz_images(figure_path, list(sdata.images), sdata)
        general.qc_adata.rawqc(sdata, figure_path, CONST.ANNOTATION_FILE, CONST.ANNOTATION_KEY)
        general.normalizations.cell_area_normalization(sdata)
        general.valid_geometries.check_for_valid_geometries(sdata, figure_path)

        print("[NOTE] Write results")
        obs_columns = helperfuncs.sdata_obs_to_parquet(sdata, figure_path, CONST.TMP_PATH, 'hqcr', obs_columns)
        print("[finish]")

    # In[]
    # Low resources and quick
    ############################
    ###### WHOLE SLIDE QC ######
    ############################
    if ( CONST.STEP in ['all', 'whole_slide_qc'] ):
        print('[NOTE] Domain QC')
        figure_path = f'{CONST.FIGURE_PATH}/whole_slide_qc/'

        ax = sdata.pl.render_images(CONST.IMAGE_TYPE).pl.show(
            title='',
            frameon=False, 
            return_ax=True,
            pad_extent=0,
            dpi=300
        )   
        ax.axis('off')
        ax.invert_yaxis()
        plt.savefig(f'{figure_path}/input_domain_thickness_analysis.png', bbox_inches='tight')
        plt.close()

        whole_slide.whole_slide_metrices.measure_stripe_thickness_and_black_area(
            f'{figure_path}/input_domain_thickness_analysis.png',
            np.array([68,1,84]),
            f'{figure_path}'
        )
        print("[finish]")

    # In[]
    # Low resources and quick
    #######################
    ###### BUBBLE QC ######
    #######################
    if ( CONST.STEP in ['all', 'unittest', 'bubbleqc'] ):
        figure_path = f'{CONST.FIGURE_PATH}/bubbleqc/'
        cell_analysis.qc_bubble.bubbleqc(sdata, figure_path, 'cell_boundaries')

        print("[NOTE] Write results")
        obs_columns = helperfuncs.sdata_obs_to_parquet(sdata, figure_path, CONST.TMP_PATH, 'hqcr', obs_columns)
        print("[finish]")

    # In[]
    ########################
    ###### DOUBLET QC ######
    ########################
    # High resources and slow (takes 18-19 hours for a full dataset)
    if ( CONST.STEP in ['all', 'unittest', 'doubletqc'] ):
        figure_path = f'{CONST.FIGURE_PATH}/doubletqc/'
        mean_diameter = np.mean(sdata['cell_circles']['radius'])*2

        print(f"[NOTE] Estimated cell diameter is {mean_diameter}")

        # TODO maybe introduce also leiden clustering for estimation
        ncelltypes = -1
        if ( CONST.ANNOTATION_FILE and CONST.N_CELLTYPES == None ):
            ncelltypes = annotation.ncelltypes
        else:
            ncelltypes = CONST.N_CELLTYPES

        timer.start()
        print(f"[NOTE] Doublet QC with {annotation.ncelltypes} estimated celltypes")
        multiplet.qc_multiplet.doubletqc(
            sdata,
            figure_path,
            CONST.TMP_PATH,
            'transcripts',
            ncelltypes,
            mean_diameter,
            3,
            2,
            3,
            [10, 60],
            1,
            10
        )
        timer.stop()

        print("[NOTE] Calculate overlap areas")
        timer.start()
        multiplet.qc_multiplet.calculate_overlap_areas(sdata)
        timer.stop()

        print("[NOTE] Write results")
        obs_columns = helperfuncs.sdata_obs_to_parquet(sdata, figure_path, CONST.TMP_PATH, 'hqcr', obs_columns)
        print("[finish]")

    # In[]
    # Low resource but long (takes 4-5 hours)
    #####################
    ###### VOID QC ######
    #####################
    importlib.reload(void.qc_void)
    importlib.reload(helperfuncs)
    if ( CONST.STEP in ['all', 'unittest', 'voidqc'] ):
        print("[NOTE] Void QC")
        figure_path = f'{CONST.FIGURE_PATH}/voidqc/'
        # You can also provide contaminants with:
        # void.qc_void.voidqc(sdata, figure_path, CONST.TMP_PATH, 30, ['CD3D', 'CD14', 'CD68'], CONST.THREADS)
        void.qc_void.voidqc(sdata, figure_path, CONST.TMP_PATH, 30, [], CONST.THREADS)

        print("[NOTE] Write results")
        obs_columns = helperfuncs.sdata_obs_to_parquet(sdata, figure_path, CONST.TMP_PATH, 'hqcr', obs_columns)
        print("[finish]")

        #TODO activate this again if you mangage to speed up the counting for all transcripts
        # helperfuncs.plot_scatter_density(
        #     sdata['table'], figure_path, None, 
        #     1, None, 'convexhull_all_transcripts', None, 'Density of convexhull'
        # )

        # TODO Check if border score differe between inner domain borders and outer slide edges borders.
        # TODO does not work correctly yet. The quality clusters sometiems just go to different cell islands.
        # I need to define bacgkround distribution first and define the bad quality cluster on that.

    # In[]
    #####################
    ###### CELL QC ######
    #####################
    importlib.reload(general.qc_transcript)
    importlib.reload(helperfuncs)

    # Low resources and quicks for full dataset (40-50 min)
    if ( CONST.STEP in ['all', 'unittest', 'cellqc'] ):
        figure_path = f'{CONST.FIGURE_PATH}/cellqc/'

        print("[NOTE] Convexity QC")
        timer.start()
        cell_analysis.cell_metrices.convexityqc(sdata, figure_path)
        timer.stop()

        print("[NOTE] Multinuclei QC")
        # TODO check this again in contradicts the convexity analysis
        timer.start()
        cell_analysis.cell_metrices.multi_nuceli_qc(sdata, figure_path)
        timer.stop()

        print("[NOTE] Border cell inspection")
        #TODO combine island score and border score to really just pick border cells and not smalle cell islands.
        timer.start()
        cell_analysis.cell_metrices.define_border_cells(sdata, figure_path, 1.0, 50, 10, CONST.THREADS)
        timer.stop()

        print("[NOTE] Cell island inspection")
        timer.start()
        hqr.hqcr.islandqc(sdata, figure_path, 15, 10)
        timer.stop()

        print("[NOTE] Low transcript quality inspection")
        timer.start()
        transcript_df = sdata['transcripts'].compute()
        # at 10x Genomics they use a threshold of qv < 20 (see 10x Baysor tutorial)
        general.qc_transcript.get_low_qc_transcript_count(transcript_df, sdata, 20, figure_path)
        timer.stop()

        print("[NOTE] Write results")
        obs_columns = helperfuncs.sdata_obs_to_parquet(sdata, figure_path, CONST.TMP_PATH, 'hqcr', obs_columns)
        print("[finish]")


    # In[]
    ##################
    ###### HQCR ######
    ##################
    importlib.reload(helperfuncs)
    importlib.reload(hqr.hqcr)
    importlib.reload(hqr.markov_random_field_zarr_parallel)
    # Low resources and for a full dataset it takes 30 - 40 min.
    if ( CONST.STEP in ['all', 'unittest', 'hqcr_ident'] ):
        cell_df, qc_metrices = hqr.hqcr.start_hqcr(sdata, CONST.TMP_PATH, imagedim, CONST, seed)
        hqr.hqcr.plots_hqcr(sdata, CONST.FIGURE_PATH, cell_df, qc_metrices)
        print("[finish]")

    # In[]
    importlib.reload(hqr.hqcr)
    # Low resources and quick.
    if ( CONST.STEP in ['all', 'hqcr_celltype'] ):
        if ( CONST.ANNOTATION_FILE ):
            hqr.hqcr.start_hqcr_celltype(sdata, CONST.TMP_PATH, imagedim, CONST)
            print("[finish]")
        else:
            print("[NOTE] No annotation file provided so I will not perform start_hqcr_celltype")

    # In[]
    # TODO I have not taking the z axis varability into account.
    ##################
    ###### HQPR ######
    ##################
    hqr.hqpr.get_hqpr(
        sdata,
        CONST.TMP_PATH,
        imagedim,
        dim_x,
        dim_y,
        CONST,
        seed,
        thresh_p=CONST.THRESHOLD_PRIOR_PIXEL,
        nstds_p=CONST.NSTDS_PRIOR_PIXEL
    )

    # In[]
    if ( CONST.ANNOTATION_FILE ):
        hqr.hqpr.celltype_refinement_of_hqpr(sdata, CONST.TMP_PATH, imagedim, dim_x, dim_y, CONST)
    else:
        print("[NOTE] No annotation file provided so I will not perform celltype_refinement_of_hqpr")

    # In[]
    #####################
    ###### AMBIENT ######
    #####################

    if ( CONST.STEP in ['all', 'hqtr', 'unittest', 'ambientqc'] ):
        figure_path = f'{CONST.FIGURE_PATH}/ambientqc/'
        _ = ambient.qc_ambient.start_qc_ambient(sdata, figure_path, CONST.TMP_PATH, CONST.THREADS)

    # In[]
    ##################
    ###### HQTR ######
    ##################
    importlib.reload(hqr.hqtr)
    hqr.hqtr.get_hqtr(sdata, CONST.TMP_PATH, imagedim, dim_x, dim_y, CONST, seed)

    # In[]
    if ( CONST.ANNOTATION_FILE ):
        hqr.hqtr.celltype_refinement_of_hqtr(sdata, CONST.TMP_PATH, imagedim, dim_x, dim_y, CONST)
    else:
        print("[NOTE] No annotation file provided so I will not perform celltype_refinement_of_hqtr")


    # In[]
    #############################
    ###### COMBINE ALL HQR ######
    #############################
    importlib.reload(helperfuncs)
    importlib.reload(hqr.combine_masks)

    if ( CONST.STEP in ['all', 'combine_masks'] ):

        hqr.combine_masks.start_combining_masks(
            CONST.FIGURE_PATH,
            CONST.TMP_PATH,
            imagedim,
            dim_x,
            dim_y,
            CONST.STAINING,
            celltype_refined=False
        )

        print('[finish]')


    # In[]
    ############################
    ###### SUBCELLULAR QC ######
    ############################

    # TODO Bento (see joplin note) --> needs to be install locally right now
    # TODO can yo identify with bento stressed cells? (cells with transcript agglomerations that are related to stress)
    # Which you can see with stress granules. 
    # Calssification of subcellular transcript aggregation forms (different form e.g.. circles) 
    # could corrspond to different function (e.g., stress granules).

    # In[]
    ###########################
    ###### TRANSCRIPT QC ######
    ###########################
    importlib.reload(general.qc_transcript)
    if ( CONST.STEP in ['all', 'transcriptqc'] ):
        # TODO include MT coverage
        # TODO include Rb coverage
        # TODO Hb coverage
        # TODO include transcript count histogram on x and y axsis as in MerQuaCo
        print('[NOTE] Transcript QC')
        figure_path = f'{CONST.FIGURE_PATH}/transcriptqc/'
        # general.qc_transcript.transcriptqc(
        #     sdata,
        #     figure_path,
        #     f'{CONST.TRANSCRIPT_REFERENCE}',
        #     'transcripts'
        # )

        general.qc_transcript.negativeprobeqc(sdata, figure_path, 'transcripts')

        # How many molecules of the gene panel have the z-level? (same height) --> plot distribution. 
        # This helps to categorize the thickness of the tissue slide.
        importlib.reload(general.qc_transcript)
        #general.qc_transcript.transcriptz(sdata, figure_path, 'transcripts')
        print("[finish]")

    # In[]
    ##########################
    ###### CELLCYCLE QC ######
    ##########################
    # Low resources and quick
    if ( CONST.STEP in ['all', 'cellcycleqc'] ):
        print("[TASK] Cell cycle check")
        figure_path = f'{CONST.FIGURE_PATH}/cellcycleqc/'

        plot_colors_phase = ['red', 'blue', 'yellow']

        # Get cell cylce genes
        cell_cycle_genes = [x.strip() for x in open(f'{CONST.CELLCYCLE_GENE_FILE}')]
        cell_cycle_genes = list(set(cell_cycle_genes))
        s_genes = cell_cycle_genes[:43]
        g2m_genes = cell_cycle_genes[43:]

        # Filter for genes that are in the sdata
        cell_cycle_genes = list(set(cell_cycle_genes) & set(rna_adata.var_names))
        s_genes = list(set(s_genes) & set(rna_adata.var_names))
        g2m_genes = list(set(g2m_genes) & set(rna_adata.var_names))

        # Just do cellcycle QC if genes are available
        if ( len(s_genes) == 0 ):
            print("[WARN] Sorry it seems your data has no S phase genes")
        elif ( len(g2m_genes) == 0 ):
            print("[WARN] Sorry it seems your data has no G2M phase genes")
        elif ( len(cell_cycle_genes) > 0 ):
            rna_adata = marker.qc_cellcycle.cellcycle_qc(
                rna_adata,
                figure_path,
                cell_cycle_genes,
                s_genes,
                g2m_genes,
                ['red', 'blue', 'yellow']
            )
            marker.qc_cellcycle.spatial_cellcycle_qc(figure_path, sdata)
        else:
            print("[WARN] Something else went wrong")

        print("[finish]")

    # In[]
    ###############################
    ###### MODEL PREPARATION ######
    ###############################
    # Low resources and quick
    if ( CONST.STEP in ['all', 'modelqc'] ):
        figure_path = f'{CONST.FIGURE_PATH}/modelqc/'

        # For model QC we need to get the normalized data
        sdata['table'].X = sdata['table'].layers['normlogscale']

        sc.tl.pca(rna_adata, n_comps=100)

        df = pd.DataFrame({
                'x': rna_adata.obsm['spatial'][:,0],
                'y': rna_adata.obsm['spatial'][:,1]
            })

        X_pca = rna_adata.obsm['X_pca']

        for i in range(0, CONST.nPCs):
            df[f'PC{i}'] = X_pca[:,i]

        sc.pl.pca_variance_ratio(rna_adata, n_pcs=100, log=True, save='.png')
        shutil.move("figures/pca_variance_ratio.png", f"{figure_path}/pca_variance_ratio.png")

        model_preparation.model_analysis.plot_pca_scatter(df, figure_path, CONST.nPCs)
        model_preparation.model_analysis.plot_spatial_vs_exression_variance(sdata, figure_path, df, CONST.nPCs)
        print("[finish]")

    # In[]
    #######################
    ###### MARKER QC ######
    #######################
    # Low resources, fast
    if ( CONST.STEP in ['all', 'markerqc'] ):
        if ( CONST.ANNOTATION_FILE ):

            figure_path = f'{CONST.FIGURE_PATH}/markerqc'
            importlib.reload(helperfuncs)

            sdata['table'].X = sdata['table'].layers['normlog']

            # TODO for testing - remove later
            # celltypes = ['A', 'B', 'C', 'D', 'E']
            # rna_adata.obs['celltype'] = random.choices(celltypes, k=rna_adata.n_obs)
            celltypes = list(set(rna_adata.obs['celltype']))

            negative_markers = dict({'Invasive_Tumor': ['KRT14', 'MMP1', 'FOXC2'],
                                    'CD8+_T_Cells': ['CD19', 'CD14', 'ITGAM'],
                                    'B_Cells': ['CD3D', 'CD4', 'ITGAM'],
                                    'Stromal': ['CD3D', 'CD14', 'CD68']
                                    })

            positive_markers = dict({'Invasive_Tumor': ['GATA3', 'ERBB2', 'EPCAM'],
                                    'CD8+_T_Cells': ['CD8A', 'CD3D', 'CD247'],
                                    'B_Cells': ['CD19', 'CD79B', 'CD1C'],
                                    'Stromal': ['ACTA2']
                                    })

            # sanity check
            for marker_list in negative_markers.values():
                for m in marker_list:
                    if m not in list(rna_adata.var.index):
                        print(f'[ERROR] I could not find negative marker {m} in rna_adata.var')
                    
            for marker_list in positive_markers.values():
                for m in marker_list:
                    if m not in list(rna_adata.var.index):
                        print(f'[ERROR] I could not find postive marker {m} in rna_adata.var')

            marker.qc_marker.plot_marker_density_and_scatter(sdata, figure_path, negative_markers, 'negative_markers')
            marker.qc_marker.plot_marker_density_and_scatter(sdata, figure_path, negative_markers, 'positive_markers')

            marker.qc_marker.plot_marker_boxplot(
                sdata,
                figure_path,
                negative_markers,
                CONST.ANNOTATION_KEY,
                'negative_markers'
            )

            marker.qc_marker.plot_marker_boxplot(
                sdata,
                figure_path,
                positive_markers, 
                CONST.ANNOTATION_KEY,
                'positive_markers'
            )

            marker.qc_marker.plot_marker_radius_line(
                sdata,
                figure_path,
                negative_markers,
                'negative_markers',
                CONST.THREADS,
                CONST.ANNOTATION_KEY,
                CONST.RADI
            )
            marker.qc_marker.plot_marker_radius_line(
                sdata,
                figure_path,
                positive_markers,
                'positive_markers',
                CONST.THREADS,
                CONST.ANNOTATION_KEY,
                CONST.RADI
            )

            marker.qc_marker.plot_sanpy_score_genes(sdata, figure_path, negative_markers, 'negative_markers')
            marker.qc_marker.plot_sanpy_score_genes(sdata, figure_path, positive_markers, 'positive_markers')

            print("[finish]")
        else:
            print("[NOTE] Marker QC will not be performmed because no annotation was provided.")


    # In[]
    #################################
    ###### ADDITIONAL ANALYSIS ######
    #################################

    staining_list = [0]
    if ( len(stainings) > 1 ):
        staining_list = [str(x) for x in range(0, len(stainings))]
    if ( 'dummy' in stainings ):
        staining_list.remove(str(stainings.index('dummy')))

    # In[]
    importlib.reload(additional_analysis.analysis)
    if ( CONST.STEP in ['all', 'analysis_overview'] and CONST.ANNOTATION_FILE):
        additional_analysis.analysis.celltype_cluster_analysis(
                sdata,
                'overview',
                CONST,
                seed,
                'raw',
                dim_x,
                dim_y,
                imagedim,
                staining_list,
                annotation,
        )
        helperfuncs.sort_files(f'{CONST.FIGURE_PATH}/analysis/overview', 'prefix', ['res.txt', 'done.txt'])
        print(f"[finish] {CONST.STEP}")


    # In[]
    importlib.reload(additional_analysis.analysis)
    if ( CONST.STEP in ['all', 'analysis_cluster'] and CONST.ANNOTATION_FILE):
        additional_analysis.analysis.celltype_cluster_analysis(
                sdata,
                'cluster',
                CONST,
                seed,
                'raw',
                dim_x,
                dim_y,
                imagedim,
                staining_list,
                annotation,
        )
        helperfuncs.sort_files(f'{CONST.FIGURE_PATH}/analysis/cluster', 'prefix', ['res.txt', 'done.txt'])
        print(f"[finish] {CONST.STEP}")

    # In[]
    importlib.reload(additional_analysis.analysis)
    if ( CONST.STEP in ['all', 'analysis_category'] and CONST.ANNOTATION_FILE):
        additional_analysis.analysis.cell_category_analysis(
                sdata,
                'category',
                CONST,
                seed,
                'raw',
                dim_x,
                dim_y,
                imagedim,
                staining_list,
        )
        print(f"[finish] {CONST.STEP}")


    print("[FINISH]")