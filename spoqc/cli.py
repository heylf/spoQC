#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

# In[]
import sys

# Utility imports
import random
import argparse
import numpy as np
import pandas as pd
import re
import importlib

# Tool imports
import spatialdata as sd
import spatialdata_plot
from spatialdata.models import PointsModel

# Own scripts
from spoqc import general
from spoqc import hqr
from spoqc import helperfuncs
from spoqc import process_datasets
from spoqc.core import figures
from spoqc import folder_structure
from spoqc import plot_config
from spoqc import subworkflows
from spoqc.core import threads
from spoqc.core import transcripts

# In[]
def main(args_ns: argparse.Namespace) -> None:
    """Run spoQC on arguments parsed by `spoqc.cli_args.build_parser()`.

    The entry point is `spoqc.__main__.main(argv)`: it parses argv, calls
    `spoqc.core.threads.configure` and only then imports this module. `main` itself
    takes the parsed Namespace (it used to take argv) and refuses to run if the
    thread budget was never configured.
    """
    if threads.N is None:
        raise RuntimeError(
            "spoqc.core.threads.configure(n) has not run; start spoQC through "
            "`spoqc` / `python -m spoqc` or call spoqc.__main__.main(argv)"
        )
    print("[START]")

# In[]

    # Setting matplot styles
    plot_config.set_pub_style()

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
            return threads.N
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
        def PIXEL_QC_CHUNK_SIZE():
            return args['pixel_qc_chunk_size']
        @constant
        def KMEANS_SAMPLE_SIZE():
            return args['kmeans_sample_size']
        @constant
        def GMM_N_INIT():
            return args['gmm_n_init']
        @constant
        def THRESHOLD_PRIOR_PIXEL():
            return args['thresh_prior_pixel']
        @constant
        def DOULET_PRIOR_STD():
            return args['doublet_prior_std']
        @constant
        def NSTDS_PRIOR_PIXEL():
            return args['nstds_prior_pixel']
        @constant
        def CLUSTER_CELLTYPE():
            return args['cluster_celltype']
        @constant
        def GENERATE_REPORT_DOC():
            if args['dev_report']:
                return True
            else:
                return False

    # Initialize constant variables
    CONST = _Const()
    constant_members = [attr_name for attr_name in vars(_Const) if isinstance(getattr(_Const, attr_name), property)]
    for attr_name in constant_members:
        attr_value = getattr(_Const, attr_name).fget(_Const)
        print(f'Attribute Name: {attr_name}')
        print(f'Attribute Value: {attr_value}')

    # Figures are written by worker processes while the main thread computes on; figures.start
    # splits the thread budget between them and the compute (see spoqc.core.figures).
    figures.start(CONST.THREADS)
    try:
        run(CONST)
    except BaseException:
        figures.abort()
        raise
    figures.stop()


def run(CONST):
    # Seeds (!!! DO NOT CHANGE THIS SEED !!!)
    seed=123
    random.seed(seed)
    np.random.seed(seed)
    print(f"[NOTE] seed {seed}")

    # Timer
    timer = helperfuncs.Timer()

    # ---------------- Folder Structure ------------
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

    # Ensure transcripts have a globally unique, monotonic index (required by
    # spatialdata>=0.7's get_centroids/transform). The Xenium zarr reader can
    # produce a points dataframe whose partitions each restart their own local
    # index, and plain ddf.reset_index(drop=True) does not fix this since dask
    # resets per-partition; deduplicate_dask_index() offsets each partition by
    # the cumulative length of the partitions before it instead. The existing
    # 'global' transform lives in .attrs, which map_partitions carries over,
    # so PointsModel.parse() picks it up without passing transformations=.
    sdata.points['transcripts'] = PointsModel.parse(
        helperfuncs.deduplicate_dask_index(sdata.points['transcripts'])
    )

    # In[]
    # Cropping for testing
    if ( CONST.TESTING > 0 ):
        print('[NOTE] Cropping for testing')
        start = 10500
        end = CONST.TESTING
        cropped_sdata, _, _ = helperfuncs.image_crop(sdata, start, start, start+end, start+end+500, 'global')
        sdata = cropped_sdata

    # In[]
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

    # In[]
    # Check for nan's in transcripts feature names
    sdata.points['transcripts']['feature_name'] = (
        sdata.points['transcripts']['feature_name']
        .astype('string')
        .fillna('NaN')
        .astype('category')
    )

    # In[]
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

    # In[]
    # I need string indexes for anndata else code breaks
    sdata['table'].obs.index = sdata['table'].obs.index.astype(str)
    sdata['table'].obs.index.name = 'index'

    # In[]
    # Get RNA data and set raw data layer
    rna_adata = sdata['table']
    rna_adata.layers['raw'] = rna_adata.X

    # Add annotation
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
    # Low resources and quick
    ########################
    ###### GENERAL QC ######
    ########################
    if ( CONST.STEP in ['all', 'unittest', 'generalqc'] ):
        print('[NOTE] General QC')
        figure_path = f'{CONST.FIGURE_PATH}/generalqc/'
        obs_columns = subworkflows.qc_sc.run_qc_sc(sdata, figure_path, CONST, obs_columns)

    # In[]
    # Low resources and quick
    ############################
    ###### WHOLE SLIDE QC ######
    ############################
    if ( CONST.STEP in ['all', 'whole_slide_qc'] ):
        print('[NOTE] Domain QC')
        figure_path = f'{CONST.FIGURE_PATH}/whole_slide_qc/'
        subworkflows.qc_wsi.generate_input(sdata, figure_path, CONST)
        subworkflows.qc_wsi.measure_stripe_thickness_and_black_area(
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
        obs_columns = subworkflows.qc_bubble.run_qc_bubble(sdata, figure_path, CONST, obs_columns)

    # In[]
    ########################
    ###### DOUBLET QC ######
    ########################
    # High resources and slow (takes 18-19 hours for a full dataset)
    if ( CONST.STEP in ['all', 'unittest', 'doubletqc'] ):
        figure_path = f'{CONST.FIGURE_PATH}/doubletqc/'
        obs_columns = subworkflows.qc_doublets.run_qc_doublets(sdata, figure_path, CONST, annotation, obs_columns)

    # In[]
    # Low resource but long (takes 4-5 hours)
    #####################
    ###### VOID QC ######
    #####################
    if ( CONST.STEP in ['all', 'unittest', 'voidqc'] ):
        figure_path = f'{CONST.FIGURE_PATH}/voidqc/'
        obs_columns = subworkflows.qc_void.run_qc_void(sdata, figure_path, CONST, obs_columns)

    # In[]
    #####################
    ###### CELL QC ######
    #####################
    # Low resources and quicks for full dataset (40-50 min)
    if ( CONST.STEP in ['all', 'unittest', 'cellqc'] ):
        figure_path = f'{CONST.FIGURE_PATH}/cellqc/'
        obs_columns = subworkflows.qc_cell.run_qc_cell(sdata, figure_path, CONST, obs_columns)

    # In[]
    #####################
    ###### AMBIENT ######
    #####################
    if ( CONST.STEP in ['all', 'hqtr', 'unittest', 'ambientqc'] ):
        figure_path = f'{CONST.FIGURE_PATH}/ambientqc/'
        _ = subworkflows.qc_ambient.start_qc_ambient(sdata, figure_path, CONST.TMP_PATH)

    # In[]
    ##################
    ###### HQTR ######
    ##################
    subworkflows.hqtr.get_hqtr(
        sdata, 
        CONST.TMP_PATH, 
        imagedim, 
        dim_x, 
        dim_y, 
        CONST, 
        seed,
        thresh_p=CONST.THRESHOLD_PRIOR_PIXEL,
        nstds_p=CONST.NSTDS_PRIOR_PIXEL,
    )

    # In[]
    ###########################
    ###### TRANSCRIPT QC ######
    ###########################
    if ( CONST.STEP in ['all', 'transcriptqc'] ):
        print('[NOTE] Transcript QC')
        figure_path = f'{CONST.FIGURE_PATH}/transcriptqc/'
        # subworkflows.qc_transcript.transcriptqc(
        #     sdata,
        #     figure_path,
        #     f'{CONST.TRANSCRIPT_REFERENCE}',
        # )
        subworkflows.qc_transcript.negativeprobeqc(sdata, figure_path)
        print("[finish]")

    # The transcript consumers (doubletqc .. transcriptqc) run back to back: load once,
    # compute all, unload before hqcr/hqpr. None of the moved steps (ambientqc, hqtr,
    # transcriptqc) reads anything hqcr/hqpr produce, and none uses a shared RNG state.
    transcripts.release(sdata)

    # In[]
    ##################
    ###### HQCR ######
    ##################
    # Low resources and for a full dataset it takes 30 - 40 min.
    if ( CONST.STEP in ['all', 'unittest', 'hqcr_ident'] ):
        subworkflows.hqcr.start_hqcr(sdata, CONST.TMP_PATH, imagedim, CONST, seed)
        print("[finish]")

    # In[]
    # Low resources and quick.
    if ( CONST.STEP in ['all', 'hqcr_celltype'] ):
        if ( CONST.ANNOTATION_FILE ):
            subworkflows.hqcr.start_hqcr_celltype(sdata, CONST.TMP_PATH, imagedim, CONST)
            print("[finish]")
        else:
            print("[NOTE] No annotation file provided so I will not perform start_hqcr_celltype")

    # In[]
    ##################
    ###### HQPR ######
    ##################
    subworkflows.hqpr.get_hqpr(
        sdata,
        CONST.TMP_PATH,
        imagedim,
        dim_x,
        dim_y,
        CONST,
        seed,
        thresh_p=CONST.THRESHOLD_PRIOR_PIXEL,
        nstds_p=CONST.NSTDS_PRIOR_PIXEL,
    )

    # In[]
    if ( CONST.ANNOTATION_FILE ):
        subworkflows.hqpr.celltype_refinement_of_hqpr(sdata, CONST.TMP_PATH, imagedim, dim_x, dim_y, CONST)
    else:
        print("[NOTE] No annotation file provided so I will not perform celltype_refinement_of_hqpr")

    # In[]
    if ( CONST.ANNOTATION_FILE ):
        subworkflows.hqtr.celltype_refinement_of_hqtr(sdata, CONST.TMP_PATH, imagedim, dim_x, dim_y, CONST)
    else:
        print("[NOTE] No annotation file provided so I will not perform celltype_refinement_of_hqtr")


    # In[]
    #############################
    ###### COMBINE ALL HQR ######
    #############################
    if ( CONST.STEP in ['all', 'combine_masks'] ):

        hqr.combine_masks.start_combining_masks(
            CONST.FIGURE_PATH,
            CONST.TMP_PATH,
            imagedim,
            dim_x,
            dim_y,
            CONST.STAINING,
            celltype_refined=False,
            threads=CONST.THREADS
        )

        print('[finish]')

    # In[]
    if ( CONST.STEP in ['combine_masks_zoom'] ):

        hqr.combine_masks_zoom.start_combining_masks(
            sdata,
            CONST.FIGURE_PATH,
            CONST.TMP_PATH,
            CONST.IMAGE_TYPE,
            CONST.RESOLUTION,
            imagedim,
            dim_x,
            dim_y,
            CONST.STAINING,
            celltype_refined=False
        )

        print('[finish]')

    # In[]
    ##########################
    ###### CELLCYCLE QC ######
    ##########################
    # Low resources and quick
    if ( CONST.STEP in ['all', 'cellcycleqc'] ):
        print("[TASK] Cell cycle check")
        figure_path = f'{CONST.FIGURE_PATH}/cellcycleqc/'
        subworkflows.qc_cellcycle.run_qc_cellcycle(sdata, figure_path, CONST)
        print("[finish]")

    # In[]
    ###############################
    ###### MODEL PREPARATION ######
    ###############################
    # Low resources and quick
    if ( CONST.STEP in ['all', 'modelqc'] ):
        figure_path = f'{CONST.FIGURE_PATH}/modelqc/'
        subworkflows.qc_model.run_qc_model(sdata, figure_path, CONST)
        print("[finish]")

    # In[]
    #######################
    ###### MARKER QC ######
    #######################
    # Low resources, fast
    if ( CONST.STEP in ['markerqc'] ):
        if ( CONST.ANNOTATION_FILE ):
            figure_path = f'{CONST.FIGURE_PATH}/markerqc'
            subworkflows.qc_marker.run_qc_marker(sdata, figure_path, CONST)
            print("[finish]")
        else:
            print("[NOTE] Marker QC will not be performmed because no annotation was provided.")

    # In[]
    #################################
    ###### ADDITIONAL ANALYSIS ######
    #################################
    if ( 'analysis' in CONST.STEP or CONST.STEP == 'all' ):
        subworkflows.qc_additional_analysis.run_qc_additional_analysis(
            sdata,
            CONST,
            annotation,
            seed,
            imagedim,
            dim_x,
            dim_y,
        )

    # In[]
    ##########################
    ###### FINAL REPORT ######
    ##########################
    # Low resources, fast
    figures.wait()  # the report reads the figures
    if ( CONST.STEP in ['all', 'final_report'] ):
        subworkflows.final_report.create_final_report(CONST.FIGURE_PATH, stainings, CONST.GENERATE_REPORT_DOC, bool(CONST.ANNOTATION_FILE))
    print("[FINISH]")
    # %%
