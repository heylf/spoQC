#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

# In[]
import sys
import numba

# Utility imports
import os
import random
import rich_click as click
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
from spoqc import folder_structure
from spoqc import plot_config
from spoqc import subworkflows
from spoqc import _core

@click.command(
    help="""
    spoQC is a modular framework for multimodal quality control (QC) of imaging-based
    spatially resolved transcriptomics (SRT).
    """
)
@click.version_option(version="0.0.1")

# Required options
@click.option("-i", "--input_file", type=str, required=True, 
              help="""Path to the input directory containing Xenium data.""")
@click.option("-o", "--output_dir", type=str, required=True, 
              help="""Path to the output directory containing the report.""")
@click.option("-t", "--tmp_dir", type=str, required=True, 
              help="""Path to the tmp directory where spoQC saves tmp files.""")
@click.option("-t", "--tmp_dir", type=str, required=True, 
              help="""Path to the tmp directory where spoQC saves tmp files.""")
@click.option("-s", "--step", type=str, required=True, default="all",
              help="""Steps to run for QC.""")

# Optional parameters
@click.option("-a", "--annotation_file", type=str, required=False,
              help="""Path to the annotation file.""")
@click.option("--cellcycle_gene_file", type=str, required=False,
              help="""Path to a JSON file with "S" and "G2M" keys listing S-phase and G2M-phase gene names.""")
@click.option("--cluster_celltype", type=str, required=False,
              help="""Name of the cluster cell type you want to specifically analyse.""")
@click.option("--dataset", type=str, required=False,
              help="""
              This is used for to apply standardization to spatial data for the data used in the publication.
              """
              )
@click.option("--doublet_prior_std", type=int, required=False, default=100,
              help="""
              The std for the doublet prior estimation. If you increase it then the impact of doublet events increaes,
              that means doublets events will impact more cells and give them lower quality.
              """
              )
@click.option("--kmeans_sample_size", type=int, required=False, default=5_000_000,
              help="""
              Number of pixels randomly subsampled to fit the pixel-cluster MiniBatchKMeans model (hqpr/hqtr).
              The full dataset is then labeled in parallel using the fitted model.
              """
              )
@click.option("-n", "--nthreads", type=int, required=False, default=1,
              help="""Number of cores to be used.""")
@click.option("--overwrite", required=False, is_flag=True,
              help="""Overwriting temporary files.""")
@click.option("--pixel_qc_chunk_size", type=int, required=False, default=200_000,
              help="""
              Row-chunk size for the pixel-level QC dask arrays/dataframes (hqpr/hqtr clustering and scoring).
              Larger values reduce dask task-graph overhead but increase peak memory per chunk.
              """
              )
@click.option("--reference_file", type=str, required=False,
              help="""Path to a transcript reference file for the transcript QC.""")
@click.option("--staining", type=str, required=False, default="0",
              help="""Staining to be used. You have to provide the integer of the staining in the spatial data.""")
@click.option("--thresh_prior_pixel", type=float, required=False, default=None,
              help="""
              You can set a prior threshold for the pixel prior distribution. 
              Please read the documentation to understand what this threshold does before you set it.
              """
              )
@click.option("--nstds_prior_pixel", type=int, required=False, default=6,
              help="""
              You can set the number of stds for the pixel prior distribution. 
              Please read the documentation to understand what this does before you set it.
              """
              )

# Testing options
@click.option("--dev_test", required=False, is_flag=True,
              help="""This is just for developing and testing the tool.""")
@click.option("--dev_report", required=False, is_flag=True,
              help="""This is just for developing and testing the tool (report).""")

def main(**kwargs) -> None:
    print("[START]")


# In[]
    importlib.reload(_core.config)
    args =  _core.config.Args(kwargs)
    args.print_args()

# In[]

    # Seeds (!!! DO NOT CHANGE THIS SEED !!!)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"[NOTE] seed {args.seed}")

    # ---------------- Environment ----------------
    # Numba threads
    print(f"[NOTE] Setting numba threads to {args.nthreads}")
    requested_threads = args.nthreads
    maximum_threads = numba.config.NUMBA_NUM_THREADS
    active_threads = min(requested_threads, maximum_threads)
    print(
        f"[NOTE] Numba thread pool maximum: {maximum_threads}; "
        f"using: {active_threads}"
    )
    numba.set_num_threads(active_threads)

    # Blosc threads (for the Zarr datasets we still write)
    os.environ["BLOSC_NTHREADS"] = str(args.nthreads)
    # Timer
    timer = helperfuncs.Timer()


# In[]

    # ---------------- Folder Structure ------------
    folder_structure.create_folder_structure(CONST)

    # ---------------- Setting matplot styles ------------
    plot_config.set_pub_style()

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
        df_labels = df_labels[['Barcode', 'Cluster']]
        df_labels.index = df_labels['Barcode']
        df_labels = df_labels.drop(columns='Barcode')
        df_labels.columns = [CONST.ANNOTATION_KEY]

        # Check if annotation and anndata have the same number of cells
        if rna_adata.n_obs != len(df_labels):
            warn_text = f"[WARN]: The annotation has a different number of cells {len(df_labels)} than your sdata "
            warn_text += f"{rna_adata.n_obs}. Please Check your annotation."
            print(warn_text)
        
        # I have to map here if that is the case.
        if 'cell_id' in rna_adata.obs.columns:
            if df_labels.index[0] in list(rna_adata.obs['cell_id']) and df_labels.index[0] not in list(rna_adata.obs.index):
                df_labels.index = df_labels.index.map(mapping)
        
        if ( type(rna_adata.obs.index[0]) == str ):
            df_labels.index = df_labels.index.map(str)



        # Sometimes annoation does not contain all cells.
        rna_adata.obs = rna_adata.obs.join(df_labels[CONST.ANNOTATION_KEY], how='left')
        rna_adata.obs[CONST.ANNOTATION_KEY] = rna_adata.obs[CONST.ANNOTATION_KEY].fillna('unkown')

        
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
            celltype_refined=False
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
        #     'transcripts'
        # )
        subworkflows.qc_transcript.negativeprobeqc(sdata, figure_path, 'transcripts')
        print("[finish]")

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
    if ( CONST.STEP in ['all', 'final_report'] ):
        subworkflows.final_report.create_final_report(CONST.FIGURE_PATH, stainings, CONST.GENERATE_REPORT_DOC)
    print("[FINISH]")
    # %%
