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

# Own scripts
from spoqc import general
from spoqc import hqr
from spoqc import helperfuncs
from spoqc import process_datasets
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
@click.option("-d", "--datatype", type=str, required=True, default="xenium",
              help="""
              The type of data you are providing.
              Please check spoQC's documentation to know which datatypes are covered.
              """
              )

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

    importlib.reload(_core.starship)
    importlib.reload(_core._config)
    importlib.reload(_core._data)
    importlib.reload(_core.dataloaders.xenium)
    enterprise = _core.starship.Enterpise(kwargs)

    # Timer class
    timer = helperfuncs.Timer()

    # In[]
    # Load data
    enterprise.load_cargo_data()

    # In[]
    # This file is useful to later figure out which folder stands for which staining.
    # Staining names can be weird and would disrupt the code, thus I have to use the indices.
    print("[NOTE] Write staining log")
    staining_log = open(f'{enterprise.args.output_dir}/staining_log.txt', 'w')
    for i, staining in enumerate(enterprise.cargo.stainings):
        staining_log.write(f'{i} = {staining} \n')
    staining_log.close()
    print("[finish]")

    
    # In[]
    enterprise.generate_unsupervised_annotation()


    # In[]
    # Low resources and quick
    ########################
    ###### GENERAL QC ######
    ########################
    if ( enterprise.args.step in ['all', 'unittest', 'generalqc'] ):
        print('[NOTE] General QC')
        figure_path = f'{enterprise.args.output_dir}/generalqc/'
        obs_columns = subworkflows.qc_sc.run_qc_sc(enterprise.cargo.sdata, figure_path, CONST, obs_columns)

    # In[]
    # Low resources and quick
    ############################
    ###### WHOLE SLIDE QC ######
    ############################
    if ( enterprise.args.step in ['all', 'whole_slide_qc'] ):
        print('[NOTE] Domain QC')
        figure_path = f'{enterprise.args.output_dir}/whole_slide_qc/'
        subworkflows.qc_wsi.generate_input(enterprise.cargo.sdata, figure_path, CONST)
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
    if ( enterprise.args.step in ['all', 'unittest', 'bubbleqc'] ):
        figure_path = f'{enterprise.args.output_dir}/bubbleqc/'
        obs_columns = subworkflows.qc_bubble.run_qc_bubble(enterprise.cargo.sdata, figure_path, CONST, obs_columns)

    # In[]
    ########################
    ###### DOUBLET QC ######
    ########################
    # High resources and slow (takes 18-19 hours for a full dataset)
    if ( enterprise.args.step in ['all', 'unittest', 'doubletqc'] ):
        figure_path = f'{enterprise.args.output_dir}/doubletqc/'
        obs_columns = subworkflows.qc_doublets.run_qc_doublets(enterprise.cargo.sdata, figure_path, CONST, annotation, obs_columns)

    # In[]
    # Low resource but long (takes 4-5 hours)
    #####################
    ###### VOID QC ######
    #####################
    if ( enterprise.args.step in ['all', 'unittest', 'voidqc'] ):
        figure_path = f'{enterprise.args.output_dir}/voidqc/'
        obs_columns = subworkflows.qc_void.run_qc_void(enterprise.cargo.sdata, figure_path, CONST, obs_columns)

    # In[]
    #####################
    ###### CELL QC ######
    #####################
    # Low resources and quicks for full dataset (40-50 min)
    if ( enterprise.args.step in ['all', 'unittest', 'cellqc'] ):
        figure_path = f'{enterprise.args.output_dir}/cellqc/'
        obs_columns = subworkflows.qc_cell.run_qc_cell(enterprise.cargo.sdata, figure_path, CONST, obs_columns)

    # In[]
    ##################
    ###### HQCR ######
    ##################
    # Low resources and for a full dataset it takes 30 - 40 min.
    if ( enterprise.args.step in ['all', 'unittest', 'hqcr_ident'] ):
        subworkflows.hqcr.start_hqcr(enterprise.cargo.sdata, CONST.TMP_PATH, imagedim, CONST, seed)
        print("[finish]")

    # In[]
    # Low resources and quick.
    if ( enterprise.args.step in ['all', 'hqcr_celltype'] ):
        if ( CONST.ANNOTATION_FILE ):
            subworkflows.hqcr.start_hqcr_celltype(enterprise.cargo.sdata, CONST.TMP_PATH, imagedim, CONST)
            print("[finish]")
        else:
            print("[NOTE] No annotation file provided so I will not perform start_hqcr_celltype")

    # In[]
    ##################
    ###### HQPR ######
    ##################
    subworkflows.hqpr.get_hqpr(
        enterprise.cargo.sdata,
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
        subworkflows.hqpr.celltype_refinement_of_hqpr(enterprise.cargo.sdata, CONST.TMP_PATH, imagedim, dim_x, dim_y, CONST)
    else:
        print("[NOTE] No annotation file provided so I will not perform celltype_refinement_of_hqpr")

    # In[]
    #####################
    ###### AMBIENT ######
    #####################
    if ( enterprise.args.step in ['all', 'hqtr', 'unittest', 'ambientqc'] ):
        figure_path = f'{enterprise.args.output_dir}/ambientqc/'
        _ = subworkflows.qc_ambient.start_qc_ambient(enterprise.cargo.sdata, figure_path, CONST.TMP_PATH)

    # In[]
    ##################
    ###### HQTR ######
    ##################
    subworkflows.hqtr.get_hqtr(
        enterprise.cargo.sdata, 
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
        subworkflows.hqtr.celltype_refinement_of_hqtr(enterprise.cargo.sdata, CONST.TMP_PATH, imagedim, dim_x, dim_y, CONST)
    else:
        print("[NOTE] No annotation file provided so I will not perform celltype_refinement_of_hqtr")


    # In[]
    #############################
    ###### COMBINE ALL HQR ######
    #############################
    if ( enterprise.args.step in ['all', 'combine_masks'] ):

        hqr.combine_masks.start_combining_masks(
            enterprise.args.output_dir,
            CONST.TMP_PATH,
            imagedim,
            dim_x,
            dim_y,
            CONST.STAINING,
            celltype_refined=False
        )

        print('[finish]')

    # In[]
    if ( enterprise.args.step in ['combine_masks_zoom'] ):

        hqr.combine_masks_zoom.start_combining_masks(
            enterprise.cargo.sdata,
            enterprise.args.output_dir,
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
    if ( enterprise.args.step in ['all', 'transcriptqc'] ):
        print('[NOTE] Transcript QC')
        figure_path = f'{enterprise.args.output_dir}/transcriptqc/'
        # subworkflows.qc_transcript.transcriptqc(
        #     enterprise.cargo.sdata,
        #     figure_path,
        #     f'{CONST.TRANSCRIPT_REFERENCE}',
        #     'transcripts'
        # )
        subworkflows.qc_transcript.negativeprobeqc(enterprise.cargo.sdata, figure_path, 'transcripts')
        print("[finish]")

    # In[]
    ##########################
    ###### CELLCYCLE QC ######
    ##########################
    # Low resources and quick
    if ( enterprise.args.step in ['all', 'cellcycleqc'] ):
        print("[TASK] Cell cycle check")
        figure_path = f'{enterprise.args.output_dir}/cellcycleqc/'
        subworkflows.qc_cellcycle.run_qc_cellcycle(enterprise.cargo.sdata, figure_path, CONST)
        print("[finish]")

    # In[]
    ###############################
    ###### MODEL PREPARATION ######
    ###############################
    # Low resources and quick
    if ( enterprise.args.step in ['all', 'modelqc'] ):
        figure_path = f'{enterprise.args.output_dir}/modelqc/'
        subworkflows.qc_model.run_qc_model(enterprise.cargo.sdata, figure_path, CONST)
        print("[finish]")

    # In[]
    #######################
    ###### MARKER QC ######
    #######################
    # Low resources, fast
    if ( enterprise.args.step in ['markerqc'] ):
        if ( CONST.ANNOTATION_FILE ):
            figure_path = f'{enterprise.args.output_dir}/markerqc'
            subworkflows.qc_marker.run_qc_marker(enterprise.cargo.sdata, figure_path, CONST)
            print("[finish]")
        else:
            print("[NOTE] Marker QC will not be performmed because no annotation was provided.")

    # In[]
    #################################
    ###### ADDITIONAL ANALYSIS ######
    #################################
    if ( 'analysis' in enterprise.args.step or enterprise.args.step == 'all' ):
        subworkflows.qc_additional_analysis.run_qc_additional_analysis(
            enterprise.cargo.sdata,
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
    if ( enterprise.args.step in ['all', 'final_report'] ):
        subworkflows.final_report.create_final_report(enterprise.args.output_dir, stainings, CONST.GENERATE_REPORT_DOC)
    print("[FINISH]")
    # %%
