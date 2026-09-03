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
    subworkflows.qc_sc.run_qc_sc(enterprise)

    # In[]
    subworkflows.qc_wsi.run_qc_wsi(enterprise)

    # In[]
    subworkflows.qc_bubble.run_qc_bubble(enterprise)

    # In[]
    subworkflows.qc_doublets.run_qc_doublets(enterprise)

    # In[]
    subworkflows.qc_void.run_qc_void(enterprise)

    # In[]
    subworkflows.qc_cell.run_qc_cell(enterprise)

    # In[]
    subworkflows.hqcr.start_hqcr(enterprise)

    # In[]
    subworkflows.hqcr.start_hqcr_celltype(enterprise)

    # In[]
    subworkflows.hqpr.get_hqpr(enterprise)

    # In[]
    subworkflows.qc_ambient.start_qc_ambient(enterprise)

    # In[]
    subworkflows.hqtr.get_hqtr(enterprise)

    # In[]
    subworkflows.combine_masks.run_combine_masks(enterprise)

    # In[]
    subworkflows.qc_transcript.run_qc_transcript(enterprise)

    # In[]
    subworkflows.qc_cellcycle.run_qc_cellcycle(enterprise)

    # In[]
    subworkflows.qc_model.run_qc_model(enterprise)

    # In[]
    subworkflows.qc_marker.run_qc_marker(enterprise)

    # In[]
    subworkflows.qc_additional_analysis.run_qc_additional_analysis(enterprise)

    # In[]
    subworkflows.final_report.run_final_report(enterprise)

    timer.stop()
    print("[FINISH]")
    # %%
