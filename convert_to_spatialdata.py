#!/usr/bin/env python
# coding: utf-8

print("[START]")

# Tool imports
import spatialdata as sd
from spatialdata_io import xenium
import argparse

#########################################
###### GLOBAL VARS and DIRECTORIES ######
#########################################

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

# optional
parser.add_argument(
    "-t", "--threads",
    dest="threads",
    type=int,
    default=1, 
    help="Number of cores to be used.",
    required=False
)

args = vars(parser.parse_args())

#######################
###### LOAD DATA ######
#######################
print('[NOTE] Load data')

sd_xenium_obj = xenium(
            f"{args['input']}/outs",
            n_jobs=args['threads'],
            cells_as_shapes=True,
            nucleus_boundaries=True,
            transcripts=True,
            morphology_mip=True,
            morphology_focus=True,
)
print(sd_xenium_obj)

print("[NOTE] Write data")
sd_xenium_obj.write(f"{args['input']}/spatialdata")

print("[FINSIH]")
