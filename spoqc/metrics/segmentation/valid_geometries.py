import pandas as pd
import numpy as np
import concurrent.futures

from scipy.spatial import cKDTree

from ... import helperfuncs
from ... import core

# This is kind of a dummy script because valid geometries have to be checked already in the beginning.
# This script's purpose is for the Metric to Prior framework only.
# Can be used as a template.

def calc_valid_gemoetries():
    print("[NOTE] Calculate valid geometries")
    
def init_metric(enterprise):

    # These have to be defined.
    metric_name = "valid_geometries"
    combined_metric_name = None
    needs_metrics = []
    step_when_it_is_calculated = ["generalqc", "all"]
    loaded_for_analysis = True
    loaded_for_visualization = True
    prior = True

    # These are given my your metric calc function.
    args = []
    kwargs = None

    metric = core.metric.Metric(
        calc_valid_gemoetries, 
        metric_name,
        combined_metric_name = combined_metric_name,
        needs_metrics = needs_metrics,
        step_when_it_is_calculated = step_when_it_is_calculated,
        loaded_for_analysis = loaded_for_analysis,
        loaded_for_visualization = loaded_for_visualization,
        prior = prior,
        args = args,
        kwargs = kwargs,
    )    
    
    return metric
