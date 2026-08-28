import pandas as pd
import numpy as np

from ... import helperfuncs

from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from dask_ml.preprocessing import MinMaxScaler

def calc_probs(sdata, figure_path, obj_type):

    data = np.array(sdata['table'].obs[f'wvalid_{obj_type}_geometry'])

    helperfuncs.plot_histogram_for_array(
        data,
        2,
        figure_path,
        f"Invalid {obj_type} geometries: t=0.0 with 0.0 x {np.round(0.0, 3)} std",
        f"invalid_{obj_type}_gemotry_prior",
        t=0.0
    )

    return 1 - np.array(sdata['table'].obs[f'wvalid_{obj_type}_geometry'])
