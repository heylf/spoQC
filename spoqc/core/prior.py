import sys
import numpy as np
import pandas as pd
import dask.dataframe as dd

from .. import helperfuncs

class Prior:
    def __init__(
            self, 
            calc_func,
            name,
            modality,
            *,
            tmp_path = None,
            needs_metrics = [],
            args=None,
            kwargs=None,
        ):

        self._calc_func = calc_func

        self.args = args if args else []
        self.kwargs = kwargs if kwargs else {}

        self.name = name
        self.tmp_path = tmp_path
        self.needs_metrics = needs_metrics
        self.modality = modality

    def calculate(self):
        return self._calc_func(*self.args, **self.kwargs)
    

class PriorSet:
    def __init__(self, name, priorset):
        self.name = name
        self.priors_calculated = False
        self.prior_df = None
        self.prior_ddf = None
        self.priors_calculated = False

        if len(priorset) == 0:
            sys.exit("[ERROR] Prior set is empty")
        else:
            self.priorset = priorset

    def calculate_priors_df(self):
        if not self.priors_calculated:

            prior_df = pd.DataFrame({})
            for prior in self.priorset:
                print(f"[NOTE] Calculating {prior.name}")
                timer = helperfuncs.Timer()
                timer.start()
                prior_df[prior.name] = prior.calculate()
                timer.stop()

                # NaN priors are treated as missing evidence in the combination, so at least report them.
                num_nans = int(prior_df[prior.name].isna().sum())
                if ( num_nans > 0 ):
                    print(f"[WARN] {prior.name} has {num_nans} NaN values, they are ignored when combining priors")

            self.prior_df = prior_df
            self.priors_calculated = True

    def calculate_priors_ddf(self):
        if not self.priors_calculated:

            ddf_list = []
            for prior in self.priorset:
                print(f"[NOTE] Calculating {prior.name}")
                timer = helperfuncs.Timer()
                timer.start()
                ddf_list.append(prior.calculate())
                timer.stop()

            image_ddf = ddf_list[0]
            if len(ddf_list) > 1:
                image_ddf = dd.concat(ddf_list, axis=1)
            
            self.prior_ddf = image_ddf
            self.priors_calculated = True

    # Asymetric evidence aggregation will put a penalty on priors that are extremely bad.
    # Example A: [.90,.90,.90,.90,.90,.90], result = 0.900
    # Example B: [.99,.99,.99,.99,.99,.10], result = 0.391
    # NaN priors are missing evidence and ignored. If all priors are NaN, the result is uninformative (0.5).
    def combine_prior_asymmetric_evidence_aggregation(self):

        def _asymmetric_evidence_aggregation(priors, gamma=2.0, axis=-1):
            priors = np.asarray(priors, dtype=float)
            priors = priors[~np.isnan(priors)]
            if ( priors.size == 0 ):
                return 0.5
            priors = np.clip(priors, 1e-12, 1.0)
            surprise = -np.log(priors)
            weighted_surprise = np.mean(surprise ** gamma, axis=axis) ** (1 / gamma)
            return np.exp(-weighted_surprise)
    
        return self.prior_df.apply(_asymmetric_evidence_aggregation, axis=1)

    
    def combine_prior_traffic_light_system(self):

        # NaN priors are missing evidence and not counted (NaN < t is always False and would look "green").
        # If all priors are NaN the cell can not be judged, thus "yellow".
        def _traffic_light(row, bad_threshold=0.3, warning_threshold=0.6):
            row = row.dropna()
            if ( len(row) == 0 ):
                return "yellow"

            n_bad = sum(p < bad_threshold for p in row)
            n_warning = sum(
                bad_threshold <= p < warning_threshold
                for p in row
            )

            if n_bad >= 2:
                return "red"

            if n_bad == 1 or n_warning >= 2:
                return "yellow"

            return "green"

        return self.prior_df.apply(_traffic_light, axis=1)


    def combine_prior_mean_ddf(self):
        colnames = [f"prob_{prior.name}" for prior in self.priorset]
        use_ddf = self.prior_ddf[colnames]
        print(f"[NOTE] Colnames in use_ddf: f{use_ddf.columns}")
        row_means_series = use_ddf.mean(axis=1)
        return row_means_series