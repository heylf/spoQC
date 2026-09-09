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

    def calculate(self):
        return self._calc_func(*self.args, **self.kwargs)
    

class PriorSet:
    def __init__(self, name, priorset):
        self.name = name
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
            self.prior_df = prior_df

            self.priors_calculated = True

    def calculate_priors_ddf(self):
        if not self.priors_calculated:

            for prior in self.priorset:
                print(f"[NOTE] Calculating {prior.name}")
                timer = helperfuncs.Timer()
                timer.start()
                prior.calculate()
                timer.stop()

            self.priors_calculated = True

    # Asymetric evidence aggregation will put a penalty on priors that are extremely bad.
    # Example A: [.90,.90,.90,.90,.90,.90], result = 0.900
    # Example B: [.99,.99,.99,.99,.99,.10], result = 0.391
    def combine_prior_asymmetric_evidence_aggregation(self):

        def _asymmetric_evidence_aggregation(priors, gamma=2.0, axis=-1):
            priors = np.asarray(priors, dtype=float)
            priors = np.clip(priors, 1e-12, 1.0)
            surprise = -np.log(priors)
            weighted_surprise = np.mean(surprise ** gamma, axis=axis) ** (1 / gamma)
            return np.exp(-weighted_surprise)
    
        return self.prior_df.apply(_asymmetric_evidence_aggregation, axis=1)

    
    def combine_prior_traffic_light_system(self):

        def _traffic_light(row, bad_threshold=0.3, warning_threshold=0.6):
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


    def combine_prior_mean(self, spoqc_tmp_folder):

        ddf_list = []
        for prior in self.priorset:
            ddf_list.append(
                    dd.read_parquet(
                    f"{spoqc_tmp_folder}/hqtr_output_qv_prob",
                    columns=[prior.name],
                    engine="pyarrow",
                    calculate_divisions=True,
                )
            )

        image_ddf = ddf_list[0]
        for i in range(1, len(ddf_list)):
            image_ddf += ddf_list[i]

        return image_ddf
