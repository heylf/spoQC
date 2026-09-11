import pandas as pd
import sys


from .. import helperfuncs

# class HQCR:
#     def __init__(
#             self, 
#         ):

class HqcrSet:
    def __init__(self, enterprise):
        all_metrics_for_comp_list = []
        for metric in enterprise.hqcr_metricset.metricset:
            all_metrics_for_comp_list += metric.submetrics
        
        self.metrics = all_metrics_for_comp_list
        self.cell_clustering_df = None
        self.cell_clustering_adata = None


    # TODO 'convexhull_all_trnascripts': sdata['table'].obs['convexhull_all_trnascripts'],
    def load_cell_clustering_df(self, enterprise):
        print(f"[NOTE] The following metrics will be used for further HQCR analysis {self.metrics}")
        helperfuncs.read_sdata_parquet_tmp_files(enterprise.cargo.sdata, enterprise.args.tmp_dir, 'hqcr')
        adata_obs = enterprise.cargo.sdata['table'].obs
        cell_clustering_df = pd.DataFrame({})
        for metric in self.metrics: 
            cell_clustering_df[metric] = adata_obs[metric]

        # Check for NaNs
        any_nas = cell_clustering_df.isna().sum().sum()
        if any_nas > 0:
            sys.exit(f"[ERROR] NaNs in columns for HQCR cell clustering df: {any_nas}")

        self.cell_clustering_df = cell_clustering_df


    def load_cell_clustering_adata(self, enterprise):

        # This adata is just used for the clustering and not for any other visualization and so on.
        qc_domains_adata = enterprise.cargo.sdata['table']
        # Here it does not matter which cols as long as I have the same number of cols as in X.
        # Only matrix X matters which will be used for clustering later.
        qc_domains_adata = qc_domains_adata[:,0:len(self.metrics)]
        qc_domains_adata.X = self.cell_clustering_df

        self.cell_clustering_adata = qc_domains_adata


class HqprSet:
    def __init__(self, enterprise):
        all_metrics_for_comp_list = []
        for metric in enterprise.hqpr_metricset.metricset:
            all_metrics_for_comp_list += metric.submetrics

        self.metrics = all_metrics_for_comp_list
        print(f"[NOTE] The following metrics will be used for further HQPR analysis {self.metrics}")


class HqtrSet:
    def __init__(self, enterprise):
        all_metrics_for_comp_list = []
        for metric in enterprise.hqtr_metricset.metricset:
            all_metrics_for_comp_list += metric.submetrics

        self.metrics = all_metrics_for_comp_list
        print(f"[NOTE] The following metrics will be used for further HQTR analysis {self.metrics}")