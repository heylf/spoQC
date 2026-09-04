import pkgutil
import importlib

from .. import core
from .. import helperfuncs
from .. import helperfuncs
from .. import metrics

def start_exploration(self):

        metricset = []
        for module_info in pkgutil.iter_modules(metrics.segmentation.__path__):
            module_name = module_info.name
            full_name = f"{metrics.segmentation.__name__}.{module_name}"
            module = importlib.import_module(full_name)

            if hasattr(module, "init_metric"):
                metricset.append(module.init_metric(self))
                print(f"Loaded metric: {module_name}")
            else:
                print(f"WARNING: {module_name} has no init_metric() function")

        hqcr_metricset = core.metric.MetricSet("hqcr", metricset)
        hqcr_metricset.calculate_metrics(self.args.step)

        print("[NOTE] Write results")
        helperfuncs.sdata_obs_to_parquet(
            self,
            self.args.step,
            'hqcr'
        )
        print("[finish]")
