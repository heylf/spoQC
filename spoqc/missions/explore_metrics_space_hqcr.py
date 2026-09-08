import pkgutil
import importlib

from .. import core
from .. import helperfuncs
from .. import helperfuncs
from .. import metrics

def start_exploration(enterprise):

        metricset = []
        for module_info in pkgutil.iter_modules(metrics.segmentation.__path__):
            module_name = module_info.name
            full_name = f"{metrics.segmentation.__name__}.{module_name}"
            module = importlib.import_module(full_name)

            if hasattr(module, "init_metric"):
                metricset.append(module.init_metric(enterprise))
                print(f"Loaded metric: {module_name}")
            else:
                print(f"WARNING: {module_name} has no init_metric() function")

        hqcr_metricset = core.metric.MetricSet("hqcr", metricset)
        hqcr_metricset.calculate_metrics(enterprise.args.step)

        print("[NOTE] Write results")
        helperfuncs.sdata_obs_to_parquet(
            enterprise,
            enterprise.args.step,
            'hqcr'
        )
        print("[finish]")
