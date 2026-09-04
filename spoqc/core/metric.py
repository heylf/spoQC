import sys

from .. import helperfuncs

class Metric:
    def __init__(
            self, 
            calc_func,
            metric_name,
            *,
            step_when_it_is_calculated=[],
            combined_metric_name=None,
            needs_metrics = [],
            loaded_for_analysis=False,
            loaded_for_visualization=False,
            prior=False,
            args=None,
            kwargs=None,
        ):
        
        self._calc_func = calc_func
        
        self.args = args if args else []
        self.kwargs = kwargs if kwargs else {}

        self.name = metric_name
        self.combined_metric_name = combined_metric_name
        self.needs_metrics = needs_metrics
        self.step_when_it_is_calculated = step_when_it_is_calculated
        self.loaded_for_analysis = loaded_for_analysis
        self.loaded_for_visualization = loaded_for_visualization
        self.prior = prior

    def calculate(self):
        return self._calc_func(*self.args, **self.kwargs)
    

class MetricSet:
    def __init__(self, name, metricset):
        self.name = name

        if len(metricset) == 0:
            sys.exit("[ERROR] Metric set is empty")
        else:
            self.metricset = metricset

    def calculate_metrics(self, step):
        metrics_by_name = {metric.name: metric for metric in self.metricset}
        pending = {metric.name for metric in self.metricset if step in metric.step_when_it_is_calculated}

        while pending:
            # Go over each name in pending (e.g., void), check if void has metrics that it depends on
            # (e.g., doublet_score) and that still needs to be calculated.
            # If that is the case then do not add it to ready.
            ready = [name for name in pending if not set(metrics_by_name[name].needs_metrics) & pending]

            if len(ready) == 0 :
                sys.exit(f"[ERROR] Circular or unresolved metric dependency among: {sorted(pending)}")

            for name in ready:
                metric = metrics_by_name[name]
                print(f"[NOTE] Calculating {metric.name}")
                timer = helperfuncs.Timer()
                timer.start()
                metric.calculate()
                timer.stop()
                pending.discard(name)
