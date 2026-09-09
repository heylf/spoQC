from .. import helperfuncs

def start_exploration(enterprise):
    enterprise.hqcr_metricset.calculate_metrics(enterprise.args.step)
    
    print("[NOTE] Write results")
    helperfuncs.sdata_obs_to_parquet(
        enterprise,
        enterprise.args.step,
        'hqcr'
    )

    enterprise.hqpr_metricset.calculate_metrics(enterprise.args.step)