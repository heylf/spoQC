from .. import helperfuncs

def start_exploration(enterprise):
    calculated_metrics = enterprise.hqcr_metricset.calculate_metrics(enterprise.args.step)
    
    # Have to do this for HQCR because some steps of spoQC calculate multiple HQCR metrics
    if len(calculated_metrics) != 0 :
        print("[NOTE] Write results")
        helperfuncs.sdata_obs_to_parquet(
            enterprise,
            enterprise.args.step,
            'hqcr'
        )

    _ = enterprise.hqpr_metricset.calculate_metrics(enterprise.args.step)
    _ = enterprise.hqtr_metricset.calculate_metrics(enterprise.args.step)