import plotly.express as px

from ... import helperfuncs
from ... import core

def calc_count_nuclei(sdata, figure_path):

    # This is from convexity calculations
    nuclei_counts = [len(x) for x in sdata['table'].obs['nuclei_idxs']]

    sdata['table'].obs['multi_nuceli'] = [True if x > 1 else False for x in nuclei_counts]
    sdata['table'].obs['wmulti_nuceli'] = [1 if x > 1 else 0 for x in nuclei_counts]

    sdata['table'].obs['nucleus_free'] = [True if x == 0 else False for x in nuclei_counts]
    sdata['table'].obs['wnucleus_free'] = [1 if x == 0 else 0 for x in nuclei_counts]

    sdata['table'].obs['nuceli_count'] = nuclei_counts

    figures = []

    fig = px.bar(sdata['table'].obs, x='nuceli_count')
    fig.update_layout(
        title=f"Nuclei counts for all samples"
    )
    helperfuncs.apply_general_plotly_layout(fig, True)
    figures.append(fig)
    fig.write_image(f"{figure_path}/barplot_nuceli_counts.png", scale=3)
    fig.write_image(f"{figure_path}/barplot_nuceli_counts.pdf", scale=3)

    with open(f'{figure_path}/nuceli_counts.html', 'w') as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    for cat in ['multi_nuceli', 'nucleus_free']:
        helperfuncs.plot_scatter(sdata['table'], figure_path, cat, None, 
                                 cat, ['lightblue', 'red'], f'Cells with {cat}')

        if ( len( [True for x in sdata['table'].obs[cat] if x == True ] ) > 10 ):
            helperfuncs.plot_density(sdata['table'], f'w{cat}', figure_path)
            helperfuncs.plot_scatter_density(sdata['table'], figure_path, cat, 
                                             cat, f'w{cat}', ['lightblue', 'red'], f'Cells with {cat}')


def init_metric(enterprise):

    # These have to be defined.
    metric_name = "nuceli_count"
    combined_metric_name = None
    needs_metrics = []
    step_when_it_is_calculated = ["cellqc", "all"]
    loaded_for_analysis = True
    loaded_for_visualization = True
    prior = False

    # These are given my your metric calc function.
    args = [enterprise.cargo.sdata, f"{enterprise.args.output_dir}/cellqc/"]
    kwargs = None

    metric = core.metric.Metric(
        calc_count_nuclei, 
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
