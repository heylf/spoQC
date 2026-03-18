import pandas as pd
import numpy as np
import plotly.express as px

from typing import Any, Dict

from .. import helperfuncs

def bubbleqc(sdata: Dict[str, Any], figure_path: str, bubble_key: str) -> None:
    """
    Analyze and visualize cell geometries to identify potential bubble-like structures in spatial data.

    This function calculates the thinness ratio for cell geometries, identifies bubble-like cells based on area and 
    thinness thresholds, and generates visualizations to summarize the findings. Results are saved to an HTML file, 
    and a boolean indicator of bubble status is added to the input data.

    Args:
        sdata (dict):
            A dictionary containing spatial data, including cell geometries and associated metadata. 
            It should have the following keys:
              - 'table': Contains a pandas DataFrame with observation data.
              - The key specified by `bubble_key`, which contains cell boundary geometries.
        figure_path (str):
            Path to the directory where visualizations will be saved.
        bubble_key (str):
            Key in `sdata` that points to the DataFrame column containing cell boundary geometries.

    Raises:
        KeyError: If required keys are not present in `sdata` or the specified bubble_key is missing.
        ValueError: If geometries are invalid or if areas/thinness scores cannot be computed.

    Returns:
        None

    Notes:
        Thinness ratio (T) is defined as:
        T = (4 * pi * measured area) / (circumference^2).
        - A perfect circle has T = 1.
        - Values close to 0 indicate elongated or irregular shapes.

        Empirical thresholds used:
        - Area > 3rd quartile of all areas: Likely to be a bubble.
        - Thinness score > 0.1: Resembles a circular shape.

    """

    DPI = 300

    cellboundaries = sdata[bubble_key]

    n_obs = sdata['table'].n_obs
    thinness_scores = [-1] * n_obs
    areas = [-1] * n_obs

    for i in range(0, n_obs):
        areas[i] = cellboundaries.iloc[i][0].area
        circumference = cellboundaries.iloc[i][0].length
        thinness_scores[i] = ( 4*np.pi*areas[i] ) / ((circumference) ** 2)

    df = pd.DataFrame({
        'thinness_score': thinness_scores,
        'log10_thinness_score': [np.log10(x) for x in thinness_scores],
        'cell_areas': areas,
        'log10_cell_areas': [np.log10(x) for x in areas]
    })

    figures = []

    fig = px.histogram(df, x='log10_thinness_score', nbins=100, width=800, height=800)
    
    fig.update_layout(
        title=f"Total distribution of log10_thinness_score for all samples"
    )

    helperfuncs.apply_general_plotly_layout(fig, True)
    
    figures.append(fig)
    fig.write_image(f'{figure_path}/histogram_log10_thinness_score.png', scale=int(DPI/100))

    fig = px.violin(df, x='log10_cell_areas', width=800, height=800)
    
    fig.update_layout(
        title=f"Cell area"
    ) 

    helperfuncs.apply_general_plotly_layout(fig, True)

    figures.append(fig)
    fig.write_image(f'{figure_path}/violinplot_log10_cell_areas.png', scale=int(DPI/100))

    with open(f'{figure_path}/bubble.html', 'w') as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    quartiles = np.quantile(areas, q=[.25, .5, .75, .9])

    # Filter for areas that are bigger than 3rd quartile.
    # If the area is huge it is most likely a bubble.
    filter_areas = [True if x > quartiles[3] else False for x in areas]

    # We consider an empirical thinness threshold of 0.1.
    # This is where cells start to look like circles.
    filter_thinness_score = [True if x > 0.1 else False for x in thinness_scores]

    sdata['table'].obs['bubble'] = [x and y for x, y in zip(filter_areas, filter_thinness_score)]
    sdata['table'].obs['thinness_score'] = thinness_scores

    num_bubbles = list(sdata['table'].obs['bubble']).count(True)
    helperfuncs.plot_scatter(sdata['table'], figure_path, 'bubbleqc', None, 'bubble', 
                             ['lightblue', 'red'], f'Number of bubbles {num_bubbles}')
