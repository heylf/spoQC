import numpy as np
import pandas as pd
import gzip
import re
import plotly.express as px
import plotly.graph_objects as go

from typing import Any

from .. import helperfuncs

# data from https://www.gencodegenes.org/human/
def parse_gtf(file_path: str) -> None:
    """
    Parses a GTF file and creates a DataFrame with gene symbols and their biotypes.

    Example:
        >>> parse_gtf('path/to/annotation/gencode.v46.annotation.gtf.gz')
        dict({'Gen_A': 'lncRNA', 'Gen_B': 'protein_coding', ...})

    Args:
        file_path (str): Path to the reference annotation (gtf.gz).

    Return
        dict({str: str}): dictionary mapping gene names to biotypes
    
    """


    # Columns to keep from the GTF file
    columns = ["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
    
    # Load GTF file into a pandas DataFrame
    print('read in gtf')
    data = []
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue  # Skip header lines
            fields = line.strip().split('\t')
            data.append(fields)
    
    print('convert to df')
    # Convert to pandas DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Filter for gene features only
    gene_df = df[df['feature'] == 'gene']
    
    # Extract gene name and biotype from the "attribute" column
    gene_df['gene_name'] = gene_df['attribute'].str.extract(r'gene_name "([^"]+)"')
    gene_df['gene_biotype'] = gene_df['attribute'].str.extract(r'gene_type "([^"]+)"')
    
    print('built dict')
    # Create a dictionary mapping gene names to biotypes
    gene_biotype_dict = dict(zip(gene_df['gene_name'], gene_df['gene_biotype']))
    
    return gene_biotype_dict


def get_rna_type(gene_symbol: str, gene_biotype_dict: dict) -> str:
    """
    Get biotype from a gene symbol.

    Example:
        >>> get_rna_type('Gen_A', gene_biotype_dict)
        protein_coding

    Args:
        gene_symbol (str): Gene symbol.
        gene_biotype_dict (dict): dictionary mapping gene names to biotypes.

    Return
        str: Biotype for the gene symbol.
    
    """
    return gene_biotype_dict.get(gene_symbol, "Unknown")


def transcriptqc(sdata, figure_path, annotation_file, key_transcripts):

    timer = helperfuncs.Timer()

    # If a transcript is NOT associated with a cell, it will get -1. 
    # If it is associated with a cell, this column will have a positive integer cell ID.
    cell_id = list(sdata.points[key_transcripts]['cell_id'])
    in_nucleus = list(sdata.points[key_transcripts]['overlaps_nucleus'])

    # Where are those transcripts located (more in the nuceus, cytoplasm)?
    # Pie chart for transcripts in cytoplasm, in nucleus, outside cell
    location_list = np.array(['Outside Cell' if x == -1 else 'In Cell' for x in cell_id])

    for i in range(0, len(cell_id)):

        if ( location_list[i] == 'In Cell' and in_nucleus[i] == 0 ):
            location_list[i] = 'In Nucleus'

    counter_in_cell = len([x for x in cell_id if x != -1])
    counter_outside_cell = len([x for x in cell_id if x == -1])
    counter_in_nucleus = len([x for x in in_nucleus if x == 0])

    d = pd.DataFrame({'total': ['Total'] * 3,
        'in_vs_out_cell': ['In Cell', 'In Cell', 'Outside Cell'],
        'in_nucleus': ['Nucleus', 'Cytoplasm', 'Outside Cell'],
        'counts': [counter_in_nucleus, counter_in_cell, counter_outside_cell]})

    fig = px.sunburst(d, path=['total', 'in_vs_out_cell', 'in_nucleus'], values='counts')
    fig.update_traces(textinfo="label+value+percent entry", textfont=dict(size=18))
    fig.update_traces(marker_colors=["lightblue","lightgreen","red","green","red"])
    fig.write_html(f"{figure_path}/transcript_location_pie.html")
    fig.write_image(f"{figure_path}/transcript_location_pie.png", scale=3)
    fig.write_image(f"{figure_path}/transcript_location_pie.pdf", scale=3)

    gene_biotype_dict = parse_gtf(f"{annotation_file}")

    rna_types = [get_rna_type(var, gene_biotype_dict) for var in list(sdata['table'].var_names)]

    df = pd.Series(rna_types).value_counts()
    df = pd.DataFrame(df)

    transcript_type_colors = helperfuncs.generate_distinct_colors(len(df))
    df['colors'] = transcript_type_colors

    rna_types_sdata = np.array([get_rna_type(var, gene_biotype_dict) \
                                for var in list(sdata.points[key_transcripts]['feature_name'])])

    # Pie chart for transcript types-
    fig = go.Figure(data=[go.Pie(labels=df.index, values=df['count'], marker=dict(colors=df['colors']))])
    fig.update_traces(textfont=dict(size=18))
    fig.write_html(f"{figure_path}/transctipt_type_pie.html")
    fig.write_image(f"{figure_path}/transctipt_type_pie.png", scale=3)
    fig.write_image(f"{figure_path}/transctipt_type_pie.pdf", scale=3)

    df = sdata[key_transcripts].compute()

    df['location'] = location_list
    df['feature_type'] = rna_types_sdata

    timer.start()
    helperfuncs.plot_scatter_density_by_category_df(df, 'location', figure_path, '1', None, None, 1)
    timer.stop()

    timer.start()
    helperfuncs.plot_scatter_density_by_category_df(df, 'feature_type', figure_path, '1', None, None, 1)
    timer.stop()

    # Phred-scaled quality value (Q-Score) estimating the probability of incorrect call
    timer.start()
    helperfuncs.plot_scatter_density_df(df, figure_path, 'qv', 'qv', 'qv', None, 'Transcript Quality Density', 1)
    timer.stop()


def negativeprobeqc(sdata: Any, figure_path: str, key_transcripts: str) -> None:
    """
    Perform quality control on negative probes by visualizing the density of negative probes 
    on a scatter plot and a kernel density estimate (KDE) plot.

    Args:
        figure_path (str): Path where the generated figure will be saved.
        sdata (Any): Spatial data containing the feature name and coordinates (x, y).
        key_transcripts (str): The key in the spatial data corresponding to transcript data.

    Returns:
        None: Saves the generated plot as a PNG file in the specified path.
    """

    df = sdata[key_transcripts].compute()

    match_neg_probes = [bool(re.compile('NegControlCodeword').match(x)) for x in list(df['feature_name'])]

    df['neg_probes'] = match_neg_probes

    helperfuncs.plot_scatter_density_df(df[df['neg_probes'] == True], figure_path, 
                                        'neg_probes', 'neg_probes', None, ['black'],
                                        'Density of negative probes')


def transcriptz(sdata: Any, figure_path: str, key_transcripts: str) -> None:
    """
    Create and save histograms for the distribution of the 'z' values from the spatial transcriptomics data.

    Args:
        figure_path (str): Path to save the generated figures in HTML format.
        sdata (Any): Spatial transcriptomics data, containing points with 'x', 'y', 'z' coordinates and associated sample names.
        key_transcripts (str): The key to access the transcript data in the `sdata`.

    Returns:
        None: Saves the generated plots as an HTML file in the specified path.
    """

    timer = helperfuncs.Timer()

    df = pd.DataFrame({
        'x': list(sdata.points[key_transcripts]['x']),
        'y': list(sdata.points[key_transcripts]['y']),
        'z': list(sdata.points[key_transcripts]['z']),
        'sample': ['sampleone'] * len(list(sdata.points[key_transcripts]['z']))
    })

    figures = []

    timer.start()
    fig = px.histogram(df, x='z', nbins=100, width=800, height=800)
    fig.update_layout(
        title=f"Total distribution of z for all samples"
    )
    helperfuncs.apply_general_plotly_layout(fig, True)
    figures.append(fig)
    fig.write_image(f"{figure_path}/histogram_z_total.png", scale=3)
    fig.write_image(f"{figure_path}/histogram_z_total.pdf", scale=3)
    timer.stop()

    # Create a histogram trace for each sample
    traces = []
    samples = list(set(df['sample']))
    for sample in samples:
        sample_data = df[df['sample'] == sample]
        trace = go.Histogram(x=sample_data['z'], nbinsx=100, name=sample)
        traces.append(trace)

    # Create the merged plot
    fig = go.Figure(data=traces)

    # Update layout
    fig.update_layout(
        title=f"Distribution of z for all samples",
        showlegend=True,
        height=800,
        margin=dict(t=150)  # Add more space between title and plot
    )

    fig.update_layout(
        dict(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=["visible", "legendonly"],
                            label="Deselect All",
                            method="restyle"
                        ),
                        dict(
                            args=["visible", True],
                            label="Select All",
                            method="restyle"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=False,
                    x=1,
                    xanchor="right",
                    y=1.1,
                    yanchor="top"
                ),
            ]
        )
    )

    fig.update_layout(
        xaxis_title='z',
        yaxis_title="Frequency"
    )

    helperfuncs.apply_general_plotly_layout(fig, True)

    figures.append(fig)
    fig.write_image(f"{figure_path}/histogram_z_all.png", scale=3)
    fig.write_image(f"{figure_path}/histogram_z_all.pdf", scale=3)

    with open(f'{figure_path}/transcript_z.html', 'w') as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    timer.start()
    helperfuncs.plot_scatter_density_df(df, figure_path, 'z', 'z', 'z', None, None)
    timer.stop()



def get_low_qc_transcript_count(transcript_df, sdata, qv_tresh, figure_path):

    transcript_df = transcript_df.loc[transcript_df['qv'] < qv_tresh]
    cell_id_counts = transcript_df['cell_id'].value_counts()
    cell_id_counts.index.name = 'index'
    cell_id_counts = cell_id_counts[cell_id_counts.index != -1] # Remove the id -1
    cell_id_counts_df = cell_id_counts.sort_index().reset_index()
    cell_id_counts_df.columns = ['index', 'count']
    cell_id_counts_df['index'] = cell_id_counts_df['index'].astype(str)

    # Join and fill missing counts with 0
    merged_df = pd.merge(sdata['table'].obs, cell_id_counts_df, on='index', how='left')
    merged_df['count'] = merged_df['count'].fillna(0).astype(int)
    merged_df = merged_df.rename(columns={'count': 'num_low_qc_transcript'})

    sdata['table'].obs['num_low_qc_transcript'] = np.array(merged_df['num_low_qc_transcript'])
    helperfuncs.plot_scatter_density(
        sdata['table'],
        figure_path,
        'num_low_qc_transcript',
        None,
        'num_low_qc_transcript',
        None,
        'Density low quality transcripts',
    )

