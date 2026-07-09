import scanpy as sc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
import plotly.graph_objects as go

from scipy.stats import pearsonr

from ... import helperfuncs
from ... import general

def add_pearsoncorr_to_plotly(fig, x, y, xpos=0.05, ypos=0.95):
    """Calculate Pearson correlation and add as annotation to Plotly figure."""
    corr, p_value = pearsonr(x, y)
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=xpos, y=ypos,
        text=f'Pearson R: {corr:.2f}<br>P-value: {p_value:.2g}',
        showarrow=False,
        font=dict(size=14, color="black"),
        align="left"
    )

# Calculate confidence intervale function
def get_ci_df(count_series, mode, rna):
    min_num_cells = np.min(count_series.size())

    # Get number of groups
    num_group = len(rna.obs.groupby(mode).size())

    num_replicates = 1000

    mean_array = np.empty([0])
    labels = np.empty([0])

    for key, item in count_series:
        labels = np.append(labels, np.array([key]*num_replicates))

        tmp_mean_array = np.empty([num_replicates])
        counts = np.array(count_series.get_group(key).tolist())

        for i in range(0, num_replicates):
            tmp_mean_array[i] = np.mean(random.choices(counts, k=min_num_cells))

        mean_array = np.append(mean_array, tmp_mean_array)
    
    d = {'group': labels, 'values': mean_array}
    d = pd.DataFrame(data=d, index=[x for x in range(0, num_group*num_replicates)])
    return(d)

def prepate_qc(sdata, figure_path=None):

    ####################################################################################################################
    ###### PREPARE QC ##################################################################################################
    ####################################################################################################################
    print("[TASK] Prepare QC")

    rna = sdata['table']

    # Perform QC
    rna.var["mt"] = rna.var_names.str.startswith("MT-") # this will add mitochondrial QC to the general QC
    rna.var["ribo"] = rna.var_names.str.startswith("RPS") | rna.var_names.str.startswith("RPL") # add ribosomal RNA QC

    qc_vars = []
    rna_mt = rna[:,rna.var['mt']]
    if ( rna_mt.n_vars == 0 ):
        rna.obs['pct_counts_mt'] = [0.0] * rna.n_obs
    else:
        qc_vars.append("mt")

    rna_rb = rna[:,rna.var['ribo']]
    if ( rna_rb.n_vars == 0 ):
        rna.obs['pct_counts_ribo'] = [0.0] * rna.n_obs
    else:
        qc_vars.append("ribo")

    if ( len(qc_vars) == 0 ):
        sc.pp.calculate_qc_metrics(rna, percent_top=(10, 20, 50, 150), inplace=True)
    else:
        sc.pp.calculate_qc_metrics(rna, qc_vars=qc_vars, percent_top=(10, 20, 50, 150), inplace=True)

    cprobes = (
        rna.obs["control_probe_counts"].sum() / rna.obs["total_counts"].sum() * 100
    )
    cwords = (
        rna.obs["control_codeword_counts"].sum() / rna.obs["total_counts"].sum() * 100
    )

    if ( figure_path ):
        with open(f'{figure_path}/rawqc.txt', 'w') as f:
            f.write(f"Negative DNA probe count % : {cprobes} \n")
            f.write(f"Negative decoding count % : {cwords} \n")

    return rna


def calc_sc_metrics(sdata, figure_path, annotation_path, annotation_key):

    ####################################################################################################################
    ###### GLOBAL VARS and DIRECTORIES #################################################################################
    ####################################################################################################################

    THRESH_GENE_FILTER = 100
    THRESH_UMI_N_GENES = 500
    THRESH_MT = 20
    THRESH_RB = 10
    DPI = 300

    print("[START]")

    # Seeds (!!! DO NOT CHANGE THIS SEED !!!)
    seed = 123
    random.seed(seed)
    print(f"[NOTE] seed {seed}")

    rna = prepate_qc(sdata, figure_path=figure_path)
    general.normalizations.fill_nans_for_0_transcript_cells(rna)

    ####################################################################################################################
    ###### QC PLOTS RNA ################################################################################################
    ####################################################################################################################
    print("[TASK] Generate QC plots RNA")

    figures = []
    samples = list(set(rna.obs['sample']))

    # ------------------------------------------------------------------------------------------------------------------
    # UMI/Gene/MT/RB ---------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    for level in ["total_counts", "transcript_counts", "n_genes_by_counts", "cell_area", "nucleus_area",
                  "pct_counts_mt", "pct_counts_ribo"]:

        fig = px.histogram(
            rna.obs, x=level, nbins=100,
            labels={level: f"{level}"},
            width=800, height=800
        )
        fig.update_layout(title=f"Total distribution of {level} for all samples")
        helperfuncs.apply_general_plotly_layout(fig, True)
        figures.append(fig)
        fig.write_image(f'{figure_path}/histogram_{level}_total.png', scale=int(DPI/100))
        fig.write_image(f'{figure_path}/histogram_{level}_total.pdf', scale=int(DPI/100))

        if ( level not in ["pct_counts_mt", "pct_counts_ribo"] ):

            rna.obs[f'log_{level}'] = np.log10(rna.obs[level] + 1)

            # Create a histogram trace for each sample
            traces = []
            for sample in samples:
                sample_data = rna.obs[rna.obs['sample'] == sample]
                trace = go.Histogram(x=sample_data[f'log_{level}'], nbinsx=100, name=sample)
                traces.append(trace)

            # Create the merged plot
            fig = go.Figure(data=traces)

            # Update layout
            fig.update_layout(
                title=f"Distribution of log10 {level} for all samples",
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
                xaxis_title=f"log_{level}",
                yaxis_title="Frequency"
            )

            helperfuncs.apply_general_plotly_layout(fig, True)

            figures.append(fig)
            fig.write_image(f'{figure_path}/histogram_log10_{level}_all.png', scale=int(DPI/100))
            fig.write_image(f'{figure_path}/histogram_log10_{level}_all.pdf', scale=int(DPI/100))

        # Create a histogram trace for each sample
        traces = []
        for sample in samples:
            sample_data = rna.obs[rna.obs['sample'] == sample]
            trace = go.Histogram(x=sample_data[level], nbinsx=100, name=sample)
            traces.append(trace)

        # Create the merged plot
        fig = go.Figure(data=traces)

        # Update layout
        fig.update_layout(
            title=f"Distribution of {level} for all samples",
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
            xaxis_title=f"{level}",
            yaxis_title="Frequency"
        )

        helperfuncs.apply_general_plotly_layout(fig, True)

        figures.append(fig)
        fig.write_image(f'{figure_path}/histogram_{level}_all.png', scale=int(DPI/100))
        fig.write_image(f'{figure_path}/histogram_{level}_all.pdf', scale=int(DPI/100))

    # ----------------------------------------------------------------------------------------------------------------------
    # Others -------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    num_genes = len([1 for i in rna.var.n_cells_by_counts > min(THRESH_GENE_FILTER, np.ceil(0.01 * rna.n_obs)) if i == False])

    rna.var['log_n_cells_by_counts'] = np.log10(rna.var['n_cells_by_counts'] + 1)

    fig = px.histogram(
        rna.var,
        x="log_n_cells_by_counts",
        nbins=100,
        labels={"log_n_cells_by_counts": "log10 number of cells expressing > 0 (n reads)"},
        title=f"Distribution of the number of cells per gene. \n From total number of {rna.n_vars} genes," + \
              f"{num_genes} genes have < {THRESH_GENE_FILTER} cells.")
    fig.add_vline(x=np.log10(THRESH_GENE_FILTER))
    helperfuncs.apply_general_plotly_layout(fig, False)
    figures.append(fig)
    fig.write_image(f'{figure_path}/histogram_log_n_cells_by_counts.png', scale=int(DPI/100))
    fig.write_image(f'{figure_path}/histogram_log_n_cells_by_counts.pdf', scale=int(DPI/100))


    n_cells = 10_000
    if ( rna.n_obs < n_cells ):
        n_cells = rna.n_obs 

    bool_array = np.full(rna.n_obs, False)
    bool_array[:n_cells] = True
    np.random.shuffle(bool_array)

    fig = go.Figure()

    check_rna = rna.obs[bool_array]

    fig.add_trace(go.Scatter(
        x=check_rna["total_counts"],
        y=check_rna["n_genes_by_counts"],
        mode="markers",
        marker=dict(
            size=check_rna["pct_counts_ribo"],
            sizemode="diameter",
            sizeref=5,
            sizemin=0.1,
            color=check_rna["pct_counts_mt"],
            colorscale="bluered",
            cmin=0,
            cmax=100,
            colorbar=dict(
                title=dict(text="pct_counts_mt", side="right")
            ),
            showscale=True,
        ),
        text=[
            f"total_counts: {count}<br>n_genes_by_counts: {genes}<br>pct_counts_ribo: {ribo}<br>pct_counts_mt: {mt}"
            for count, genes, ribo, mt in zip(
                check_rna["total_counts"],
                check_rna["n_genes_by_counts"],
                np.round(check_rna["pct_counts_ribo"]),
                np.round(check_rna["pct_counts_mt"])
            )
        ]
    ))

    fig.add_shape(
        type="line",
        x0=THRESH_UMI_N_GENES,
        x1=THRESH_UMI_N_GENES,
        y1=max(check_rna["n_genes_by_counts"]),
        line=dict(
            color="black",
            width=1
        )
    )

    fig.add_shape(
        type="line",
        y0=THRESH_UMI_N_GENES,
        y1=THRESH_UMI_N_GENES,
        x1=max(check_rna["total_counts"]),
        line=dict(
            color="black",
            width=1
        )
    )

    fig.update_layout(
        title=f"Scatter plot of total_counts vs n_genes_by_counts vs pct_counts_mt vs pct_counts_ribo of {n_cells} cells. \
            Size of the diameter corresponds to the pct_counts_ribo.",
        xaxis=dict(
            type="log",
            title="log10 total_counts"
        ),
        yaxis=dict(
            type="log",
            title="log10 n_genes_by_counts"
        )
    )

    helperfuncs.apply_general_plotly_layout(fig, True)

    figures.append(fig)
    fig.write_image(f'{figure_path}/scatterplot_total_counts_vs_n_genes_by_counts.png', scale=int(DPI/100))
    fig.write_image(f'{figure_path}/scatterplot_total_counts_vs_n_genes_by_counts.pdf', scale=int(DPI/100))

    # nGENES = 30
    # idx = np.argsort(rna.var["n_cells_by_counts"])[-nGENES:]
    # expression = rna.X.todense()[:, idx]
    # genes = rna.var_names[idx]
    # df = pd.DataFrame(expression, columns=genes)
    # df_perct = df.div(np.array(rna.obs['total_counts']), axis=0)
    # # Calculate the median for each gene
    # gene_means = df_perct.mean()
    # # Sort the genes based on their median values
    # sorted_genes = gene_means.sort_values().index[::-1]
    # fig = go.Figure()
    # for gene in sorted_genes:
    #     fig.add_trace(go.Box(y=df_perct[gene], name=gene))
    # fig.update_layout(
    #     title=f"Boxplot of {nGENES} highest expressed genes",
    #     xaxis=dict(title="Genes"),
    #     yaxis=dict(title="% of total counts")
    # )
    # helperfuncs.apply_general_plotly_layout(fig, True)
    # figures.append(fig)
    # fig.write_image(f'{figure_path}/boxplot_expressed_genes.png', scale=int(DPI/100))

    ### PEARSON CORRELATION PLOTS
    for x in ['transcript_counts', 'n_genes_by_counts']:
        y = 'cell_area'
        
        hue = None
        if ( annotation_path ):
            hue = annotation_key
            celltypes = list(set(rna.obs[annotation_key]))

            for celltype in celltypes:
                celltype_sdata = rna.obs[rna.obs[annotation_key] == celltype]


                if ( len(celltype_sdata) > 10 ):
                    fig = px.scatter(
                        celltype_sdata,
                        x=x,
                        y=y,
                        title=f"Scatter Plot of {x} vs {y} ({celltype})",
                        opacity=0.7
                    )

                    add_pearsoncorr_to_plotly(fig, celltype_sdata[x], celltype_sdata[y])
                    
                    figures.append(fig)
                    fig.write_image(f"{figure_path}/scatterplot_pearsoncorr_{x}_{y}_{celltype}.png", scale=int(DPI/100))
                    fig.write_image(f"{figure_path}/scatterplot_pearsoncorr_{x}_{y}_{celltype}.pdf", scale=int(DPI/100))
                else:
                    print(f"[NOTE] Not enough cells to analyse correlation of {x} vs {y} for celltype {celltype}")

        fig = px.scatter(
            rna.obs,
            x=x,
            y=y,
            color=hue,
            title=f"Scatter Plot of {x} vs {y}",
            opacity=0.7
        )

        add_pearsoncorr_to_plotly(fig, rna.obs[x], rna.obs[y])

        figures.append(fig)
        fig.write_image(f"{figure_path}/scatterplot_pearsoncorr_{x}_{y}.png", scale=int(DPI/100))
        fig.write_image(f"{figure_path}/scatterplot_pearsoncorr_{x}_{y}.pdf", scale=int(DPI/100))

    # ------------------------------------------------------------------------------------------------------------------
    # Generate HTML ----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    with open(f'{figure_path}/rna_qc_sample_mqc.html', 'w') as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))