# In[]
import os
import re
import base64

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def format_description(text):
    """Render a lightweight markdown-like description block as semantic HTML.

    Blank-line-separated blocks become paragraphs; a block whose every line
    starts with '- ' becomes a bulleted list; '**bold**' becomes <strong>.
    """
    def inline(s):
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)

    blocks = re.split(r"\n\s*\n", text.strip())
    html_blocks = []
    for block in blocks:
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        if not lines:
            continue
        if all(line.startswith("- ") for line in lines):
            items = "".join(f"<li>{inline(line[2:])}</li>" for line in lines)
            html_blocks.append(f"<ul>{items}</ul>")
        else:
            html_blocks.append(f"<p>{inline(' '.join(lines))}</p>")
    return "\n".join(html_blocks)


def render_image_gallery(
        header, 
        image_filename, 
        stainings,
        performed_stainings,
        folder_path,
        *,
        folder_path_continue=None, 
        description=None
    ):
    html = f"""
    <h2>{header}</h2>
    """
    if description:
        html += format_description(description)
    for s in performed_stainings:
        size = 24
        if len(performed_stainings) == 1:
            size = 45
        img_file = f"{folder_path}/{s}/{image_filename}"
        if folder_path_continue:
            img_file = f"{folder_path}/{folder_path_continue}/{s}/{image_filename}"
        img_data = image_to_base64(img_file)
        html += f"""
        <div style="display:inline-block; max-width:{size}%;">
            <span style="display:block; background:rgba(0,0,0,0.6); color:#fff; font-size:12px;
            padding:2px 6px; border-radius:3px; width:fit-content;">{stainings[s]}</span>
            <img src="data:image/png;base64,{img_data}" style="width:100%; display:block;">
        </div>
        """
    return html


def render_numbered_image_gallery(header, folder_path, file_prefix, description=None):
    pattern = re.compile(rf"^{re.escape(file_prefix)}_(\w+)\.png$")
    matches = []
    if os.path.exists(folder_path):
        for fname in os.listdir(folder_path):
            m = pattern.match(fname)
            if m:
                matches.append((m.group(1), fname))
    matches.sort(key=lambda x: (0, int(x[0])) if x[0].isdigit() else (1, x[0]))

    html = f"""
    <h2>{header} ({len(matches)})</h2>
    """
    if description:
        html += format_description(description)
    size = 24
    if len(matches) == 1:
        size = 45
    for cluster_id, fname in matches:
        img_data = image_to_base64(f"{folder_path}/{fname}")
        html += f"""
        <div style="display:inline-block; max-width:{size}%;">
            <img src="data:image/png;base64,{img_data}" style="width:100%; display:block;">
        </div>
        """
    return html


def render_numbered_paired_image_gallery(folder_path, folder_path_2, file_prefix, file_prefix_2, description=None):
    html = ""
    pattern = re.compile(rf"^{re.escape(file_prefix)}_(\d+)_filtered_out\.png$")
    matches = []
    if os.path.exists(folder_path):
        for fname in os.listdir(folder_path):
            m = pattern.match(fname)
            if m:
                matches.append((m.group(1), fname))
    matches.sort(key=lambda x: (0, int(x[0])) if x[0].isdigit() else (1, x[0]))

    pattern_2 = re.compile(rf"^{re.escape(file_prefix_2)}_(\d+)_filtered_out\.png$")
    matches_2 = []
    if os.path.exists(folder_path_2):
        for fname in os.listdir(folder_path_2):
            m = pattern_2.match(fname)
            if m:
                matches_2.append((m.group(1), fname))
    matches_2.sort(key=lambda x: (0, int(x[0])) if x[0].isdigit() else (1, x[0]))

    if description:
        html += format_description(description)

    for i in range(0, len(matches)):
        html += f"""
        <h2>Filter HQPRs {i}</h2>
        """

        img_first = image_to_base64(f"{folder_path}/{matches[i][1]}")
        img_second = image_to_base64(f"{folder_path_2}/{matches_2[i][1]}")
        html += f"""
        <div style="display:inline-block; max-width:100%;">
            <img src="data:image/png;base64,{img_first}" style="width:45%;">
            <img src="data:image/png;base64,{img_second}" style="width:45%;">
        </div>
        """
    return html


# In[]
def create_final_report(figure_path, stainings):

    performed_stainings = [int(x) for x in os.listdir(f'{figure_path}/combine_masks')]

    ####################################################################################################################
    # Start
    ####################################################################################################################

    html_overview = ""

    ####################################################################################################################
    # First page
    ####################################################################################################################

    ##########
    # Overview
    ##########

    html_overview += f"""
    <h1>High quality regions (HQRs) for HQCR (high quality cell regions), HQPR (high quality pixel regions), and HQTR (high quality transcript regions)</h1>
    """

    folder_path = f'{figure_path}/combine_masks'

    description = """
    **Summary:** Dark (dim) regions indicate areas affected by low quality. Large dark patches across the slide suggest a systematic quality issue.

    **Details:** Combined belief (probability) masks for all three data modalities: HQCR, HQPR, and HQTR. Each modality is projected into pixel coordinates so that it can be represented as an image. Brighter pixels indicate a higher probability that the underlying observation is of good quality. Dark (dim) regions therefore highlight areas affected by low quality.
    """
    html_overview += render_image_gallery("Combined beliefs", "imageplot_combined_beliefs.png", stainings, 
                                          performed_stainings, folder_path, description=description)

    description = """
    **Summary:** Dark (dim) regions indicate areas affected by low quality. Large dark patches across the slide suggest a systematic quality issue.

    **Details:** Beliefs for all three data modalities (HQCR, HQPR, and HQTR) are thresholded at 0.5 (1 if > 0.5, 0 otherwise). For each pixel, spoQC sums the resulting binary masks across modalities.

    **Categories:**

    - No mask: the pixel is covered by 0 masks.
    - 1 mask: the pixel is covered by exactly 1 mask.
    - 2 masks: the pixel is covered by 2 masks.
    - All masks: the pixel is covered by all 3 masks.
    """
    html_overview += render_image_gallery("Combined masks", "imageplot_combined_masks.png", stainings,
                                          performed_stainings, folder_path, description=description)


    description = """
    **Summary:** Percentage overlap between data modalities. A modality with markedly lower overlap than the others suggests a modality-specific quality issue.

    **Details:** Using the integer masks described above, spoQC reports the percentage overlap for each data modality.
    """
    html_overview += render_image_gallery("Venndiagram of overlapping HQRs", "venn_combined_masks.png", stainings,
                                          performed_stainings, folder_path, description=description)


    html_overview += f"""
    <h1>High quality regions (HQRs) with spatial dependencies</h1>
    """

    description = """
    **Summary:** Dark (dim) regions indicate areas affected by low quality. Large dark patches across the slide suggest a systematic quality issue.

    **Details:** Unlike the belief masks above, this analysis incorporates spatial information: the quality of a pixel is influenced by the quality of its neighbourhood. Check the spoQC documentation for further details.
    """
    html_overview += render_image_gallery("Combined beliefs with spatial dependencies",
                                "imageplot_combined_beliefs_smoothed.png", stainings, performed_stainings, folder_path,
                                description=description)

    description = """
    **Summary:** Dark (dim) regions indicate areas affected by low quality. Large dark patches across the slide suggest a systematic quality issue.

    **Details:** Labels are taken directly from spoQC's Markov random field model, which predicts the most likely latent state for each pixel, so no additional threshold is required. As above, this incorporates spatial information into the integrated quality mask, and spoQC sums the resulting binary masks across modalities for each pixel.

    **Categories:**

    - No mask: the pixel is covered by 0 masks.
    - 1 mask: the pixel is covered by exactly 1 mask.
    - 2 masks: the pixel is covered by 2 masks.
    - All masks: the pixel is covered by all 3 masks.
    """
    html_overview += render_image_gallery("Combined masks with spatial dependencies",
                                "imageplot_combined_masks_smoothed.png", stainings, performed_stainings, folder_path,
                                description=description)


    description = """
    **Summary:** Percentage overlap between data modalities after incorporating spatial dependencies. A modality with markedly lower overlap than the others suggests a modality-specific quality issue.

    **Details:** Using the spatially smoothed integer masks described above, spoQC reports the percentage overlap for each data modality.
    """
    html_overview += render_image_gallery("Venndiagram of overlapping HQRs with spatial dependencies",
                                "venn_combined_masks_smoothed.png", stainings, performed_stainings, folder_path,
                                description=description)

    ##########
    # In-depth
    ##########

    img_funkyheatmap = image_to_base64(f"{figure_path}/analysis/overview/funkyheatmap/funkyheatmap_1.png")

    description_summary = """
    **Summary:** A summary of spoQC's beliefs and metrics for each data modality. The y-axis shows the Leiden clusters generated from the provided cell type annotation.

    Inspect the HQCR, HQPR, and HQTR beliefs first. A cluster with markedly lower beliefs than the others indicates that it may contain quality-impacted cells; if such a cluster is present, inspect its individual metrics below for a possible explanation. For example, low HQCR (cell segmentation) beliefs may result from a low transcript count or from a high proportion of cells close to vertical doublet events.
    """

    additional_description_summary = """
    **Details:** spoQC performs an additional Leiden clustering at a resolution chosen to yield three more clusters than the provided cell type annotation. The cell type annotation is either supplied by the user or generated by an initial Leiden clustering with an spoQC-optimized resolution. spoQC adds three clusters on the hypothesis that the data contain cell type clusters affected by quality. Further detail can be found in the <u>All HQCR metrics</u>, <u>All HQPR metrics</u>, and <u>All HQTR metrics</u> panels.
    """
    html_overview += f"""
    <h1>Summary of spoQC</h1>
    {format_description(description_summary)}
    {format_description(additional_description_summary)}
    <img src="data:image/png;base64,{img_funkyheatmap}" style="max-width:80%;">
    """

    description_umap = """
    **Summary:** UMAP embeddings of the cells, colored by the applied cell type annotation (left) and by the Leiden clustering derived from that annotation (right). These plots support further investigation of the Leiden clusters flagged above.

    For example, low HQCR (cell segmentation) beliefs may result from a low transcript count or from a high proportion of cells close to vertical doublet events, and may manifest as a Leiden cluster that separates markedly from the rest of the cell population; a further indication that the cluster is quality-impacted.
    """

    reference_description_umap = """
    **See also:** the <u>Spatial plots Leiden clusters</u> panel shows the spatial distribution of each Leiden cluster, and the <u>Spatial plots annotation clusters</u> panel shows the spatial distribution of each cell type cluster.
    """

    with open(f"{figure_path}/analysis/overview/umap/umap_plot_celltype.html") as f:
        html_annotation = f.read()
    with open(f"{figure_path}/analysis/overview/umap/umap_plot_leiden.html") as f:
        html_leiden = f.read()
    html_overview += f"""
    <h2>Annotation and Leiden clustering</h2>
    {format_description(description_umap)}
    {format_description(reference_description_umap)}
    <div style="display:inline-block; vertical-align:top; width:48%;">{html_annotation}</div>
    <div style="display:inline-block; vertical-align:top; width:48%;">{html_leiden}</div>
    """

    description_filter = """
    **Summary:** UMAP embedding of the Leiden clustering (left) and the corresponding spatial plot of each cell (right). Cells marked in black are excluded by spoQC's HQR filter. Previously identified quality-impacted clusters should be picked up by this filter.

    **Details:** spoQC filters by HQR across all three data modalities. A cell is excluded (marked as low quality) if any of the following hold:

    - HQCR beliefs < 0.45
    - mean informative HQPR beliefs < 0.45
    - mean informative HQTR beliefs < 0.45

    Because a segmented cell typically spans more than one transcript, spoQC calculates the mean informative belief (mean of values > 0.2) for this modality, which avoids the low-value-inflation bias that would otherwise correlate with cell size.
    """

    reference_description_filter = """
    **See also:** the <u>Individual HQR filters</u> panel shows the individual HQCR, HQPR, and HQTR filters that are combined to produce the HQR filter.
    """
    img_ump_hqr = image_to_base64(f"{figure_path}/analysis/overview/umap/umap_plot_hqr_filtered_out.png")
    img_spa_hqr = image_to_base64(f"{figure_path}/analysis/overview/scatterplot/scatterplot_hqr_filtered_out.png")
    html_overview += f"""
    <h2>Filter HQRs</h2>
    {format_description(description_filter)}
    {format_description(reference_description_filter)}
    <img src="data:image/png;base64,{img_ump_hqr}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_spa_hqr}" style="max-width:45%;">
    """

    description_bar_control_probes = """
    **Summary:** Percentage of cells with a control probe count greater than 0, per cell cluster. This helps to more clearly identify quality-impacted cell clusters. A higher percentage for a given cluster indicates poorer quality.
    """
    img_bar_cpc = image_to_base64(f"{figure_path}/analysis/overview/barplot/barplot_pct_control_probe_counts.png")
    html_overview += f"""
    <h2>Control probe counts per cluster</h2>
    {format_description(description_bar_control_probes)}
    <img src="data:image/png;base64,{img_bar_cpc}" style="max-width:45%;">
    """

    description_doublet = """
    **Summary:** Percentage of cells classified as being close to a doublet event, per cell cluster. This helps to more clearly identify quality-impacted cell clusters. A higher percentage for a given cluster indicates poorer quality.
    """
    img_bar_dc = image_to_base64(f"{figure_path}/analysis/overview/barplot/barplot_pct_doublet_celltype.png")
    img_bar_dl = image_to_base64(f"{figure_path}/analysis/overview/barplot/barplot_pct_doublet_leiden.png")
    html_overview += f"""
    <h2>Doublet counts per cluster</h2>
    {format_description(description_doublet)}
    <img src="data:image/png;base64,{img_bar_dc}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_bar_dl}" style="max-width:45%;">
    """

    description_nucelus_free = """
    **Summary:** Percentage of cells classified as nucleus-free, per cell cluster. This helps to more clearly identify quality-impacted cell clusters. A higher percentage for a given cluster indicates poorer quality.
    """
    img_bar_nfc = image_to_base64(f"{figure_path}/analysis/overview/barplot/barplot_pct_nucleus_free_celltype.png")
    img_bar_nfl = image_to_base64(f"{figure_path}/analysis/overview/barplot/barplot_pct_nucleus_free_leiden.png")
    html_overview += f"""
    <h2>Nucleus-free cell counts per cluster</h2>
    {format_description(description_nucelus_free)}
    <img src="data:image/png;base64,{img_bar_nfc}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_bar_nfl}" style="max-width:45%;">
    """

    ##########
    # Individual but important
    ##########
    html_overview += f"""
    <h1>Individual plots worth inspecting</h1>
    """

    description = """
    **Summary:** Spatial density plots of the cells, weighted by cell convexity (left) or by the mean nucleus convexity per cell (right); the mean is used because a cell may contain more than one nucleus. Low-density regions suggest potential segmentation issues.

    **Details:** False (black): the cell or nucleus is non-convex (convexity ≤ 0.5) and may have an irregular shape. True (blue): the cell or nucleus is convex (convexity > 0.5). We also observed that a high proportion of highly convex cells and nuclei can signal segmentation issues, related to spoQC's approach for correcting invalid geometries. Check whether high-density regions overlap with high-density invalid-geometry regions (below).
    """
    img_cell = image_to_base64(f"{figure_path}/cellqc/scatterplot_densityplot_convexity_cell_convexity_metric_cell.png")
    img_nucl = image_to_base64(f"{figure_path}/cellqc/scatterplot_densityplot_convexity_nuclei_convexity_mean_nuceli.png")
    html_overview += f"""
    <h2>Convexity of cell and nucleus shapes</h2>
    {format_description(description)}
    <img src="data:image/png;base64,{img_cell}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_nucl}" style="max-width:45%;">
    """

    description = """
    **Summary:** Density map of invalid cell or nucleus geometries. High-density regions indicate potential segmentation issues.
    """
    img_cell = image_to_base64(f"{figure_path}/generalqc/scatterplot_densityplot_invalid_cell_geometry.png")
    img_nucl = image_to_base64(f"{figure_path}/generalqc/scatterplot_densityplot_invalid_nucleus_geometry.png")
    html_overview += f"""
    <h2>Invalid cell and nucleus geometries</h2>
    {format_description(description)}
    <img src="data:image/png;base64,{img_cell}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_nucl}" style="max-width:45%;">
    """

    description = """
    **Summary:** Spatial density plot of the cells, weighted by the number of low-quality (QV < 20) transcripts. High-density regions indicate quality issues.

    **Details:** QV is the Phred-scaled quality value (Q-score) defined by 10x Genomics, estimating the probability of an incorrect base call. A QV threshold of 20 is used, following the 10x Genomics convention.
    """
    img_low_qv_trans = image_to_base64(f"{figure_path}/cellqc/scatterplot_densityplot_num_low_qc_transcript.png")
    html_overview += f"""
    <h2>Low quality transcripts (QV &lt; 20)</h2>
    {format_description(description)}
    <img src="data:image/png;base64,{img_low_qv_trans}" style="max-width:45%;">
    """

    description = """
    **Summary:** Spatial density plot of the cells, weighted by the number of negative probes. High-density regions indicate quality issues.
    """
    img_neg_probes = image_to_base64(f"{figure_path}/transcriptqc/scatterplot_densityplot_neg_probes.png")
    html_overview += f"""
    <h2>Negative probes</h2>
    {format_description(description)}
    <img src="data:image/png;base64,{img_neg_probes}" style="max-width:45%;">
    """

    description = """
    **Summary:** If spoQC detects nucleus-free cells, doublet events, and border cells, three plots are shown:

    - Spatial density plot of nucleus-free cells: high density may indicate segmentation issues.
    - Spatial density plot of cells associated with a doublet event: high density may indicate quality issues.
    - Border cells identified by spoQC (red): border cells can behave differently from the rest of the cell population and warrant further investigation.
    """
    img_nucelus_free = image_to_base64(f"{figure_path}/cellqc/scatterplot_nucleus_free.png")
    if os.path.exists(f"{figure_path}/hqcr/hqcr_celltype/scatterplot_densityplot_nucleus_free.png"):
        img_nucelus_free = image_to_base64(f"{figure_path}/cellqc/scatterplot_densityplot_nucleus_free.png")
    img_doublets = image_to_base64(f"{figure_path}/doubletqc/scatterplot_densityplot_doublet.png")
    img_border_cells = image_to_base64(f"{figure_path}/cellqc/scatterplot_border_cell.png")
    html_overview += f"""
    <h2>Nucleus-free cells, vertical doublets, and border cells</h2>
    {format_description(description)}
    <img src="data:image/png;base64,{img_nucelus_free}" style="max-width:30%;">
    <img src="data:image/png;base64,{img_doublets}" style="max-width:30%;">
    <img src="data:image/png;base64,{img_border_cells}" style="max-width:30%;">
    """

    # Cellcycle QC if exists
    description = """
    **Summary:** spoQC infers cell cycle phase when cell-cycling genes are present in the data.

    **Details:** The spatial density plot highlights cells associated with a specific phase.
    """
    if os.path.exists(f"{figure_path}/cellcycleqc/barplot_sample_cellcycle_fractions.png"):
        img_cc_bar = image_to_base64(f"{figure_path}/cellcycleqc/barplot_sample_cellcycle_fractions.png")
        img_cc_spatial = image_to_base64(f"{figure_path}/cellcycleqc/scatterplot_densityplot_phase_1.png")
        html_overview += f"""
        <h2>Cell cycle QC</h2>
        {format_description(description)}
        <img src="data:image/png;base64,{img_cc_bar}" style="max-width:45%;">
        <img src="data:image/png;base64,{img_cc_spatial}" style="max-width:45%;">
        """

    description = """
    **Summary:** spoQC investigates cell-free areas, referred to as voids, by plotting the number of unassigned transcripts (uRNAs) within them. Dark areas indicate potential segmentation issues or regions where cells might benefit from transcript reassignment.

    **Details:** Unassigned transcripts are transcripts that could not be assigned to a cell. Some rate of unassigned transcripts is expected, but this rate is influenced by the segmentation or segmentation-free algorithm used.
    """
    if os.path.exists(f"{figure_path}/voidqc/spatial_traingle_all_clsuters_log10_transcripts_counts_outside_cell.png"):
        img_void_sp = image_to_base64(
            f"{figure_path}/voidqc/spatial_traingle_all_clsuters_log10_transcripts_counts_outside_cell.png"
        )
        html_overview += f"""
        <h2>Void analysis</h2>
        {format_description(description)}
        <img src="data:image/png;base64,{img_void_sp}" style="max-width:45%;">
        """

    ####################################################################################################################
    # Second page
    ####################################################################################################################

    description_subcluster = """
    **Summary:** spoQC selects the cell type cluster with the largest number of cells for further analysis. Subclustering helps to reveal quality-impacted cell clusters that were not apparent in the <u>Overview</u> panel.

    **Details:** spoQC performs Leiden clustering at a resolution that yields up to 15 subclusters.
    """
    html_subcluster = f"""
    <h1>Subcluster analysis</h1>
    {format_description(description_subcluster)}
    """

    second_page_present = False
    if os.path.exists(f"{figure_path}/analysis/cluster/funkyheatmap/funkyheatmap_1.png"):
        second_page_present = True

    if second_page_present:
        img_funkyheatmap = image_to_base64(f"{figure_path}/analysis/cluster/funkyheatmap/funkyheatmap_1.png")
        html_subcluster += f"""
        <h2>Subcluster purity analysis</h2>
        {format_description(description_summary)}
        <img src="data:image/png;base64,{img_funkyheatmap}" style="max-width:80%;">
        """

        with open(f"{figure_path}/analysis/cluster/umap/umap_plot_celltype.html") as f:
            html_annotation = f.read()
        with open(f"{figure_path}/analysis/cluster/umap/umap_plot_leiden.html") as f:
            html_leiden = f.read()
        html_subcluster += f"""
        <h2>Annotation and Leiden clustering</h2>
        {format_description(description_umap)}
        <div style="display:inline-block; vertical-align:top; width:48%;">{html_annotation}</div>
        <div style="display:inline-block; vertical-align:top; width:48%;">{html_leiden}</div>
        """

        img_ump_hqr = image_to_base64(f"{figure_path}/analysis/cluster/umap/umap_plot_hqr_filtered_out.png")
        img_spa_hqr = image_to_base64(f"{figure_path}/analysis/cluster/scatterplot/scatterplot_hqr_filtered_out.png")
        html_subcluster += f"""
        <h2>Filter HQRs</h2>
        {format_description(description_filter)}
        <img src="data:image/png;base64,{img_ump_hqr}" style="max-width:45%;">
        <img src="data:image/png;base64,{img_spa_hqr}" style="max-width:45%;">
        """

        img_bar_cpc = image_to_base64(f"{figure_path}/analysis/cluster/barplot/barplot_pct_control_probe_counts.png")
        html_subcluster += f"""
        <h2>Control probe counts per cluster</h2>
        {format_description(description_bar_control_probes)}
        <img src="data:image/png;base64,{img_bar_cpc}" style="max-width:45%;">
        """

        img_bar_dc = image_to_base64(f"{figure_path}/analysis/cluster/barplot/barplot_pct_doublet_celltype.png")
        img_bar_dl = image_to_base64(f"{figure_path}/analysis/cluster/barplot/barplot_pct_doublet_leiden.png")
        html_subcluster += f"""
        <h2>Doublet counts per cluster</h2>
        {format_description(description_doublet)}
        <img src="data:image/png;base64,{img_bar_dc}" style="max-width:45%;">
        <img src="data:image/png;base64,{img_bar_dl}" style="max-width:45%;">
        """

        img_bar_nfc = image_to_base64(f"{figure_path}/analysis/cluster/barplot/barplot_pct_nucleus_free_celltype.png")
        img_bar_nfl = image_to_base64(f"{figure_path}/analysis/cluster/barplot/barplot_pct_nucleus_free_leiden.png")
        html_subcluster += f"""
        <h2>Nucleus-free cell counts per cluster</h2>
        {format_description(description_nucelus_free)}
        <img src="data:image/png;base64,{img_bar_nfc}" style="max-width:45%;">
        <img src="data:image/png;base64,{img_bar_nfl}" style="max-width:45%;">
        """


    ####################################################################################################################
    # Third page
    ####################################################################################################################

    html_sp_leiden = ""

    html_sp_leiden += f"""
    <h1>Spatial plots of Leiden clusters</h1>
    """

    description = """
    Spatial plots for the individual Leiden clusters identified in the <u>Overview</u> panel.
    """

    folder_path = f"{figure_path}/analysis/overview/scatterplot"
    html_sp_leiden += render_numbered_image_gallery(
        "Leiden clusters", folder_path, "scatterplot_leiden_cluster",
        description=description
    )

    ####################################################################################################################
    # Fourth page
    ####################################################################################################################

    html_sp_ann = ""

    html_sp_ann += f"""
    <h1>Spatial plots of annotation clusters</h1>
    """

    description = """
    Spatial plots for the individual cell type clusters, either provided as input or identified by spoQC via optimized Leiden clustering.
    """

    folder_path = f"{figure_path}/analysis/overview/scatterplot"
    html_sp_ann += render_numbered_image_gallery(
        "Annotation clusters", folder_path, "scatterplot_annotation",
        description=description
    )

    ####################################################################################################################
    # Fifth page
    ####################################################################################################################

    html_hqr = ""

    html_hqr += f"""
    <h2>HQRs selected by spoQC and saved in the metadata</h2>
    """

    description = """
    Cells marked in red belong to an HQCR. Individual HQCRs are stored in the AnnData object and in a separate metadata JSON file (see the spoQC documentation for details on the metadata format).
    """
    img_hqcr_selected = image_to_base64(f"{figure_path}/hqcr/hqcr_ident/scatterplot_refined_qc_class.png")
    html_hqr += f"""
    <h2>HQCR selected by spoQC</h2>
    {format_description(description)}
    <img src="data:image/png;base64,{img_hqcr_selected}" style="max-width:45%;">
    """

    description = """
    Red boxes mark areas belonging to an HQPR. Individual HQPRs are stored in the AnnData object and in a separate metadata file (see the spoQC documentation for details on the metadata format).
    """
    folder_path = f'{figure_path}/hqpr'
    folder_path_continue = 'hqpr_bounding_box/'
    html_hqr += render_image_gallery(
        "HQPRs selected by spoQC", "imageplot_marked_subfigures.png", stainings,
        performed_stainings, folder_path, 
        folder_path_continue=folder_path_continue, description=description
    )


    description = """
    Red boxes mark areas belonging to an HQTR. Individual HQTRs are stored in the AnnData object and in a separate metadata file (see to the spoQC documentation for details on the metadata format).
    """
    img_bb_hqtr = image_to_base64(f"{figure_path}/hqtr/hqtr_bounding_box/imageplot_marked_subfigures.png")
    html_hqr += f"""
    <h2>HQTR selected by spoQC</h2>
    {format_description(description)}
    <img src="data:image/png;base64,{img_bb_hqtr}" style="max-width:45%;">
    """

    ####################################################################################################################
    # Sixth page
    ####################################################################################################################

    html_filters = ""

    description = """
    **Summary:** Individual HQCR filter. Cells marked in black would be excluded.

    **Details:** A cell is excluded (marked as low quality) if its HQCR belief < 0.45.
    """
    img_ump_hqcr = image_to_base64(f"{figure_path}/analysis/overview/umap/umap_plot_hqcr_filtered_out.png")
    img_spa_hqcr = image_to_base64(f"{figure_path}/analysis/overview/scatterplot/scatterplot_hqcr_filtered_out.png")
    html_filters += f"""
    <h2>Filter HQCRs</h2>
    {format_description(description)}
    <img src="data:image/png;base64,{img_ump_hqcr}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_spa_hqcr}" style="max-width:45%;">
    """

    description = """
    **Summary:** Individual HQPR filter. Cells marked in black would be excluded.

    **Details:** A cell is excluded (marked as low quality) if its mean informative HQPR belief < 0.45. Because a segmented cell typically spans more than one pixel, spoQC calculates the mean informative belief (mean of values > 0.2) for this modality, which avoids the low-value-inflation bias that would otherwise correlate with cell size.
    """
    html_filters += render_numbered_paired_image_gallery(
        f'{figure_path}/analysis/overview/umap/',
        f'{figure_path}/analysis/overview/scatterplot/',
        'umap_plot_hqpr',
        'scatterplot_hqpr',
        description=description
    )


    description = """
    **Summary:** Individual HQTR filter. Cells marked in black would be excluded.

    **Details:** A cell is excluded (marked as low quality) if its mean informative HQTR belief < 0.45. Because a segmented cell typically spans more than one transcript, spoQC calculates the mean informative belief (mean of values > 0.2) for this modality, which avoids the low-value-inflation bias that would otherwise correlate with cell size.
    """
    img_ump_hqtr = image_to_base64(f"{figure_path}/analysis/overview/umap/umap_plot_hqtr_filtered_out.png")
    img_spa_hqtr = image_to_base64(f"{figure_path}/analysis/overview/scatterplot/scatterplot_hqtr_filtered_out.png")
    html_filters += f"""
    <h2>Filter HQTRs</h2>
    {format_description(description)}
    <img src="data:image/png;base64,{img_ump_hqtr}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_spa_hqtr}" style="max-width:45%;">
    """

    ####################################################################################################################
    # Seventh page
    ####################################################################################################################

    html_hqcr = ""

    html_hqcr += f"""
    <h1>All HQCR metrics used by spoQC</h1>
    <p>Boxplots show the Leiden clusters identified in the <u>Overview</u> panel.</p>
    """

    folder_path = f"{figure_path}/analysis/overview/boxplot/"
    if os.path.exists(folder_path):
        hqcr_files = sorted(
            f for f in os.listdir(folder_path) if f.endswith(".png") and "hqpr" not in f and "hqtr" not in f
        )
        for fname in hqcr_files:
            img_data = image_to_base64(f"{folder_path}/{fname}")
            html_hqcr += f"""
            <img src="data:image/png;base64,{img_data}" style="max-width:45%;">
            """

    ####################################################################################################################
    # Eigth page
    ####################################################################################################################

    html_hqpr = ""

    html_hqpr += f"""
    <h1>All HQPR metrics</h1>
    <p>Boxplots show the Leiden clusters identified in the <u>Overview</u> panel.</p>
    """

    folder_path = f"{figure_path}/analysis/overview/boxplot/"
    if os.path.exists(folder_path):
        hqpr_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".png") and "hqpr" in f)
        for fname in hqpr_files:
            img_data = image_to_base64(f"{folder_path}/{fname}")
            html_hqpr += f"""
            <img src="data:image/png;base64,{img_data}" style="max-width:45%;">
            """


    ####################################################################################################################
    # Nineth page
    ####################################################################################################################

    html_hqtr = ""

    html_hqtr += f"""
    <h1>All HQTR metrics</h1>
    <p>Boxplots show the Leiden clusters identified in the <u>Overview</u> panel.</p>
    """

    folder_path = f"{figure_path}/analysis/overview/boxplot/"
    if os.path.exists(folder_path):
        hqtr_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".png") and "hqtr" in f)
        for fname in hqtr_files:
            img_data = image_to_base64(f"{folder_path}/{fname}")
            html_hqtr += f"""
            <img src="data:image/png;base64,{img_data}" style="max-width:45%;">
            """


    ####################################################################################################################
    # End
    ####################################################################################################################

    ### First half ####
    pages_first_half = [{"id": "overview", "title": "Overview", "content": html_overview}]
    if second_page_present:
        pages_first_half.append({"id": "subcluster", "title": "Subcluster analysis", "content": html_subcluster})
    pages_first_half.append({"id": "spatialplot_leiden", "title": "Spatial plots Leiden clusters", "content": html_sp_leiden})
    pages_first_half.append({"id": "spatialplot_annotation", "title": "Spatial plots annotation clusters", "content": html_sp_ann})
    pages_first_half.append({"id": "hqr", "title": "High quality regions (HQRs)", "content": html_hqr})
    pages_first_half.append({"id": "individual_filters", "title": "Individual HQR filters", "content": html_filters})
    pages_first_half.append({"id": "hqcr_boxplots", "title": "All HQCR metrics", "content": html_hqcr})
    pages_first_half.append({"id": "hqpr_boxplots", "title": "All HQPR metrics", "content": html_hqpr})
    pages_first_half.append({"id": "hqtr_boxplots", "title": "All HQTR metrics", "content": html_hqtr})

    ### Second half ###
    # Fold the other Plotly-generated HTML fragments into the same document as
    # additional pages instead of moving them out as standalone sibling files.
    pages_second_half = []

    if os.path.exists(f"{figure_path}/generalqc/rna_qc_sample_mqc.html"):
        with open(f"{figure_path}/generalqc/rna_qc_sample_mqc.html") as f:
            pages_second_half.append({"id": "rna_qc_sample", "title": "RNA QC Sample", "content": f.read()})

    if os.path.exists(f"{figure_path}/hqcr/hqcr_celltype/hqcr_celltype.html"):
        with open(f"{figure_path}/hqcr/hqcr_celltype/hqcr_celltype.html") as f:
            pages_second_half.append({"id": "hqcr_celltype", "title": "HQCR Celltype", "content": f.read()})

    if os.path.exists(f"{figure_path}/hqcr/hqcr_ident/hqcr_cell_region.html"):
        with open(f"{figure_path}/hqcr/hqcr_ident/hqcr_cell_region.html") as f:
            pages_second_half.append({"id": "hqcr_cell_region", "title": "HQCR Cell Region", "content": f.read()})

    for s in performed_stainings:
        if os.path.exists(f"{figure_path}/hqpr/hqpr_celltype/{s}/celltype_qc_analysis.html"):
            with open(f"{figure_path}/hqpr/hqpr_celltype/{s}/celltype_qc_analysis.html") as f:
                pages_second_half.append({
                    "id": f"hqpr_{s}_celltype", "title": f"HQPR Celltype QC ({stainings[s]})", "content": f.read()
                })

    if os.path.exists(f"{figure_path}/hqtr/hqtr_celltype/celltype_qc_analysis.html"):
        with open(f"{figure_path}/hqtr/hqtr_celltype/celltype_qc_analysis.html") as f:
            pages_second_half.append({"id": "hqtr_celltype", "title": "HQTR Celltype QC", "content": f.read()})

    ### Rendering ###
    def build_report_html(pages):
        sidebar_buttons = "\n".join(
            f'<button id="nav-{p["id"]}" class="{"active" if i == 0 else ""}" '
            f'onclick="showPage(\'{p["id"]}\')">{p["title"]}</button>'
            for i, p in enumerate(pages)
        )
        content_divs = "\n".join(
            f'<div id="page-{p["id"]}" class="page{" active" if i == 0 else ""}">{p["content"]}</div>'
            for i, p in enumerate(pages)
        )

        return f"""
    <html>
    <head>
    <style>
        body {{ margin: 0; display: flex; font-family: sans-serif; }}
        #sidebar {{ width: 240px; flex-shrink: 0; background: #f4f4f4; padding: 10px;
                    box-sizing: border-box; height: 100vh; overflow-y: auto; }}
        #sidebar button {{ display: block; width: 100%; text-align: left; padding: 8px 10px;
                            margin-bottom: 4px; border: none; background: none; cursor: pointer;
                            border-radius: 4px; font-size: 14px; }}
        #sidebar button:hover, #sidebar button.active {{ background: #dbe4ff; }}
        #content {{ flex: 1; padding: 20px; overflow-y: auto; height: 100vh; box-sizing: border-box; }}
        #content p {{ line-height: 1.5; max-width: 900px; }}
        #content ul {{ line-height: 1.5; max-width: 900px; }}
        .page {{ display: none; }}
        .page.active {{ display: block; }}
    </style>
    <script>
    function showPage(id) {{
        document.querySelectorAll('.page').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('#sidebar button').forEach(el => el.classList.remove('active'));
        document.getElementById('page-' + id).classList.add('active');
        document.getElementById('nav-' + id).classList.add('active');
    }}
    </script>
    </head>
    <body>
    <div id="sidebar">
    {sidebar_buttons}
    </div>
    <div id="content">
    {content_divs}
    </div>
    </body>
    </html>
    """

    pages = pages_first_half + pages_second_half

    with open(f"{figure_path}/report.html", "w") as f:
        f.write(build_report_html(pages))

    with open(f"{figure_path}/report_part1.html", "w") as f:
        f.write(build_report_html(pages_first_half))

    with open(f"{figure_path}/report_part2.html", "w") as f:
        f.write(build_report_html(pages_second_half))

# %%
