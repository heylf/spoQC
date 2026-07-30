# In[]
import os
import re
import base64

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def render_image_gallery(header, image_filename, stainings, folder_path, folder_path_continue=None, description=None):
    html = f"""
    <h2>{header}</h2>
    """
    if description:
        html += f"""
    <p>{description}</p>
    """
    for s, staining in enumerate(stainings):
        size = 24
        if len(stainings) == 1:
            size = 45
        img_file = f"{folder_path}/{s}/{image_filename}"
        if folder_path_continue:
            img_file = f"{folder_path}/{s}/{folder_path_continue}/{image_filename}"
        img_data = image_to_base64(img_file)
        html += f"""
        <div style="display:inline-block; max-width:{size}%;">
            <span style="display:block; background:rgba(0,0,0,0.6); color:#fff; font-size:12px;
            padding:2px 6px; border-radius:3px; width:fit-content;">{staining}</span>
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
        html += f"""
    <p>{description}</p>
    """
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
        html += f"""
    <p>{description}</p>
    """

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
    <h1>High quality regions (HQRs)</h1>
    """

    folder_path = f'{figure_path}/combine_masks'
    html_overview += render_image_gallery("Combined beliefs", "imageplot_combined_beliefs.png", stainings, folder_path,
                                description="TODO: describe this section")
    html_overview += render_image_gallery("Combined masks", "imageplot_combined_masks.png", stainings, folder_path,
                                description="TODO: describe this section")
    html_overview += render_image_gallery("Venndiagram of overlapping HQRs", "venn_combined_masks.png", stainings, folder_path,
                                description="TODO: describe this section")

    html_overview += f"""
    <h1>High quality regions (HQRs) with spatial dependencies</h1>
    """

    html_overview += render_image_gallery("Combined beliefs with spatial dependencies",
                                "imageplot_combined_beliefs_smoothed.png", stainings, folder_path,
                                description="TODO: describe this section")
    html_overview += render_image_gallery("Combined masks with spatial dependencies",
                                "imageplot_combined_masks_smoothed.png", stainings, folder_path,
                                description="TODO: describe this section")
    html_overview += render_image_gallery("Venndiagram of overlapping HQRs with spatial dependencies",
                                "venn_combined_masks_smoothed.png", stainings, folder_path,
                                description="TODO: describe this section")


    ##########
    # In-depth
    ##########

    img_funkyheatmap = image_to_base64(f"{figure_path}/analysis/overview/funkyheatmap/funkyheatmap_1.png")
    html_overview += f"""
    <h1>Summary of spoQC</h1>
    <p>TODO: describe this section</p>
    <img src="data:image/png;base64,{img_funkyheatmap}" style="max-width:80%;">
    """

    with open(f"{figure_path}/analysis/overview/umap/umap_plot_celltype.html") as f:
        html_annotation = f.read()
    with open(f"{figure_path}/analysis/overview/umap/umap_plot_leiden.html") as f:
        html_leiden = f.read()
    html_overview += f"""
    <h2>Annotation and Leiden clustering</h2>
    <p>TODO: describe this section</p>
    <div style="display:inline-block; vertical-align:top; width:48%;">{html_annotation}</div>
    <div style="display:inline-block; vertical-align:top; width:48%;">{html_leiden}</div>
    """

    img_ump_hqr = image_to_base64(f"{figure_path}/analysis/overview/umap/umap_plot_hqr_filtered_out.png")
    img_spa_hqr = image_to_base64(f"{figure_path}/analysis/overview/scatterplot/scatterplot_hqr_filtered_out.png")
    html_overview += f"""
    <h2>Filter HQRs</h2>
    <p>TODO: describe this section</p>
    <img src="data:image/png;base64,{img_ump_hqr}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_spa_hqr}" style="max-width:45%;">
    """

    img_bar_cpc = image_to_base64(f"{figure_path}/analysis/overview/barplot/barplot_pct_control_probe_counts.png")
    html_overview += f"""
    <h2>Control probe counts per cluster</h2>
    <p>TODO: describe this section</p>
    <img src="data:image/png;base64,{img_bar_cpc}" style="max-width:45%;">
    """

    img_bar_dc = image_to_base64(f"{figure_path}/analysis/overview/barplot/barplot_pct_doublet_celltype.png")
    img_bar_dl = image_to_base64(f"{figure_path}/analysis/overview/barplot/barplot_pct_doublet_leiden.png")
    html_overview += f"""
    <h2>Doublet counts per cluster</h2>
    <p>TODO: describe this section</p>
    <img src="data:image/png;base64,{img_bar_dc}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_bar_dl}" style="max-width:45%;">
    """

    img_bar_nfc = image_to_base64(f"{figure_path}/analysis/overview/barplot/barplot_pct_nucleus_free_celltype.png")
    img_bar_nfl = image_to_base64(f"{figure_path}/analysis/overview/barplot/barplot_pct_nucleus_free_leiden.png")
    html_overview += f"""
    <h2>Nucelus free cell counts per cluster</h2>
    <p>TODO: describe this section</p>
    <img src="data:image/png;base64,{img_bar_nfc}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_bar_nfl}" style="max-width:45%;">
    """

    ##########
    # Individual but important
    ##########
    html_overview += f"""
    <h1>Individual plots worth inspecting</h1>
    """

    img_cell = image_to_base64(f"{figure_path}/cellqc/scatterplot_densityplot_convexity_cell_convexity_metric_cell.png")
    img_nucl = image_to_base64(f"{figure_path}/cellqc/scatterplot_densityplot_convexity_nuclei_convexity_mean_nuceli.png")
    html_overview += f"""
    <h2>Convexity of cell and nuceli shapes</h2>
    <p>TODO: describe this section</p>
    <img src="data:image/png;base64,{img_cell}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_nucl}" style="max-width:45%;">
    """

    img_cell = image_to_base64(f"{figure_path}/generalqc/scatterplot_densityplot_invalid_cell_geometry.png")
    img_nucl = image_to_base64(f"{figure_path}/generalqc/scatterplot_densityplot_invalid_nucleus_geometry.png")
    html_overview += f"""
    <h2>Invalid cell and nuceli geometries</h2>
    <p>TODO: describe this section</p>
    <img src="data:image/png;base64,{img_cell}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_nucl}" style="max-width:45%;">
    """

    img_low_qv_trans = image_to_base64(f"{figure_path}/cellqc/scatterplot_densityplot_num_low_qc_transcript.png")
    html_overview += f"""
    <h2>Low quality transcripts (qc < 20)</h2>
    <p>TODO: describe this section</p>
    <img src="data:image/png;base64,{img_low_qv_trans}" style="max-width:45%;">
    """

    img_neg_probes = image_to_base64(f"{figure_path}/transcriptqc/scatterplot_densityplot_neg_probes.png")
    html_overview += f"""
    <h2>Negative probes</h2>
    <p>TODO: describe this section</p>
    <img src="data:image/png;base64,{img_neg_probes}" style="max-width:45%;">
    """

    img_nucelus_free = image_to_base64(f"{figure_path}/cellqc/scatterplot_nucleus_free.png")
    if os.path.exists(f"{figure_path}/hqcr/hqcr_celltype/scatterplot_densityplot_nucleus_free.png"):
        img_nucelus_free = image_to_base64(f"{figure_path}/cellqc/scatterplot_densityplot_nucleus_free.png")
    img_doublets = image_to_base64(f"{figure_path}/doubletqc/scatterplot_densityplot_doublet.png")
    img_border_cells = image_to_base64(f"{figure_path}/cellqc/scatterplot_border_cell.png")
    html_overview += f"""
    <h2>Nucleus free cells, vertical doublets and border cells</h2>
    <p>TODO: describe this section</p>
    <img src="data:image/png;base64,{img_nucelus_free}" style="max-width:30%;">
    <img src="data:image/png;base64,{img_doublets}" style="max-width:30%;">
    <img src="data:image/png;base64,{img_border_cells}" style="max-width:30%;">
    """

    # Cellcycle QC if exists
    if os.path.exists(f"{figure_path}/cellcycleqc/barplot_sample_cellcycle_fractions.png"):
        img_cc_bar = image_to_base64(f"{figure_path}/cellcycleqc/barplot_sample_cellcycle_fractions.png")
        img_cc_spatial = image_to_base64(f"{figure_path}/cellcycleqc/scatterplot_densityplot_phase_1.png")
        html_overview += f"""
        <h2>Cellcycle QC</h2>
        <p>TODO: describe this section</p>
        <img src="data:image/png;base64,{img_cc_bar}" style="max-width:45%;">
        <img src="data:image/png;base64,{img_cc_spatial}" style="max-width:45%;">
        """

    img_void_sp = image_to_base64(
        f"{figure_path}/voidqc/spatial_traingle_all_clsuters_log10_transcripts_counts_outside_cell.png"
    )
    html_overview += f"""
    <h2>Void analysis</h2>
    <p>TODO: describe this section</p>
    <img src="data:image/png;base64,{img_void_sp}" style="max-width:45%;">
    """

    ####################################################################################################################
    # Second page
    ####################################################################################################################

    html_subcluster = ""

    img_funkyheatmap = image_to_base64(f"{figure_path}/analysis/cluster/funkyheatmap/funkyheatmap_1.png")
    html_subcluster += f"""
    <h1>Subcluster purity analysis</h1>
    <p>TODO: describe this section</p>
    <img src="data:image/png;base64,{img_funkyheatmap}" style="max-width:80%;">
    """

    with open(f"{figure_path}/analysis/cluster/umap/umap_plot_celltype.html") as f:
        html_annotation = f.read()
    with open(f"{figure_path}/analysis/cluster/umap/umap_plot_leiden.html") as f:
        html_leiden = f.read()
    html_subcluster += f"""
    <h2>Annotation and Leiden clustering</h2>
    <p>TODO: describe this section</p>
    <div style="display:inline-block; vertical-align:top; width:48%;">{html_annotation}</div>
    <div style="display:inline-block; vertical-align:top; width:48%;">{html_leiden}</div>
    """

    img_ump_hqr = image_to_base64(f"{figure_path}/analysis/cluster/umap/umap_plot_hqr_filtered_out.png")
    img_spa_hqr = image_to_base64(f"{figure_path}/analysis/cluster/scatterplot/scatterplot_hqr_filtered_out.png")
    html_subcluster += f"""
    <h2>Filter HQRs</h2>
    <p>TODO: describe this section</p>
    <img src="data:image/png;base64,{img_ump_hqr}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_spa_hqr}" style="max-width:45%;">
    """

    img_bar_cpc = image_to_base64(f"{figure_path}/analysis/cluster/barplot/barplot_pct_control_probe_counts.png")
    html_subcluster += f"""
    <h2>Control probe counts per cluster</h2>
    <p>TODO: describe this section</p>
    <img src="data:image/png;base64,{img_bar_cpc}" style="max-width:45%;">
    """

    img_bar_dc = image_to_base64(f"{figure_path}/analysis/cluster/barplot/barplot_pct_doublet_celltype.png")
    img_bar_dl = image_to_base64(f"{figure_path}/analysis/cluster/barplot/barplot_pct_doublet_leiden.png")
    html_subcluster += f"""
    <h2>Doublet counts per cluster</h2>
    <p>TODO: describe this section</p>
    <img src="data:image/png;base64,{img_bar_dc}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_bar_dl}" style="max-width:45%;">
    """

    img_bar_nfc = image_to_base64(f"{figure_path}/analysis/cluster/barplot/barplot_pct_nucleus_free_celltype.png")
    img_bar_nfl = image_to_base64(f"{figure_path}/analysis/cluster/barplot/barplot_pct_nucleus_free_leiden.png")
    html_subcluster += f"""
    <h2>Nucelus free cell counts per cluster</h2>
    <p>TODO: describe this section</p>
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

    folder_path = f"{figure_path}/analysis/overview/scatterplot"
    html_sp_leiden += render_numbered_image_gallery(
        "Leiden clusters", folder_path, "scatterplot_leiden_cluster",
        description="TODO: describe this section"
    )

    ####################################################################################################################
    # Fourth page
    ####################################################################################################################

    html_sp_ann = ""

    html_sp_ann += f"""
    <h1>Spatial plots of annotation clusters</h1>
    """

    folder_path = f"{figure_path}/analysis/overview/scatterplot"
    html_sp_ann += render_numbered_image_gallery(
        "Annotation clusters", folder_path, "scatterplot_annotation",
        description="TODO: describe this section"
    )

    ####################################################################################################################
    # Fifth page
    ####################################################################################################################

    html_hqr = ""

    html_hqr += f"""
    <h2>HQRs selected by spoQC and saved in the metadata</h1>
    """

    img_bb_hqtr = image_to_base64(f"{figure_path}/hqcr/hqcr_ident/scatterplot_refined_qc_class.png")
    html_hqr += f"""
    <h2>HQCR selected by spoQC</h2>
    <p>TODO: describe this section</p>
    <img src="data:image/png;base64,{img_cell}" style="max-width:45%;">
    """

    folder_path = f'{figure_path}/hqpr'
    folder_path_continue = 'hqpr_bounding_box/'
    html_hqr += render_image_gallery("HQPRs selected by spoQC", "imageplot_marked_subfigures.png", stainings,
                                folder_path, folder_path_continue, description="TODO: describe this section")


    img_bb_hqtr = image_to_base64(f"{figure_path}/hqtr/hqtr_bounding_box/imageplot_marked_subfigures.png")
    html_hqr += f"""
    <h2>HQTR selected by spoQC</h2>
    <p>TODO: describe this section</p>
    <img src="data:image/png;base64,{img_bb_hqtr}" style="max-width:45%;">
    """

    ####################################################################################################################
    # Sixth page
    ####################################################################################################################

    html_filters = ""

    img_ump_hqcr = image_to_base64(f"{figure_path}/analysis/overview/umap/umap_plot_hqcr_filtered_out.png")
    img_spa_hqcr = image_to_base64(f"{figure_path}/analysis/overview/scatterplot/scatterplot_hqcr_filtered_out.png")
    html_filters += f"""
    <h2>Filter HQTRs</h2>
    <p>TODO: describe this section</p>
    <img src="data:image/png;base64,{img_ump_hqcr}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_spa_hqcr}" style="max-width:45%;">
    """

    html_filters += render_numbered_paired_image_gallery(
        f'{figure_path}/analysis/overview/umap/',
        f'{figure_path}/analysis/overview/scatterplot/',
        'umap_plot_hqpr',
        'scatterplot_hqpr',
        description="TODO: describe this section"
    )

    img_ump_hqtr = image_to_base64(f"{figure_path}/analysis/overview/umap/umap_plot_hqtr_filtered_out.png")
    img_spa_hqtr = image_to_base64(f"{figure_path}/analysis/overview/scatterplot/scatterplot_hqtr_filtered_out.png")
    html_filters += f"""
    <h2>Filter HQTRs</h2>
    <p>TODO: describe this section</p>
    <img src="data:image/png;base64,{img_ump_hqtr}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_spa_hqtr}" style="max-width:45%;">
    """

    ####################################################################################################################
    # Seventh page
    ####################################################################################################################

    html_hqcr = ""

    html_hqcr += f"""
    <h1>All HQCR metrics</h1>
    <p>TODO: describe this section</p>
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
    <p>TODO: describe this section</p>
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
    <h1>All HQPR metrics</h1>
    <p>TODO: describe this section</p>
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
    pages_first_half.append({"id": "subcluster", "title": "Subcluster anlysis", "content": html_subcluster})
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

    for s, staining in enumerate(stainings):
        if os.path.exists(f"{figure_path}/hqpr/{s}/hqpr_celltype/celltype_qc_analysis.html"):
            with open(f"{figure_path}/hqpr/{s}/hqpr_celltype/celltype_qc_analysis.html") as f:
                pages_second_half.append({"id": f"hqpr_{s}_celltype", "title": f"HQPR Celltype QC ({staining})", "content": f.read()})

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
