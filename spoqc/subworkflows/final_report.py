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
            img_file = f"{folder_path}/{folder_path_continue}/{s}/{image_filename}"
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
    <h1>High quality regions (HQRs) for HQCR (high quality cell regions), HQPR (high quality pixel regions) and HQTR (high quality transcript regions)</h1>
    """

    folder_path = f'{figure_path}/combine_masks'
    
    description = """
    Short description:
    Dark (dim) regions highlight areas impacted by low quality. 
    If you see large patches in your slide then spoQC tells you that there is a fundamental problem.

    Additional description:
    Combined belief (probability) masks for all data modalities: HQCR, HQPR, and HQTR.
    Each data modality operates in pixel coordinates, that means, that spoQC transformes each data modality into an image.
    The brighter a pixel is the higher is the probability that the pixel is of good quality.
    Thus, dark (dim) regions highlight areas impacted by low quality.
    """
    html_overview += render_image_gallery("Combined beliefs", "imageplot_combined_beliefs.png", stainings, folder_path,
                                description=description)
    
    description = """
    Short description:
    Dark (dim) regions highlight areas impacted by low quality.
    If you see large patches in your slide then spoQC tells you that there is a fundamental problem.

    Additional description:
    The beliefs (probabiltiies) for all data modalities (HQCR, HQPR, and HQTR) where filtered by a cutoff of > 0.5 (=1 else 0).
    For each pixel spoQC calculates the sum of 1's.

    Categories:
    * no mask: Pixel is covered by 0 masks.
    * 1 mask: Pixel is covered by only 1 mask.
    * 2 mask: Pixel is covered by 2 masks.
    * all mask: Pixel is covered by all 3 masks.
    """
    html_overview += render_image_gallery("Combined masks", "imageplot_combined_masks.png", stainings, folder_path,
                                description=description)
    

    description = """
    Short description:
    Percentage of overlap for each data modality. 
    If one modality has very low overlap with the other data modalities then this is an indication of a data modality specific issue.

    Additional description:
    Using the integer masks from before, spoQC's displays the percentages of ovlerap for each data modality.
    """
    html_overview += render_image_gallery("Venndiagram of overlapping HQRs", "venn_combined_masks.png", stainings, folder_path,
                                description=description)


    html_overview += f"""
    <h1>High quality regions (HQRs) with spatial dependencies</h1>
    """

    description = """
    Short description:
    Dark (dim) regions highlight areas impacted by low quality. 
    If you see large patches in your slide then spoQC tells you that there is a fundamental problem.

    Additional description:
    The difference from before is that we now take spatial inforamtion info account.
    That means, pixel quality is impacted by the general pixel quality in the neighbourhood.
    For more information read our detailed documentation.
    """
    html_overview += render_image_gallery("Combined beliefs with spatial dependencies",
                                "imageplot_combined_beliefs_smoothed.png", stainings, folder_path,
                                description=description)
    
    description = """
    Short description:
    Dark (dim) regions highlight areas impacted by low quality.
    If you see large patches in your slide then spoQC tells you that there is a fundamental problem.

    Additional description:
    We take directly the predicted labels from a spoQC's markov random field model, which predicts the best set of laten factors for each pixel.
    That is why, we do not have to apply a threshold.
    As before, this incoorperates spatial information into the integrated quality mask.
    For each pixel spoQC calculates the sum of 1's.

    Categories:
    * no mask: Pixel is covered by 0 masks.
    * 1 mask: Pixel is covered by only 1 mask.
    * 2 mask: Pixel is covered by 2 masks.
    * all mask: Pixel is covered by all 3 masks.
    """
    html_overview += render_image_gallery("Combined masks with spatial dependencies",
                                "imageplot_combined_masks_smoothed.png", stainings, folder_path,
                                description=description)
    

    description = """
    Short description:
    Percentage of overlap for each data modality. 
    If one modality has very low overlap with the other data modalities then this is an indication of a data modality specific issue.

    Additional description:
    Using the integer masks from before, spoQC's displays the percentages of ovlerap for each data modality.
    """
    html_overview += render_image_gallery("Venndiagram of overlapping HQRs with spatial dependencies",
                                "venn_combined_masks_smoothed.png", stainings, folder_path,
                                description="TODO: describe this section")

    ##########
    # In-depth
    ##########

    img_funkyheatmap = image_to_base64(f"{figure_path}/analysis/overview/funkyheatmap/funkyheatmap_1.png")

    description_summary = """
    Short description:
    This is a total summary of spoQC's beliefs and metrics set for each data modality.
    On the y-axis spoQC depicts the Leiden clusters which are generated from the provided annotation.
    First check the HQCR, HQPR, and HQTR beliefs.
    If any cluster has significant lower beleifs than all other clusters then this is an indication that the Leiden cluster encompasses quality impacted cells.
    If you see such a cluster, then go furhter and check the individual metrics of spoQC.
    For example, low HQCR (cell/segemtnation) beliefs might be the resolut of a low number of transciprt counts or many cells close to vertical doublet events.
    """

    additional_description_summary = """
    Additional description:
    SpoQC performs additional Leiden clustering based on a resultion that achieves a number of clusters that has 3 more clusters than the provided cell type annotation.
    The cell type annotation is either provided by the user, or an intital Leiden clustering with optimized resolution done by spoQC.
    SpoQC picks 3 additional clusters since it hypothesises that the data includes cell type clusters impacted by quality.
    More details be observed in the panel's underscore(All HQCR metrics), underscore(All HQPR metrics), and underscore(All HQTR metrics).
    """
    html_overview += f"""
    <h1>Summary of spoQC</h1>
    <p>{description_summary}</p>
    <p>{additional_description_summary}</p>
    <img src="data:image/png;base64,{img_funkyheatmap}" style="max-width:80%;">
    """

    description_umap = """
    Short description:
    UMAPs of the cells, left with the applied cell type annotation and right with the Leiden clustering considering the cell type annotation.
    These plots help you to investigate the Leiden clusters that you might pin-point from above.
    For example, low HQCR (cell/segemtnation) beliefs might be the resolut of a low number of transciprt counts or many cells close to vertical doublet events.
    This might manifest in a Leiden cluster that significantly clusters away from the rest of the cell population, which is another indication that the Leiden cluster is quality impacted.
    """

    reference_description_umap = """
    Reference:
    The underscore(Spatial plots Leiden clusters) panel will show the spatial orchestration of the individual Leiden clusters.
    The underscore(Spatial plots annotation clusters) panel will show the spatial orchestration of the individual cell type clusters.
    """
    
    with open(f"{figure_path}/analysis/overview/umap/umap_plot_celltype.html") as f:
        html_annotation = f.read()
    with open(f"{figure_path}/analysis/overview/umap/umap_plot_leiden.html") as f:
        html_leiden = f.read()
    html_overview += f"""
    <h2>Annotation and Leiden clustering</h2>
    <p>{description_umap}</p>
    <p>{reference_description_umap}</p>
    <div style="display:inline-block; vertical-align:top; width:48%;">{html_annotation}</div>
    <div style="display:inline-block; vertical-align:top; width:48%;">{html_leiden}</div>
    """

    description_filter = """
    Short description:
    Left the UMAP for the Leiden clustering from before and right the spatial plot of each cell.
    Black marked cells are filtered out by spoQC's HQR filtering.
    Previously identified quality impacted cell clusters should be pciked up by spoQC's filter.

    Additional description:
    SpoQC filters by HQR considering all data modality.
    Cells are filter out (marked as low quality) if:
    * HQCR beliefs > 0.45
    * mean informative HQPR beliefs > 0.45
    * mean informative HQTR beliefs > 0.45
    Since a segmented cell can have more than one pixel and transcript we calculate the mean inforamtive beliefs for those data modalities.
    The mean informative beliefs is the mean value of all bliefs with a value > 0.
    The mean informative belief thus prevents the issue of a zero inflated distribution that correlates with the size of the segmented cell.
    """

    reference_description_filter = """
    Reference:
    The underscore(Individual HQR filters) panel will show the individual HQCR, HQPR and HQTR filers that are combined for the HQR filtering.
    """
    img_ump_hqr = image_to_base64(f"{figure_path}/analysis/overview/umap/umap_plot_hqr_filtered_out.png")
    img_spa_hqr = image_to_base64(f"{figure_path}/analysis/overview/scatterplot/scatterplot_hqr_filtered_out.png")
    html_overview += f"""
    <h2>Filter HQRs</h2>
    <p>{description_filter}</p>
    <p>{reference_description_filter}</p>
    <img src="data:image/png;base64,{img_ump_hqr}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_spa_hqr}" style="max-width:45%;">
    """

    description_bar_control_probes = """
    Short description:
    Percentage of cells having a control probe count > 0 for each cell cluster.
    This can help you to identify more clearly quality impacted cell clusters.
    Higher percentages for a cluster indicate a poor quality.
    """
    img_bar_cpc = image_to_base64(f"{figure_path}/analysis/overview/barplot/barplot_pct_control_probe_counts.png")
    html_overview += f"""
    <h2>Control probe counts per cluster</h2>
    <p>{description_bar_control_probes}</p>
    <img src="data:image/png;base64,{img_bar_cpc}" style="max-width:45%;">
    """

    description_doublet = """
    Short description:
    Percentage of cells classified to be close to doublet events for each cell cluster.
    This can help you to identify more clearly quality impacted cell clusters.
    Higher percentages for a cluster indicate a poor quality.
    """
    img_bar_dc = image_to_base64(f"{figure_path}/analysis/overview/barplot/barplot_pct_doublet_celltype.png")
    img_bar_dl = image_to_base64(f"{figure_path}/analysis/overview/barplot/barplot_pct_doublet_leiden.png")
    html_overview += f"""
    <h2>Doublet counts per cluster</h2>
    <p>{description_doublet}</p>
    <img src="data:image/png;base64,{img_bar_dc}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_bar_dl}" style="max-width:45%;">
    """

    description_nucelus_free = """
    Short description:
    Percentage of cells classified to be nucelus free for each cell cluster.
    This can help you to identify more clearly quality impacted cell clusters.
    Higher percentages for a cluster indicate a poor quality.
    """
    img_bar_nfc = image_to_base64(f"{figure_path}/analysis/overview/barplot/barplot_pct_nucleus_free_celltype.png")
    img_bar_nfl = image_to_base64(f"{figure_path}/analysis/overview/barplot/barplot_pct_nucleus_free_leiden.png")
    html_overview += f"""
    <h2>Nucelus free cell counts per cluster</h2>
    <p>{description_nucelus_free}</p>
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
    Short description:
    Spatial-density plot of the cells, where the density is weighted by the convexity of the cell (left) or the mean convexity of the nuclei of the cell (right).
    Since a cell might have more than one nulceus we calcuate the mean convexity.
    Low dense regions signal potential segmentations issues.

    Additional description:
    False (black): cell or nucleus is not convex (convexity <=0.5) and might have a weird shape.
    True (blue): cell or nucleus is convex (convexity > 0.5).
    We observed that many high convex cells and nuceli also can signal segmentations issues.
    This has something to do with our approach to correct invalid geometries.
    Thus check if highly dense regions overlap with highly dense invalid gemotry regions.
    """
    img_cell = image_to_base64(f"{figure_path}/cellqc/scatterplot_densityplot_convexity_cell_convexity_metric_cell.png")
    img_nucl = image_to_base64(f"{figure_path}/cellqc/scatterplot_densityplot_convexity_nuclei_convexity_mean_nuceli.png")
    html_overview += f"""
    <h2>Convexity of cell and nuceli shapes</h2>
    <p>f{description}</p>
    <img src="data:image/png;base64,{img_cell}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_nucl}" style="max-width:45%;">
    """

    description = """
    Short description:
    Density map of invalid cell or nucelus geometries.
    High density regions singla potential segmentation issues.    
    """
    img_cell = image_to_base64(f"{figure_path}/generalqc/scatterplot_densityplot_invalid_cell_geometry.png")
    img_nucl = image_to_base64(f"{figure_path}/generalqc/scatterplot_densityplot_invalid_nucleus_geometry.png")
    html_overview += f"""
    <h2>Invalid cell and nuceli geometries</h2>
    <p>{description}</p>
    <img src="data:image/png;base64,{img_cell}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_nucl}" style="max-width:45%;">
    """

    description = """
    Short description:
    Spatial-density plot of the cells, where the density is weighted by the number of low quality (qv < 20) transcripts.
    Highly dense region signal quality issues.

    Additional description:
    QV: Phred-scaled quality value (Q-Score) estimating the probability of incorrect call defined by 10x Genomics.
    A qv threshold of 20 was chosen as done by 10x Genomics.
    """
    img_low_qv_trans = image_to_base64(f"{figure_path}/cellqc/scatterplot_densityplot_num_low_qc_transcript.png")
    html_overview += f"""
    <h2>Low quality transcripts (qv < 20)</h2>
    <p>{description}</p>
    <img src="data:image/png;base64,{img_low_qv_trans}" style="max-width:45%;">
    """

    description = """
    Short description:
    Spatial-density plot of the cells weighted by the number of negative probes.
    Highly dense region signal quality issues.
    """
    img_neg_probes = image_to_base64(f"{figure_path}/transcriptqc/scatterplot_densityplot_neg_probes.png")
    html_overview += f"""
    <h2>Negative probes</h2>
    <p>f{description}</p>
    <img src="data:image/png;base64,{img_neg_probes}" style="max-width:45%;">
    """

    description = """
    Short description:
    If spoQC finds occasions of nulceus free cells, doublet events and border cells then you will see three plots.
    
    * A spatial-density plot of the nucleus free cells.
    A high density might signal segmentation issues.
    * A spatial-density plot of the cells associated with doublet event.
    A high density might signal quality issues.
    * Identification of border cells (red) done by spoQC.
    Border cells might behave differently the rest of your cells, so pleaes investigate them.
    """
    img_nucelus_free = image_to_base64(f"{figure_path}/cellqc/scatterplot_nucleus_free.png")
    if os.path.exists(f"{figure_path}/hqcr/hqcr_celltype/scatterplot_densityplot_nucleus_free.png"):
        img_nucelus_free = image_to_base64(f"{figure_path}/cellqc/scatterplot_densityplot_nucleus_free.png")
    img_doublets = image_to_base64(f"{figure_path}/doubletqc/scatterplot_densityplot_doublet.png")
    img_border_cells = image_to_base64(f"{figure_path}/cellqc/scatterplot_border_cell.png")
    html_overview += f"""
    <h2>Nucleus free cells, vertical doublets and border cells</h2>
    <p>{description}</p>
    <img src="data:image/png;base64,{img_nucelus_free}" style="max-width:30%;">
    <img src="data:image/png;base64,{img_doublets}" style="max-width:30%;">
    <img src="data:image/png;base64,{img_border_cells}" style="max-width:30%;">
    """

    # Cellcycle QC if exists
    description = """
    Short description:
    SpoQC investigates cell cycle phases if cell cycling genes are present in your data.

    Additional description:
    The spatial density plot highlight cells associated with a specific phase.
    """
    if os.path.exists(f"{figure_path}/cellcycleqc/barplot_sample_cellcycle_fractions.png"):
        img_cc_bar = image_to_base64(f"{figure_path}/cellcycleqc/barplot_sample_cellcycle_fractions.png")
        img_cc_spatial = image_to_base64(f"{figure_path}/cellcycleqc/scatterplot_densityplot_phase_1.png")
        html_overview += f"""
        <h2>Cellcycle QC</h2>
        <p>{description}</p>
        <img src="data:image/png;base64,{img_cc_bar}" style="max-width:45%;">
        <img src="data:image/png;base64,{img_cc_spatial}" style="max-width:45%;">
        """

    description = """
    Short description:
    SpoQC's investigates cell free areas, which we call voids.
    The plot displayed the number of unassigned transcripts (uRNAs) in those areas.
    Dark areas signal potential segmentation issues or areas where cells might benefit from transcript reassignement. 

    Additional description:
    Unassigned transcripts are transcripts that could not be assigned to a cell in your data.
    This can always happen, but the rate of uRNAs is dictaed by the segmentation or segemtnation-free algorithm.
    """
    if os.path.exists(f"{figure_path}/voidqc/spatial_traingle_all_clsuters_log10_transcripts_counts_outside_cell.png"):
        img_void_sp = image_to_base64(
            f"{figure_path}/voidqc/spatial_traingle_all_clsuters_log10_transcripts_counts_outside_cell.png"
        )
        html_overview += f"""
        <h2>Void analysis</h2>
        <p>{description}</p>
        <img src="data:image/png;base64,{img_void_sp}" style="max-width:45%;">
        """

    ####################################################################################################################
    # Second page
    ####################################################################################################################

    html_subcluster = """
    <h1>Subcluster analysis</h1>
    <p>Currenlty spoQC pick the cell type cluster with the hightest number of cells.
    The subcluster can help to figure out if cell type cluster still contain potential quality impacted cell clusters that were not observable in the underscore(Overview) panel.
    
    Additional description:
    SpoQC perform Leiden clustering with a resoltion that generates up to 15 subclsuters.
    </p>
    """

    second_page_present = False
    if os.path.exists(f"{figure_path}/analysis/cluster/funkyheatmap/funkyheatmap_1.png"):
        second_page_present = True

    if second_page_present:
        img_funkyheatmap = image_to_base64(f"{figure_path}/analysis/cluster/funkyheatmap/funkyheatmap_1.png")
        html_subcluster += f"""
        <h2>Subcluster purity analysis</h2>
        <p>{description_summary}</p>
        <img src="data:image/png;base64,{img_funkyheatmap}" style="max-width:80%;">
        """

        with open(f"{figure_path}/analysis/cluster/umap/umap_plot_celltype.html") as f:
            html_annotation = f.read()
        with open(f"{figure_path}/analysis/cluster/umap/umap_plot_leiden.html") as f:
            html_leiden = f.read()
        html_subcluster += f"""
        <h2>Annotation and Leiden clustering</h2>
        <p>{description_umap}</p>
        <div style="display:inline-block; vertical-align:top; width:48%;">{html_annotation}</div>
        <div style="display:inline-block; vertical-align:top; width:48%;">{html_leiden}</div>
        """

        img_ump_hqr = image_to_base64(f"{figure_path}/analysis/cluster/umap/umap_plot_hqr_filtered_out.png")
        img_spa_hqr = image_to_base64(f"{figure_path}/analysis/cluster/scatterplot/scatterplot_hqr_filtered_out.png")
        html_subcluster += f"""
        <h2>Filter HQRs</h2>
        <p>{description_filter}</p>
        <img src="data:image/png;base64,{img_ump_hqr}" style="max-width:45%;">
        <img src="data:image/png;base64,{img_spa_hqr}" style="max-width:45%;">
        """

        img_bar_cpc = image_to_base64(f"{figure_path}/analysis/cluster/barplot/barplot_pct_control_probe_counts.png")
        html_subcluster += f"""
        <h2>Control probe counts per cluster</h2>
        <p>{description_bar_control_probes}</p>
        <img src="data:image/png;base64,{img_bar_cpc}" style="max-width:45%;">
        """

        img_bar_dc = image_to_base64(f"{figure_path}/analysis/cluster/barplot/barplot_pct_doublet_celltype.png")
        img_bar_dl = image_to_base64(f"{figure_path}/analysis/cluster/barplot/barplot_pct_doublet_leiden.png")
        html_subcluster += f"""
        <h2>Doublet counts per cluster</h2>
        <p>{description_doublet}</p>
        <img src="data:image/png;base64,{img_bar_dc}" style="max-width:45%;">
        <img src="data:image/png;base64,{img_bar_dl}" style="max-width:45%;">
        """

        img_bar_nfc = image_to_base64(f"{figure_path}/analysis/cluster/barplot/barplot_pct_nucleus_free_celltype.png")
        img_bar_nfl = image_to_base64(f"{figure_path}/analysis/cluster/barplot/barplot_pct_nucleus_free_leiden.png")
        html_subcluster += f"""
        <h2>Nucelus free cell counts per cluster</h2>
        <p>{description_nucelus_free}</p>
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
    Spatialplots for the individual Leiden clsuters identfied in the overview panel.
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
    Spatialplots for the individual cell type clsuters either provided as an input or identified by spoQC (optimized Leiden clustering).
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
    <h2>HQRs selected by spoQC and saved in the metadata</h1>
    """

    description = """
    Marked (red) cells belonging to HQCR's.
    Individual HQCR's are saved in the anndata and as a seperate metadata json file.
    Please read spoQC'documentation to find out more about the metadata.
    """
    img_bb_hqtr = image_to_base64(f"{figure_path}/hqcr/hqcr_ident/scatterplot_refined_qc_class.png")
    html_hqr += f"""
    <h2>HQCR selected by spoQC</h2>
    <p>{description}</p>
    <img src="data:image/png;base64,{img_cell}" style="max-width:45%;">
    """

    description = """
    Marked (red boxes) areas belonging to HQPR's.
    Individual HQPR's are saved in the anndata and as a seperate metadata file.
    Please read spoQC'documentation to find out more about the metadata.
    """
    folder_path = f'{figure_path}/hqpr'
    folder_path_continue = 'hqpr_bounding_box/'
    html_hqr += render_image_gallery("HQPRs selected by spoQC", "imageplot_marked_subfigures.png", stainings,
                                folder_path, folder_path_continue, description=description)


    description = """
    Marked (red boxes) areas belonging to HQTR's.
    Individual HQTR's are saved in the anndata and as a seperate metadata file.
    Please read spoQC'documentation to find out more about the metadata.
    """
    img_bb_hqtr = image_to_base64(f"{figure_path}/hqtr/hqtr_bounding_box/imageplot_marked_subfigures.png")
    html_hqr += f"""
    <h2>HQTR selected by spoQC</h2>
    <p>{description}</p>
    <img src="data:image/png;base64,{img_bb_hqtr}" style="max-width:45%;">
    """

    ####################################################################################################################
    # Sixth page
    ####################################################################################################################

    html_filters = ""

    description = """
    Individual HQCR filter.
    Black mared cells would be filtered out.

    Additional description:
    Cells are filter out (marked as low quality) if HQCR beliefs > 0.45.
    """
    img_ump_hqcr = image_to_base64(f"{figure_path}/analysis/overview/umap/umap_plot_hqcr_filtered_out.png")
    img_spa_hqcr = image_to_base64(f"{figure_path}/analysis/overview/scatterplot/scatterplot_hqcr_filtered_out.png")
    html_filters += f"""
    <h2>Filter HQTRs</h2>
    <p>{description}</p>
    <img src="data:image/png;base64,{img_ump_hqcr}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_spa_hqcr}" style="max-width:45%;">
    """

    description = """
    Individual HQPR filter.
    Black mared cells would be filtered out.

    Additional description:
    Cells are filter out (marked as low quality) if mean informative HQPR beliefs > 0.45.
    Since a segmented cell can have more than one pixel and transcript we calculate the mean inforamtive beliefs for those data modalities.
    The mean informative beliefs is the mean value of all bliefs with a value > 0.
    The mean informative belief thus prevents the issue of a zero inflated distribution that correlates with the size of the segmented cell.
    """
    html_filters += render_numbered_paired_image_gallery(
        f'{figure_path}/analysis/overview/umap/',
        f'{figure_path}/analysis/overview/scatterplot/',
        'umap_plot_hqpr',
        'scatterplot_hqpr',
        description=description
    )


    description = """
    Individual HQTR filter.
    Black mared cells would be filtered out.

    Additional description:
    Cells are filter out (marked as low quality) if mean informative HQTR beliefs > 0.45.
    Since a segmented cell can have more than one pixel and transcript we calculate the mean inforamtive beliefs for those data modalities.
    The mean informative beliefs is the mean value of all bliefs with a value > 0.
    The mean informative belief thus prevents the issue of a zero inflated distribution that correlates with the size of the segmented cell.
    """
    img_ump_hqtr = image_to_base64(f"{figure_path}/analysis/overview/umap/umap_plot_hqtr_filtered_out.png")
    img_spa_hqtr = image_to_base64(f"{figure_path}/analysis/overview/scatterplot/scatterplot_hqtr_filtered_out.png")
    html_filters += f"""
    <h2>{description}>
    <p>TODO: describe this section</p>
    <img src="data:image/png;base64,{img_ump_hqtr}" style="max-width:45%;">
    <img src="data:image/png;base64,{img_spa_hqtr}" style="max-width:45%;">
    """

    ####################################################################################################################
    # Seventh page
    ####################################################################################################################

    html_hqcr = ""

    html_hqcr += f"""
    <h1>All HQCR metrics used by spoQC</h1>
    <p>Boxplots showcase Leiden clusters identified in the underscore(Overview) panel.</p>
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
    <p>Boxplots showcase Leiden clusters identified in the underscore(Overview) panel.</p>
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
    <p>Boxplots showcase Leiden clusters identified in the underscore(Overview) panel.</p>
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
        if os.path.exists(f"{figure_path}/hqpr/hqpr_celltype/{s}/celltype_qc_analysis.html"):
            with open(f"{figure_path}/hqpr/hqpr_celltype/{s}/celltype_qc_analysis.html") as f:
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
