import numpy as np
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px   # plotly
import pandas as pd
import matplotlib.path
from matplotlib.collections import PolyCollection
import seaborn as sns
import concurrent.futures
import os

from ... import helperfuncs

def count_stuff_in_triangles(triangle_array, stuff):
    counts = np.zeros(len(triangle_array), dtype=int)
    indices_list = []

    for i, triangle in enumerate(triangle_array):
        path = matplotlib.path.Path(triangle) # Create a polygon path for the triangle
        indices = np.where(path.contains_points(stuff))
        indices_list.append(indices)
        counts[i] = np.count_nonzero(path.contains_points(stuff))

    return counts, indices_list

# Define a single worker function
def process_triangle_block(block_index, triangle_block, stuff):
    counts = []
    indices = []

    for triangle in triangle_block:
        path = matplotlib.path.Path(triangle)
        mask = path.contains_points(stuff)
        indices.append(np.where(mask))
        counts.append(np.count_nonzero(mask))
    
    return counts, indices, block_index

def count_stuff_in_triangles_fast(triangle_array, stuff, threads):
    count_list = [None] * threads
    indices_list = [None] * threads
    thread_split_triangle_array = helperfuncs.thread_split_list(triangle_array, threads)
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(process_triangle_block,
                    block_index,
                    triangle_block,
                    stuff
                ) for block_index, triangle_block in enumerate(thread_split_triangle_array)]
        for future in concurrent.futures.as_completed(futures):
            results = future.result()
            count_list[results[2]] = results[0]
            indices_list[results[2]] = results[1]

    # Unpack stuff
    unpacked_count_list_sorted = [item for sublist in count_list for item in sublist]
    unpacked_indices_list_sorted = [item for sublist in indices_list for item in sublist]
    return unpacked_count_list_sorted, unpacked_indices_list_sorted


def build_triangle_graph_using_neighbors(delaunay, points):
    triangles = delaunay.simplices
    neighbors = delaunay.neighbors
    edge_length_list = []
    triangle_dict = {}

    for i, triangle in enumerate(triangles):
        for k, neighbor_idx in enumerate(neighbors[i]):
            if neighbor_idx != -1 and neighbor_idx > i:  # Only process once
                # Find the common edge
                common_vertices = set(triangle) & set(triangles[neighbor_idx])
                if len(common_vertices) == 2:
                    a, b = tuple(common_vertices)
                    dist = np.linalg.norm(points[a] - points[b])
                    edge_length_list.append(dist)
                    
                    if i not in triangle_dict:
                        triangle_dict[i] = []
                    if neighbor_idx not in triangle_dict:
                        triangle_dict[neighbor_idx] = []

                    triangle_dict[i].append((neighbor_idx, dist))
                    triangle_dict[neighbor_idx].append((i, dist))
    return edge_length_list, triangle_dict, triangles

# Interesting voids hold still a lot of information.
# Less interesting voids almost have nothing in there.
def calc_void(
        sdata,
        figure_path,
        spoqc_tmp_folder,
        min_n_triangles_in_cluster,
        contaminant_list,
        threads,
        *,
        debug=False,
        flip=False
    ):

    timer = helperfuncs.Timer()

    ################################
    #### Delaunay triangulation ####
    ################################
    print("[NOTE] Start void QC")
    timer.start()

    # Get cellcoordinates
    points = list(zip(sdata['table'].obsm['spatial'][:,0], sdata['table'].obsm['spatial'][:,1]))
    points = np.array(points)

    # Delaunay triangulation
    delaunay = spatial.Delaunay(points)
    timer.stop()

    #####################################
    #### Built traingular dictionary ####
    #####################################
    print("[NOTE] Built traingular dictionary")
    timer.start()
    edge_length_list, triangle_dict, triangles  = build_triangle_graph_using_neighbors(delaunay, points)
    num_triangles = len(triangles)
    timer.stop()

    # Set as threshold for edge length the 3rd quartile.
    min_edge_length = np.quantile(edge_length_list, q=.75)

    # Create traingular graph.
    traingular_graph = nx.Graph()

    ################################
    #### Built triangular graph ####
    ################################
    print("[NOTE] Built triangular graph")
    timer.start()
    # A triangle is a vertex.
    # Add two neigbhouring triangles together by en edge if the edge is bigger or equal than threshold.
    # Thus two vertices are connected and edge weighter by the triangle edge (cell to cell distance).
    # Tirangle cluster = all connected triangles
    for triangle, neighbors in triangle_dict.items():
        for neighbor, edge_length in neighbors:
            if edge_length >= min_edge_length:
                traingular_graph.add_edge(triangle, neighbor, weight=edge_length)
    timer.stop()

    # Just for testing
    # Draw the graph
    if ( debug ):
        print("[NOTE] Draw graph")
        plt.figure(figsize=(6, 6))
        pos = nx.spring_layout(traingular_graph, seed=42)  # Layout for visualization
        nx.draw(
            traingular_graph,
            pos,
            with_labels=True,
            node_size=1000,
            node_color="lightblue",
            edge_color="gray",
            font_size=12
        )

        # Draw edge labels (edge lengths)
        edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in traingular_graph.edges(data=True)}
        nx.draw_networkx_edge_labels(traingular_graph, pos, edge_labels=edge_labels, font_size=10)

        plt.title("Triangle Connectivity Graph")
        plt.savefig(f'{figure_path}/traingle_connectivit_graph.png', bbox_inches='tight', dpi=300)
        plt.savefig(f'{figure_path}/traingle_connectivit_graph.pdf', bbox_inches='tight', dpi=300)
        plt.close()

    ###########################
    #### Check Edge length ####
    ###########################
    # Check the distribution of the edge length to verify the threholds for the tirangular cluster graph.
    figures = []

    fig = px.box(x=edge_length_list, width=800, height=800)
    fig.update_layout(
        title=f"Total distribution of edge lengths"
    )

    helperfuncs.apply_general_plotly_layout(fig, True)

    figures.append(fig)
    fig.write_image(f"{figure_path}/boxplot_edge_lengths.png", scale=3)
    fig.write_image(f"{figure_path}/boxplot_edge_lengths.pdf", scale=3)

    with open(f'{figure_path}/void.html', 'w') as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    ##################################
    #### Find triangular clusters ####
    ##################################
    print("[NOTE] Find triangular clusters")
    timer.start()

    # Get triangular clusters
    triangular_clsuters = [list(comp) for comp in nx.connected_components(traingular_graph)]

    # Get point (cell) coordinates for the 3 points of each triangle
    triangles_pointcoords = []
    for triangle in triangles:
        triangles_pointcoords.append([list(points[id]) for id in triangle])
    triangles_pointcoords = np.array(triangles_pointcoords)

    timer.stop()

    #####################################
    #### Built traingular data frame ####
    #####################################
    print("[NOTE] Built traingular data frame")
    timer.start()

    # Built trainglular cluser dataframe that holds information for each cluster.
    triangular_cluster_ids = [-1] * num_triangles
    triangular_cluster_colors = ['#FFFFFF'] * num_triangles
    tcluster_colors = helperfuncs.generate_distinct_colors_with_jitter(len(triangular_clsuters))
    triangular_cluster_ntriangles = [1] * num_triangles

    for i, cluster in enumerate(triangular_clsuters):
        for t_id in cluster:
            triangular_cluster_ids[t_id] = i
            triangular_cluster_colors[t_id] = tcluster_colors[i]
            triangular_cluster_ntriangles[t_id] = len(cluster)

    triangles_df = pd.DataFrame({
        'id': [i for i in range(0,len(triangles))],
        'triangular_cluster_ids': triangular_cluster_ids,
        'triangular_cluster_colors': triangular_cluster_colors,
        'triangular_cluster_ntriangles': triangular_cluster_ntriangles
    })

    timer.stop()

    ########################
    #### Plot triangles ####
    ########################
    print("[NOTE] Plot triangles")
    timer.start()

    # Create a PolyCollection
    triangles_plotting = [points[simplex] for simplex in triangles]
    collection = PolyCollection(
        triangles_plotting,
        facecolors=triangular_cluster_colors,
        edgecolors="gray",
        alpha=0.8,
        linewidths=0
    )

    # Plot all triangles colored by each clsuter id unless they are not associated with a cluster, then they are
    # just gray.
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_collection(collection)
    ax.scatter(points[:, 0], points[:, 1], c="blue", marker="o", label="Points", s=0.01)
    if ( flip ):
        ax.invert_yaxis()
    ax.set_title("Largest Enclosed Empty Patches")
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(f'{figure_path}/spatial_traingle_all_clsuters.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{figure_path}/spatial_traingle_all_clsuters.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    timer.stop()

    ####################################
    #### Triangle cluster filtering ####
    ####################################
    print(f"[NOTE] Apply triangle cluster filtering with min {min_n_triangles_in_cluster} traingles.")
    timer.start()

    # Now check and just select voids that have a minimal number of triangels.
    triangle_filter = triangles_df['triangular_cluster_ntriangles'] < min_n_triangles_in_cluster
    triangles_df.loc[triangle_filter, 'triangular_cluster_colors'] = '#FFFFFF'
    triangular_cluster_colors = list(triangles_df['triangular_cluster_colors'])
    triangles_df.loc[triangle_filter, 'triangular_cluster_ids'] = -1
    triangular_cluster_ids = list(triangles_df['triangular_cluster_ids'])

    # Make a plot with triangle clsuter numbers and without.
    for with_numbers in [True, False]:

        collection = PolyCollection(
            triangles_plotting,
            facecolors=triangular_cluster_colors,
            edgecolors="gray",
            alpha=0.8,
            linewidths=0.5  # optional: thinner edges to speed up even more
        )

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.add_collection(collection)

        # Plot each triangle with the corresponding color
        id_list = []
        for simplex, id in zip(triangles, triangular_cluster_ids):
            if ( id != -1 and with_numbers and id not in id_list):
                triangle = points[simplex]
                centroid = np.mean(triangle, axis=0) # Compute centroid of the triangle
                ax.text(
                    centroid[0],
                    centroid[1],
                    str(id),
                    color="black",
                    fontsize=5,
                    ha="center",
                    va="center",
                    fontweight="bold"
                )
                id_list.append(id)

        ax.scatter(points[:, 0], points[:, 1], c="blue", marker="o", label="Points", s=0.01)
        if ( flip ):
            plt.gca().invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        plt.title("Largest Enclosed Empty Patches")
        plt.savefig(
            f'{figure_path}/spatial_traingle_filtered_clusters_{with_numbers}.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.savefig(
            f'{figure_path}/spatial_traingle_filtered_clusters_{with_numbers}.pdf',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
    
    timer.stop()

    ##################################################
    #### Triangle cluster counting (outside cell) ####
    ##################################################
    transcripts_df = sdata['transcripts'].compute()

    # Doublet information has to be loaded here because I fill filter transcripts_df.
    doublet_check = False
    if ( os.path.exists(f'{spoqc_tmp_folder}/doublet_output_transcripts.parquet') ):
        doublet_check = True
    if ( doublet_check ):
        print("[NOTE] Load doublet information for void QC")
        tmp_data = pd.read_parquet(f'{spoqc_tmp_folder}/doublet_output_transcripts.parquet')
        transcripts_df = transcripts_df.join(tmp_data, how='left')

    ###### Count
    # Count transcripts that are outside the cell for each triangle
    print(f'[NOTE] counting transcripts that are outside the cell for {len(triangles_pointcoords)} triangles')
    transcripts_outside_cell_df = transcripts_df.loc[transcripts_df['cell_id'] == -1]
    transcript_ocell_coords = np.array(list(zip(transcripts_outside_cell_df['x'], transcripts_outside_cell_df['y'])))
    timer.start()
    counts, indices = count_stuff_in_triangles_fast(triangles_pointcoords, transcript_ocell_coords, threads)
    timer.stop()
    triangles_df['transcripts_counts_outside_cell'] = counts

    ###### Calculate
    # Calculate the variance of z coordiantes for each triangle.
    # print(f'[NOTE] Calculate the variance of z coordiantes for each triangle.')
    # timer.start()
    # triangle_z_var = [-1] * len(triangles_df)
    # for i in range(0, len(triangles_df)):
    #     # Something weird is heppening here
    #     if ( len(indices[i]) > 0 ):
    #         triangle_z_var[i] = np.var(transcripts_df.iloc[indices[i]]['z'])
    #     else:
    #         triangle_z_var[i] = 0
    # triangles_df['triangle_z_var'] = triangle_z_var
    # timer.stop()

    ###### Count
    # Count transcripts that belong to doublet locations for each triangle.
    # We just consider transcripts outside the cell.
    if ( doublet_check ):
        print(f'[NOTE] counting transcripts belong to doublet regions for {len(triangles_pointcoords)} triangles')
        transcripts_doublet_df = transcripts_outside_cell_df.loc[transcripts_outside_cell_df['doublet']]
        if ( len(transcripts_doublet_df) > 0 ):
            transcript_doublet_coords = np.array(list(zip(transcripts_doublet_df['x'], transcripts_doublet_df['y'])))
            counts, indices = count_stuff_in_triangles_fast(triangles_pointcoords, transcript_doublet_coords, threads)
            triangles_df['transcripts_counts_doublets'] = counts
        else:
            print(f'[NOTE] no doublets found')
            triangles_df['transcripts_counts_doublets'] = np.zeros(len(triangles_df))

    ###### Count
    # Count transcripts that belong to a contaminant.
    # We just consider transcripts outside the cell.
    if ( len(contaminant_list) > 0 ):
        for contaminant in contaminant_list:
            print(f'[NOTE] counting transcripts belong to contaminatn {contaminant} ' + \
                  f'for {len(triangles_pointcoords)} triangles')
            transcripts_contaminant_df = transcripts_outside_cell_df.loc[
                transcripts_outside_cell_df['feature_name'] == contaminant
            ]
            if ( len(transcripts_contaminant_df) > 0 ):
                transcript_contaminant_coords = np.array(list(zip(transcripts_contaminant_df['x'],
                                                              transcripts_contaminant_df['y'])))
                counts, indices = count_stuff_in_triangles_fast(
                    triangles_pointcoords,
                    transcript_contaminant_coords,
                    threads
                )
                triangles_df[f'transcripts_counts_{contaminant}'] = counts
            else:
                print(f'[NOTE] no transcripts for contaminant found')
                triangles_df[f'transcripts_counts_{contaminant}'] = np.zeros(len(triangles_df))

    ###### Count
    # Count nuclei in triangles. 
    # This is more of a sanity.
    print(f'[NOTE] counting nuclei for {len(triangles_pointcoords)} triangles')
    timer.start()
    nuclei_centoid_coords = np.array( 
        list( 
            zip(
                [poly.centroid.x for poly in sdata['nucleus_boundaries']['geometry']],
                [poly.centroid.y for poly in sdata['nucleus_boundaries']['geometry']]
            ) 
        ) 
    )
    timer.stop()
    counts, indices = count_stuff_in_triangles_fast(triangles_pointcoords, nuclei_centoid_coords, threads)

    # Lets just consider nulcei_counts > 3 because my triangle consists of 3 cells.
    # Of course this does not consider cells with multi-nucei or multiplets.
    counts = np.array(counts) - 3
    counts = [x if x > 0 else 0 for x in counts]
    triangles_df['nuclei_counts'] = counts

    ###################
    #### Bar plots ####
    ###################

    # Buil up traingular clsuter data frame
    check_triangle_cluster_ids = list(set(triangular_cluster_ids))
    check_triangle_cluster_ids.remove(-1)
    check_triangle_cluster_ids.sort()

    transcripts_counts = []
    transcripts_counts_doublets = []
    # max_of_vars_z = []
    nuclei_count = []
    colors = []
    contaminants_dict = {}

    if ( len(contaminant_list) > 0 ):
        for contaminant in contaminant_list:
            contaminants_dict[contaminant] = []

    for cid in check_triangle_cluster_ids:
        idxs = triangles_df['triangular_cluster_ids'] == cid
        transcripts_counts.append( sum(triangles_df.loc[idxs, 'transcripts_counts_outside_cell']) )
        # max_of_vars_z.append( np.max(triangles_df.loc[idxs, 'triangle_z_var']) )
        nuclei_count.append( sum(triangles_df.loc[idxs, 'nuclei_counts']) )
        colors.append( list(set(triangles_df.loc[idxs, 'triangular_cluster_colors']))[0] )
        if ( doublet_check ):
            transcripts_counts_doublets.append( sum(triangles_df.loc[idxs, 'transcripts_counts_doublets']) )
        if ( len(contaminant_list) > 0 ):
            for contaminant in contaminant_list:
                contaminants_dict[contaminant].append( sum(
                    triangles_df.loc[idxs, f'transcripts_counts_{contaminant}']
                ) )

    triangle_cluster_df = pd.DataFrame({
        'triangular_cluster_ids': check_triangle_cluster_ids,
        'log10_transcripts_counts_outside_cell': np.log10(( np.array(transcripts_counts) + 1) ),
        # 'max_of_vars_z': max_of_vars_z,
        'nuclei_count': nuclei_count
    })

    if ( doublet_check ):
        triangle_cluster_df['transcripts_counts_doublets'] = transcripts_counts_doublets
    if ( len(contaminant_list) > 0 ):
        for contaminant in contaminant_list:
            triangle_cluster_df[f'transcripts_counts_{contaminant}'] = contaminants_dict[contaminant]

    # Create the bar plot
    cats_to_check = list(triangle_cluster_df.columns)
    cats_to_check.remove('triangular_cluster_ids')
    for y in cats_to_check:
        plt.figure(figsize=(8, 6))
        sns.barplot(triangle_cluster_df, x="triangular_cluster_ids", y=y, palette=colors)

        # Add title and labels
        plt.xlabel("Void number")
        plt.ylabel(y)

        plt.savefig(f'{figure_path}/barplot_void_{y}.png', bbox_inches='tight', dpi=300)
        plt.savefig(f'{figure_path}/barplot_void_{y}.pdf', bbox_inches='tight', dpi=300)
        plt.close()

    #################################################
    #### Triangle cluster counting (convex hull) ####
    #################################################

    ###### Count
    print("[NOTE] Count transcripts outside the cell in the convexhull of the cell")

    timer.start()
    counts_array = np.array(triangles_df['transcripts_counts_outside_cell'])

    # Create a mapping of cell index to triangles for faster lookup
    triangle_indices = {}
    for t_idx, triangle in enumerate(triangles):
        for vertex in triangle:
            triangle_indices.setdefault(vertex, []).append(t_idx)

    convexhull_outside_transcripts = [
        counts_array[triangle_indices[i]].sum() if i in triangle_indices else 0
        for i in range(sdata['table'].n_obs)
    ]

    sdata['table'].obs['convexhull_outside_trnascripts'] = convexhull_outside_transcripts
    timer.stop()

    ###### Count
    # print("[NOTE] Count all transcripts in the convexhull of the cell")

    # # For convexhull investigation lets also take all transcripts into account.
    # # This can help to identify interesting regions.
    # all_transcripts_df = sdata['transcripts'].compute()
    # all_transcript_coords = np.array(list(zip(all_transcripts_df['x'], all_transcripts_df['y'])))

    # print(f'[NOTE] counting transcripts for {len(triangles_pointcoords)} triangles')

    # counts_all, indices_all = count_stuff_in_triangles_fast(triangles_pointcoords, all_transcript_coords, threads)
    # triangles_df['transcripts_counts_all'] = counts_all

    # convexhull_all_transcripts = []
    # counts_array = np.array(triangles_df['transcripts_counts_all'])

    # convexhull_all_transcripts = [
    #     counts_array[triangle_indices[i]].sum() if i in triangle_indices else 0
    #     for i in range(sdata['table'].n_obs)
    # ]

    # sdata['table'].obs['convexhull_all_trnascripts'] = convexhull_all_transcripts

    ####################
    #### More plots ####
    ####################

    print("[NOTE] More triangular plots")
    timer.start()
    for cat in cats_to_check:
        triangle_cluster_df[f'triangular_cluster_colors_{cat}'] = helperfuncs.values_to_hex_gradient(
                                                                                            triangle_cluster_df[cat], 
                                                                                            'hot', 
                                                                                            reverse=True
                                                                                            )

        # Apply the map
        triangles_df[f'triangular_cluster_colors_{cat}'] = ['#FFFFFF'] * num_triangles
        
        for cluster_id in triangle_cluster_df['triangular_cluster_ids']:
            mask_triangles = triangles_df['triangular_cluster_ids'] == cluster_id
            mask_triangle_clusters = triangle_cluster_df['triangular_cluster_ids'] == cluster_id
            triangles_df.loc[mask_triangles, f'triangular_cluster_colors_{cat}'] = list(triangle_cluster_df.loc[mask_triangle_clusters, f'triangular_cluster_colors_{cat}'])[0]

        triangles_df.loc[triangle_filter, f'triangular_cluster_colors_{cat}'] = '#FFFFFF'

        fig, ax = plt.subplots(figsize=(8, 8))
        triangles_plotting = [points[simplex] for simplex in triangles]

        collection = PolyCollection(
            triangles_plotting,
            facecolors=triangles_df[f'triangular_cluster_colors_{cat}'],
            edgecolors="gray",
            alpha=0.8,
            linewidths=0
        )

        ax.add_collection(collection)
        ax.scatter(points[:, 0], points[:, 1], c="blue", marker="o", label="Points", s=0.01)
        if ( flip ):
            ax.invert_yaxis()
        ax.set_title(f"{cat}")
        ax.set_aspect('equal', adjustable='box')

        vmin = triangle_cluster_df[cat].min()
        vmax = triangle_cluster_df[cat].max()
        sm = plt.cm.ScalarMappable(cmap=plt.cm.hot_r, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=cat)

        plt.savefig(f'{figure_path}/spatial_traingle_all_clsuters_{cat}.png', bbox_inches='tight', dpi=300)
        plt.savefig(f'{figure_path}/spatial_traingle_all_clsuters_{cat}.pdf', bbox_inches='tight', dpi=300)
        plt.close()


    print("[NOTE] Void QC done!")
    timer.stop()
  
# %%
