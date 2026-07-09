import ovrlpy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from ... import helperfuncs

# window_sizes = for plotting. You can selected more windowsizes. This is just to zoom in or out for double plots.
# num_doublet = is just the amount of doublet that will be plottet as examples.
# distance = Threshold to use to call a cell a doublet cell if its close to the detected doublet signal of ovrlpy.
def calc_doublet_score(
        sdata,
        figure_path,
        spoqc_tmp_folder,
        threads,
        key_transcripts, 
        n_expected_celltypes,
        cell_diameter,
        minimum_signal_strength,
        integrity_sigma,
        signal_threshold,
        window_sizes,
        num_doublet,
        distance_thresh
):

    transcript_coordinates_df = sdata.points[key_transcripts].compute()
    transcript_coordinates_df = transcript_coordinates_df.rename(columns={'feature_name': 'gene'})

    # ovrlpy does a werid thing to overwrite the coordinates and set the origin to 0.0.
    # I bring the doublet coordinates into the original data points coord system.
    # So I have to save the origin.
    min_x = min(transcript_coordinates_df['x'])
    min_y = min(transcript_coordinates_df['y'])

    n_components = n_expected_celltypes if (n_expected_celltypes and n_expected_celltypes > 0) else 30
    ovrlp = ovrlpy.Ovrlp(
        transcript_coordinates_df,
        min_distance=cell_diameter,
        n_components=n_components,
        n_workers=threads,
    )
    ovrlp.analyse()

    doublet_df = ovrlp.detect_doublets(
        min_signal=minimum_signal_strength,
        integrity_sigma=integrity_sigma,
    ).to_pandas()

    plt.scatter(
        doublet_df["x"],
        doublet_df["y"],
        c=doublet_df["integrity"],
        s=1,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    plt.gca().set_aspect("equal")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f'{figure_path}/scatter_signal_integrity.png')
    plt.savefig(f'{figure_path}/scatter_signal_integrity.pdf')
    plt.close()

    transcripts_processed = ovrlp.transcripts.to_pandas()
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection="3d")
    for i in range(-2, 3):
        subset = transcripts_processed[
            (transcripts_processed['z'] - transcripts_processed['z_center']).between(i, i + 1)
        ]
        # downsample the number of transcripts
        subset = subset[::100]

        ax.scatter(subset["x"], subset["y"], i, s=1, alpha=0.1)
    ratio = transcripts_processed["x"].max() / transcripts_processed["y"].max()
    ax.set_box_aspect([ratio, 1, 0.75])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout(pad=2)
    plt.savefig(f'{figure_path}/scatter_signal_integrity_3d.png')
    plt.savefig(f'{figure_path}/scatter_signal_integrity_3d.pdf')
    plt.close()

    # Integrity density plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(doublet_df["integrity"], kde=True, bins=30, color="blue", alpha=0.6)
    plt.title("Density Histogram of Integrity")
    plt.xlabel("Integrity")
    plt.ylabel("Density")

    # Signal density plot
    plt.subplot(1, 2, 2)
    sns.histplot(doublet_df["signal"], kde=True, bins=30, color="green", alpha=0.6)
    plt.title("Density Histogram of Signal")
    plt.xlabel("Signal")
    plt.ylabel("Density")
    plt.tight_layout()

    plt.savefig(f'{figure_path}/histogram_signal_integrity_and_signal.png')
    plt.savefig(f'{figure_path}/histogram_signal_integrity_and_signal.pdf')
    plt.close()
    fig = ovrlpy.plot_signal_integrity(ovrlp, signal_threshold=signal_threshold)
    plt.tight_layout()
    plt.savefig(f'{figure_path}/spatial_signal_integrity_map.png')
    plt.savefig(f'{figure_path}/spatial_signal_integrity_map.pdf')
    plt.close()

    if ( len(doublet_df) < num_doublet ):
        num_doublet = len(doublet_df)

    for i in range(0, num_doublet):

        doublet_case = i
        x, y = doublet_df.loc[doublet_case, ["x", "y"]]
        fig = ovrlpy.plot_region_of_interest(
            ovrlp,
            x,
            y,
            window_size=window_sizes[0],
        )
        # Adjust layout to prevent overlap
        fig.tight_layout()
        fig.savefig(f'{figure_path}/doublet_case_{i}_zoomed.png')
        fig.savefig(f'{figure_path}/doublet_case_{i}_zoomed.pdf')

        x, y = doublet_df.loc[doublet_case, ["x", "y"]]
        fig = ovrlpy.plot_region_of_interest(
            ovrlp,
            x,
            y,
            window_size=window_sizes[1],
        )
        fig.tight_layout()
        fig.savefig(f'{figure_path}/doublet_case_{i}.png')
        fig.savefig(f'{figure_path}/doublet_case_{i}.pdf')

    # Link doublet detection back to spatial.
    # Based on a distance parameter say if a cell might be a doublet or not.
    cell_dobulet_df = pd.DataFrame({
        'x': [poly.centroid.x for poly in sdata['cell_boundaries']['geometry']],
        'y': [poly.centroid.y for poly in sdata['cell_boundaries']['geometry']],
        'doublet': [False] * sdata['table'].n_obs,
        'wdoublet': [0] * sdata['table'].n_obs,
        'doublet_distance': [100_000.0] * sdata['table'].n_obs
    })

    corrected_doublet_df = doublet_df.copy()

    # Bring doublets back to the original coordinate system.
    corrected_doublet_df['x'] = doublet_df['x'] + min_x
    corrected_doublet_df['y'] = doublet_df['y'] + min_y

    final_distances = np.array([100_000.0] * sdata['table'].n_obs)
    for i, doublet in corrected_doublet_df.iterrows():
        x1, y1 = doublet['x'], doublet['y']
        distances = np.sqrt((cell_dobulet_df['x'] - x1)**2 + (cell_dobulet_df['y'] - y1)**2)
        final_distances = np.minimum(final_distances, distances) 
        cell_dobulet_df.loc[distances <= distance_thresh, 'doublet'] = True
        cell_dobulet_df.loc[distances <= distance_thresh, 'wdoublet'] = 1
    cell_dobulet_df['doublet_distance'] = final_distances

    # Plot doublet density
    helperfuncs.plot_scatter_density_df(
        cell_dobulet_df,
        figure_path,
        'doublet',
        'doublet',
        'wdoublet',
        ['lightblue', 'black'],
        None
    )

    # Write into sdata
    sdata['table'].obs['doublet'] = np.array(cell_dobulet_df['doublet'])
    sdata['table'].obs['wdoublet'] = np.array(cell_dobulet_df['wdoublet'])
    sdata['table'].obs['doublet_distance'] = np.array(cell_dobulet_df['doublet_distance'])

    # Have to call this again because overlpy corrects also the transcript coordinates
    transcript_coordinates_df = sdata.points[key_transcripts].compute()

    # Detect transcript that might belong to doublets
    transcript_doublet = np.array([False] * len(transcript_coordinates_df))
    transcript_wdoublet = np.array([0] * len(transcript_coordinates_df))
    for i, doublet in corrected_doublet_df.iterrows():
        x1, y1 = doublet['x'], doublet['y']
        distances = np.sqrt((transcript_coordinates_df['x'] - x1)**2 + (transcript_coordinates_df['y'] - y1)**2)
        transcript_doublet[distances <= distance_thresh] = True
        transcript_wdoublet[distances <= distance_thresh] = 1

    # Write out transcript doublet information for later usage
    transcript_doublet_df = pd.DataFrame({
        'doublet': transcript_doublet,
        'wdoublet': transcript_wdoublet,
    })
    transcript_doublet_df.index = transcript_coordinates_df.index

    helperfuncs.df_to_parquet(transcript_doublet_df, 'doublet', spoqc_tmp_folder, [], 'transcripts')