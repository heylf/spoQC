import ovrlpy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from ... import helperfuncs
from ... import _ovrlpy_fast
from ...core import spatial, transcripts
from spoqc.core.figures import save_figure

# Initial doublet_distance of every cell, kept where no doublet is closer.
NO_DOUBLET_DISTANCE = 100_000.0


def flag_transcripts_near_doublets(
    transcript_coordinates_df, corrected_doublet_df, distance_thresh, threads
):
    """
    Flag transcripts within distance_thresh of any doublet.

    The original per-doublet pandas expression ran in the transcripts' float32, with each
    doublet coordinate cast to float32; pairs_within evaluates exactly that. (With numexpr
    installed, pandas would have evaluated the original's full-length arithmetic in float64.)
    """
    transcript_xy = transcript_coordinates_df[["x", "y"]].to_numpy()
    transcript_pos, _ = spatial.pairs_within(
        transcript_xy,
        corrected_doublet_df[["x", "y"]].to_numpy(),
        distance_thresh,
        threads,
        dtype=transcript_xy.dtype,
    )
    transcript_doublet = np.zeros(len(transcript_xy), dtype=bool)
    transcript_doublet[transcript_pos] = True
    return transcript_doublet, transcript_doublet.astype(int)


def write_transcript_doublets(sdata, corrected_doublet_df, distance_thresh, threads, spoqc_tmp_folder):
    """Flag the transcripts near a doublet and write them, indexed like the transcripts element.

    The coordinates come from the run's transcripts (ovrlpy never changes the caller's frame,
    so the original second .compute() was not needed).
    """
    # Detect transcript that might belong to doublets
    transcript_doublet, transcript_wdoublet = flag_transcripts_near_doublets(
        transcripts.load_transcripts(sdata, ['x', 'y']).to_pandas(), corrected_doublet_df, distance_thresh, threads
    )

    # Write out transcript doublet information for later usage
    transcript_doublet_df = pd.DataFrame({
        'doublet': transcript_doublet,
        'wdoublet': transcript_wdoublet,
    })
    transcript_doublet_df.index = transcripts.transcript_index(sdata)  # labels, not 0..n-1 after a crop

    helperfuncs.df_to_parquet(transcript_doublet_df, 'doublet', spoqc_tmp_folder, [], 'transcripts')


def flag_cells_near_doublets(cell_xy, doublet_xy, distance_thresh, threads):
    """
    Distance of every cell to its nearest doublet (capped at NO_DOUBLET_DISTANCE), and
    whether a doublet lies within distance_thresh. The minimum is exact (spatial.nearest),
    and min <= thresh holds exactly when some doublet is within thresh.
    """
    _, distance = spatial.nearest(cell_xy, doublet_xy, threads)
    doublet = distance <= distance_thresh
    return doublet, doublet.astype(int), np.minimum(NO_DOUBLET_DISTANCE, distance)



def downsample_transcript_layers(transcripts, layers=range(-2, 3), stride=100):
    """Rows for the 3D depth scatter, one (x, y) pair per depth layer, plus the aspect ratio.

    `transcripts` is ovrlpy's polars frame holding every transcript in the sample. It used
    to be materialised in full with .to_pandas() purely to feed this plot, which then keeps
    1/stride of it and reads only x and y. A tree-RSS trace measured that one call adding
    ~48 GB in under 10 s on a 229,970-cell 5K-panel sample, taking the run to 217.79 GB and
    past the memory budget.

    Filtering and striding in polars first, converting only the surviving two columns, feeds
    the scatter identical rows in identical order:
      - pandas `.between(i, i + 1)` is inclusive at both ends, hence >= / <=
      - the stride is applied AFTER the filter, so the same rows survive
      - polars `filter` preserves row order, like a pandas boolean mask
      - the ratio uses FULL-column maxima, not the downsampled subset

    tests/test_doublet_3d_plot_equivalence.py asserts this against the original verbatim.
    """
    depth = transcripts['z'] - transcripts['z_center']
    per_layer = []
    for i in layers:
        subset = transcripts.filter((depth >= i) & (depth <= i + 1))
        # downsample the number of transcripts
        subset = subset.gather_every(stride).select(['x', 'y']).to_pandas()
        per_layer.append((i, subset))
    ratio = transcripts["x"].max() / transcripts["y"].max()
    return per_layer, ratio


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
        distance_thresh,
):

    transcript_coordinates_df = sdata.points[key_transcripts].compute()
    transcript_coordinates_df = transcript_coordinates_df.rename(columns={'feature_name': 'gene'})

    # ovrlpy does a werid thing to overwrite the coordinates and set the origin to 0.0.
    # I bring the doublet coordinates into the original data points coord system.
    # So I have to save the origin.
    min_x = min(transcript_coordinates_df['x'])
    min_y = min(transcript_coordinates_df['y'])

    n_components = 30
    if (n_expected_celltypes and n_expected_celltypes > 0):
        n_components = n_expected_celltypes
    if ( sdata['table'].n_obs < 500 ):
        n_components = 10
    if ( sdata['table'].n_obs < 100 ):
        n_components = 2

    # Accumulate ovrlpy's per-gene embedding over each gene's nonzero rows only, rather
    # than a fresh (n_pixels, n_components) temporary per gene. Bit-identical; raises on
    # any ovrlpy this shim was not written against. See spoqc/_ovrlpy_fast.py.
    _ovrlpy_fast.install()

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
    save_figure(plt.gcf(), f'{figure_path}/scatter_signal_integrity.png', f'{figure_path}/scatter_signal_integrity.pdf')
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection="3d")
    per_layer, ratio = downsample_transcript_layers(ovrlp.transcripts)
    for i, subset in per_layer:
        ax.scatter(subset["x"], subset["y"], i, s=1, alpha=0.1)
    ax.set_box_aspect([ratio, 1, 0.75])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.tight_layout(pad=2)
    save_figure(plt.gcf(), f'{figure_path}/scatter_signal_integrity_3d.png', f'{figure_path}/scatter_signal_integrity_3d.pdf')
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

    save_figure(plt.gcf(), f'{figure_path}/histogram_signal_integrity_and_signal.png', f'{figure_path}/histogram_signal_integrity_and_signal.pdf')
    plt.close()
    fig = ovrlpy.plot_signal_integrity(ovrlp, signal_threshold=signal_threshold)
    plt.tight_layout()
    save_figure(plt.gcf(), f'{figure_path}/spatial_signal_integrity_map.png', f'{figure_path}/spatial_signal_integrity_map.pdf')
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
        save_figure(fig, f'{figure_path}/doublet_case_{i}_zoomed.png', f'{figure_path}/doublet_case_{i}_zoomed.pdf')

        x, y = doublet_df.loc[doublet_case, ["x", "y"]]
        fig = ovrlpy.plot_region_of_interest(
            ovrlp,
            x,
            y,
            window_size=window_sizes[1],
        )
        fig.tight_layout()
        save_figure(fig, f'{figure_path}/doublet_case_{i}.png', f'{figure_path}/doublet_case_{i}.pdf')

    # Link doublet detection back to spatial.
    # Based on a distance parameter say if a cell might be a doublet or not.
    corrected_doublet_df = doublet_df.copy()

    # Bring doublets back to the original coordinate system.
    corrected_doublet_df['x'] = doublet_df['x'] + min_x
    corrected_doublet_df['y'] = doublet_df['y'] + min_y

    cell_dobulet_df = pd.DataFrame({
        'x': [poly.centroid.x for poly in sdata['cell_boundaries']['geometry']],
        'y': [poly.centroid.y for poly in sdata['cell_boundaries']['geometry']],
    })
    (
        cell_dobulet_df['doublet'],
        cell_dobulet_df['wdoublet'],
        cell_dobulet_df['doublet_distance'],
    ) = flag_cells_near_doublets(
        cell_dobulet_df[['x', 'y']].to_numpy(),
        corrected_doublet_df[['x', 'y']].to_numpy(),
        distance_thresh,
        threads,
    )

    # Plot doublet density
    helperfuncs.plot_scatter_density_df(
        cell_dobulet_df,
        figure_path,
        'doublet',
        'doublet',
        'wdoublet',
        ['lightblue', 'black'],
        'Cells close to doublet events'
    )

    # Write into sdata
    sdata['table'].obs['doublet'] = np.array(cell_dobulet_df['doublet'])
    sdata['table'].obs['wdoublet'] = np.array(cell_dobulet_df['wdoublet'])
    sdata['table'].obs['doublet_distance'] = np.array(cell_dobulet_df['doublet_distance'])

    write_transcript_doublets(sdata, corrected_doublet_df, distance_thresh, threads, spoqc_tmp_folder)