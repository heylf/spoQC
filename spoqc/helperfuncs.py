import numpy as np
import random
import colorsys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import gc
import dask.dataframe as dd
import dask
import time
import pyarrow as pa
import pyarrow.parquet as pq
import shutil
import scanpy as sc
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from typing import NamedTuple, Dict, List, Union, Tuple, Any, Optional, Sequence
from anndata import AnnData
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from matplotlib.colors import to_hex
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

class ImageDimStruct(NamedTuple):
    bb_xmin: int
    bb_ymin: int
    bb_xmax: int
    bb_ymax: int

class AnnotationStruct(NamedTuple):
    ncelltypes: int
    celltypes: int

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        self._start_time = time.time()
        #print("Timer started...")

    def stop(self):
        """Stop the timer and return elapsed time in seconds"""
        if self._start_time is None:
            raise RuntimeError("Timer has not been started. Use .start() first.")
        
        elapsed_time = time.time() - self._start_time
        self._start_time = None  # reset
        print(f"Elapsed time: {elapsed_time:.2f} seconds")


_CMAP_DENSITY = mcolors.LinearSegmentedColormap.from_list(
    "white_blue_yellow", ["white", "blue", "yellow"]
)

# The list remove_from_moving are files which should not be sorted.
# The parameter prefix_or_suffix sets if you want to sort by prefix or suffix.
def sort_files(data_path, prefix_or_suffix, remove_from_moving):
    s = 0
    if ( prefix_or_suffix == 'suffix' ):
        s = 1
    files = os.listdir(data_path)

    if ( 'hist' not in files ):
        files = [x for x in files if x not in remove_from_moving]
        files_prefixes = list(set([x.split("_")[s] for x in files]))
        for prefix in files_prefixes:
            if ( not os.path.exists(f'{data_path}/{prefix}') ):
                os.makedirs(f'{data_path}/{prefix}')
        for f in files:
            shutil.move(f"{data_path}/{f}", f"{data_path}/{f.split('_')[s]}/{f}")


def create_fraction_df(adata: AnnData, group: str, category: str) -> Dict[str, Union[List[str], List, np.ndarray]]:
    """
    Create a dictionary representing fractions of categories within groups from AnnData object.

    Parameters:
    adata (AnnData): Annotated data matrix containing observation data.
    group (str): The column name in `adata.obs` to group by.
    category (str): The column name in `adata.obs` representing categories.

    Returns:
    Dict[str, Union[List[str], List, np.ndarray]]: A dictionary containing:
        - 'label': List of category labels repeated for each group.
        - 'x': List of group names repeated for each category.
        - 'fractions': Flattened numpy array of category fractions for each group.
    """

    series = adata.obs.groupby(group)

    sorted_categories = sorted(set(adata.obs[category].tolist()))

    # List of shape i = len(category), j = len(group), 
    fractions = np.zeros( (len(sorted_categories), len(series)) )
    
    g = 0
    groups = []
    for j, item in series:    
        
        groups.append(j)

        labels_df = series.get_group(j).groupby(category).size()
        labels_total_counts = labels_df.tolist()
        labels_names = labels_df.index.tolist()
        
        for i in range (0, len(labels_names)):
            fraction_of_labels = labels_total_counts[i] / sum(labels_total_counts)
            fractions[sorted_categories.index(labels_names[i])][g] = fraction_of_labels
            
        g += 1
        
    d = {'label': [x for x in sorted_categories for _ in range(0, len(series))],
         'x': groups * len(sorted_categories),
         'fractions': np.concatenate(fractions, axis = None)}

    return(d)    


def image_crop(sdata: Any, bb_xmin: float, bb_ymin: float, 
               bb_xmax: float, bb_ymax: float, coordsystem: str) -> Tuple[Any, float, float]:
    """
    Crop a spatial dataset to a specified bounding box within a given coordinate system.

    Parameters:
    sdata (SpatialData): The spatial dataset to crop.
    bb_xmin (float): Minimum x-coordinate of the bounding box.
    bb_ymin (float): Minimum y-coordinate of the bounding box.
    bb_xmax (float): Maximum x-coordinate of the bounding box.
    bb_ymax (float): Maximum y-coordinate of the bounding box.
    coordsystem (str): The coordinate system used for cropping.

    Returns:
    Tuple[SpatialData, float, float]: A tuple containing:
        - The cropped spatial dataset.
        - The minimum x-coordinate of the bounding box.
        - The minimum y-coordinate of the bounding box.
    """

    sdata_filtered_cs = sdata.filter_by_coordinate_system(coordsystem)

    cropped_sdata = None
    try:
        cropped_sdata = sdata_filtered_cs.query.bounding_box(
            axes=["x", "y"],
            min_coordinate=[bb_xmin, bb_ymin],
            max_coordinate=[bb_xmax, bb_ymax],
            target_coordinate_system=coordsystem,
        )
    except: 
        # This erorr sometimes happen - ValueError: Number of partitions do not match (1 != 8)
        print(f"[Error] Cropping failed with {bb_xmin}, {bb_ymin}, {bb_xmax}, {bb_ymax}. \
              Please check the coordinates and try again.")
        return None, None, None

    if ( 'table' in cropped_sdata._shared_keys ):
        # This has to be done because else those levels have different cell_ids captures.
        # I think this happends because the cropping does not capture polygons on the cropping border.
        ids = cropped_sdata['table'].obs.index
        ids = ids.astype(type(sdata['cell_boundaries'].index[0])).tolist()
            
        for id in ids:
            if ( id not in sdata['cell_boundaries'].index ):
                sys.exit(f"[Error] Please check your indexing of sdata['table'].obs.index" + \
                        f" and sdata['cell_boundaries'].index the index {id} is not in the latter index.")

        cropped_sdata['cell_boundaries'] = sdata['cell_boundaries'].loc[ids]
        cropped_sdata['nucleus_boundaries'] = sdata['nucleus_boundaries'].loc[ids]

        return cropped_sdata, bb_xmin, bb_ymin
    else:
        print("[NOTE] No table in sdata so returning None")
        return None, None, None


def plotly_save_as_png(fig, plot_path, w=4, h=3, dpi=300):
    width_px  = w * dpi
    height_px = h * dpi
    fig.update_layout(margin=dict(l=40, r=20, t=30, b=40))
    fig.write_image(plot_path, width=width_px, height=height_px, scale=1)
    

def generate_distinct_colors(num_colors: int) -> List[str]:
    """
    Generate a list of visually distinct colors in hexadecimal format.

    Parameters:
    num_colors (int): The number of distinct colors to generate.

    Returns:
    List[str]: A list of distinct colors represented as hexadecimal strings.
    """
    colors = []
    golden_ratio_conjugate = 0.618033988749895
    hue = random.random()

    for _ in range(num_colors):
        hue += golden_ratio_conjugate
        hue %= 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.6, 0.9)  # Adjust saturation and value as needed
        color = '#%02X%02X%02X' % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        colors.append(color)

    return colors

def generate_distinct_colors_with_jitter(num_colors: int) -> List[str]:
    colors = []
    golden_ratio_conjugate = 0.618033988749895
    hue = random.random()

    for _ in range(num_colors):
        # Add slight jitter to avoid mechanical uniformity
        hue += golden_ratio_conjugate + random.uniform(-0.02, 0.02)
        hue %= 1.0
        saturation = random.uniform(0.65, 0.85)
        value = random.uniform(0.8, 0.95)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02X}{:02X}{:02X}'.format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        colors.append(hex_color)

    return colors

def get_cbar_shrink(df_shrink):
    y_range = np.max(df_shrink['y']) - np.min(df_shrink['y'])
    x_range = np.max(df_shrink['x']) - np.min(df_shrink['x'])
    shrink = y_range / x_range if x_range > 0 else 1.0
    shrink = max(0.05, min(shrink, 1.0))
    return shrink


def fast_kde2d(
    x, y,
    weights=None,
    bins=300,                 # grid resolution (increase for smoother look)
    bw_adjust=0.5,            # <1 sharper, >1 smoother
    xlim=None, ylim=None
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size

    if xlim is None: xlim = (np.min(x), np.max(x))
    if ylim is None: ylim = (np.min(y), np.max(y))

    # 1) Fast binning into a 2D histogram
    h, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[xlim, ylim], weights=weights
    )

    # 2) Pick bandwidth via Scott's rule (d=2): h = n^(-1/(d+4)) * std
    #    Map bandwidth (in data units) to sigma in *pixels* for gaussian_filter
    xstd = np.std(x, ddof=1) if n > 1 else 1.0
    ystd = np.std(y, ddof=1) if n > 1 else 1.0
    h_factor = n ** (-1.0 / 6.0)  # 1/(d+4) with d=2
    hx = bw_adjust * h_factor * xstd
    hy = bw_adjust * h_factor * ystd

    # convert data bandwidth to pixel sigmas
    dx = (xedges[1] - xedges[0])
    dy = (yedges[1] - yedges[0])
    sigma_x = max(hx / dx, 1e-6)
    sigma_y = max(hy / dy, 1e-6)

    # 3) Gaussian blur (separable, super fast)
    z = gaussian_filter(h.T, sigma=(sigma_y, sigma_x), mode="constant")

    # 4) Coordinates at bin centers (for contourf)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])

    return xc, yc, z, (xlim, ylim)




# Create a function to generate density plots for each category of a key
def plot_density_by_category(df: pd.DataFrame, key: str, figure_path: Union[str, None], flip=False) -> None:
    """
    Generate and save density plots for each category in a given column of a DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data to plot, with columns 'x' and 'y'.
    key (str): The column name in `df` containing the categories for grouping.
    figure_path (Union[str, None]): Path to save the generated plot images. If None, plots are not saved.

    Returns:
    None
    """

    categories = df[key].unique()

    # Set up the plot grid
    plt.figure(figsize=(20, 10))

    ax = plt.gca()
    
    for i, category in enumerate(categories):
        plt.subplot(1, len(categories), i + 1)
        
        subset = df[df[key] == category]

        # Create a kernel density estimate (KDE) plot
        xc, yc, z, (xlim, ylim) = fast_kde2d(
            subset['x'].values, subset['y'].values,
            bins=1000,
            bw_adjust=0.5
        )
        xx, yy = np.meshgrid(xc, yc)
        cf = plt.contourf(xx, yy, z, levels=20, cmap=_CMAP_DENSITY, alpha=0.7, zorder=2)
        cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, shrink=get_cbar_shrink(df))
        cbar.set_label('Density' if key is None else f'Density (weighted by {key})')
        
        plt.title(f'Density Plot: {category}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(np.min(df['x']), np.max(df['x']))
        plt.ylim(np.min(df['y']), np.max(df['y']))
        ax.set_aspect('equal', adjustable='box')

        if ( flip ):
            plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'{figure_path}/densityplot_{key}.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{figure_path}/densityplot_{key}.pdf', bbox_inches='tight', dpi=300)
    plt.close()

# Same as kde but scatter plot
def plot_scatter_by_category(df: pd.DataFrame, key: str, figure_path: str, suffix: str,
                             title: Optional[str], palette: Union[str, Dict[str, str]],
                             pointsize=1.0, flip=False) -> None:
    """
    Generate and save a scatter plot grouped by a categorical key.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data to plot, with columns 'x' and 'y'.
    key (str): The column name in `df` containing the categories for coloring the scatter plot.
    figure_path (str): Path to save the generated plot image.
    suffix (str): Suffix to append to the filename of the saved plot.
    pointsize (float): Size of the points in the scatter plot.
    title (Optional[str]): Title of the plot. If None, no title is displayed.
    palette (Union[str, Dict[str, str]]): Color palette for the plot. Can be a string or a dictionary mapping categories to colors.

    Returns:
    None
    """

    # Create a scatter plot with Seaborn
    sns.scatterplot(data=df, x='x', y='y', hue=key, s=pointsize, palette=palette)

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # Add labels and title
    if ( title != None ):
        plt.title(f'{title}')
    plt.xlabel('x')
    plt.ylabel('y')

    if ( flip ):
        plt.gca().invert_yaxis()

    # Move the legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., markerscale=1)

    plt.tight_layout()
    plt.savefig(f'{figure_path}/scatterplot_{key}_{suffix}.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{figure_path}/scatterplot_{key}_{suffix}.pdf', bbox_inches='tight', dpi=300)
    plt.close()


def plot_scatter_density_by_category_df(df: pd.DataFrame, key: str, figure_path: Union[str, None], suffix: str,
                                        palette: Union[str, dict, None],
                                        title: Optional[str], pointsize=1.0, flip=False) -> None:
    """
    Generate and save density plots for each category in a given column of a DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data to plot, with columns 'x' and 'y'.
    key (str): The column name in `df` containing the categories for grouping.
    figure_path (Union[str, None]): Path to save the generated plot images. If None, plots are not saved.

    Returns:
    None
    """

    categories = df[key].unique()

    fig, axes = plt.subplots(1, len(categories), figsize=(5 * len(categories), 5), squeeze=False)
    axes = axes.ravel()

    for i, category in enumerate(categories):
        ax = axes[i]
        subset = df[df[key] == category]

        # Scatter
        sns.scatterplot(
            data=subset, x='x', y='y', s=pointsize,
            palette=palette, legend=False, ax=ax
        )

        # KDE
        xc, yc, z, (xlim, ylim) = fast_kde2d(
            subset['x'].values, subset['y'].values,
            bins=1000, bw_adjust=0.5
        )
        xx, yy = np.meshgrid(xc, yc)
        cf = ax.contourf(xx, yy, z, levels=20, cmap=_CMAP_DENSITY, alpha=0.7, zorder=2)

        # Individual colorbar
        cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, shrink=get_cbar_shrink(df))
        cbar.set_label('Density' if key is None else f'Density (weighted by {key})')

        ax.set_title(category)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(np.min(df['x']), np.max(df['x']))
        ax.set_ylim(np.min(df['y']), np.max(df['y']))
        ax.set_aspect('equal', adjustable='box')

        if flip:
            ax.invert_yaxis()

    plt.tight_layout()
    if figure_path is not None:
        plt.savefig(f'{figure_path}/scatterplot_densityplot_{key}_{suffix}.png',
                    bbox_inches='tight', dpi=300)
        plt.savefig(f'{figure_path}/scatterplot_densityplot_{key}_{suffix}.pdf',
                    bbox_inches='tight', dpi=300)
    plt.close()



# Density plot for one particular key
def plot_density(adata: AnnData, key: str, figure_path: str, flip=False) -> None:
    """
    Generate and save a density plot weighted by a specified key.

    Parameters:
    adata (AnnData): Annotated data object containing spatial coordinates in `obsm['spatial']`.
    key (str): The column name in `df` to use as weights for the kernel density estimate (KDE) plot.
    figure_path (str): Path to save the generated plot image.

    Returns:
    None
    """
    
    df = pd.DataFrame({
        'x': adata.obsm['spatial'][:,0],
        'y': adata.obsm['spatial'][:,1]
    })

    df[key] = list(adata.obs[key])

    # Set up the plot grid
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # Create a weighed kernel density estimate (KDE) plot. 
    # The kenrel is weighted by the expression of the marker.
    xc, yc, z, (xlim, ylim) = fast_kde2d(
        df['x'].values, df['y'].values,
        weights=df[key].values,
        bins=1000,
        bw_adjust=0.5
    )
    xx, yy = np.meshgrid(xc, yc)
    cf = plt.contourf(xx, yy, z, levels=20, cmap=_CMAP_DENSITY, alpha=0.7, zorder=2)
    cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, shrink=get_cbar_shrink(df))
    cbar.set_label('Density' if key is None else f'Density (weighted by {key})')

    plt.title(f'Density Plot: {key}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(np.min(df['x']), np.max(df['x']))
    plt.ylim(np.min(df['y']), np.max(df['y']))
    ax.set_aspect('equal', adjustable='box')

    if ( flip ):
        plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(f'{figure_path}/densityplot_{key}.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{figure_path}/densityplot_{key}.pdf', bbox_inches='tight', dpi=300)
    plt.close()


def plot_scatter(adata: AnnData, figure_path: str, suffix: str, rect: Optional[Any],
                 colorcat: Optional[str], palette: Union[str, dict, None], title: Optional[str],
                 pointsize=1.0, flip=False) -> None:
    """
    Generate and save a scatter plot from spatial data in an AnnData object.

    Parameters:
    adata (AnnData): Annotated data object containing spatial coordinates in `obsm['spatial']`.
    figure_path (str): Path to save the generated plot image.
    suffix (str): Suffix to append to the filename of the saved plot.
    pointsize (float): Size of the points in the scatter plot.
    rect (Optional[Any]): Rectangle to overlay on the plot. If None, no rectangle is drawn.
    colorcat (Optional[str]): Column name in `adata.obs` to use for coloring points. If None, no coloring is applied.
    palette (Union[str, dict, None]): Color palette for the plot. 
                                      Can be a string, a dictionary mapping categories to colors, or None.
    title (Optional[str]): Title of the plot. If None, no title is displayed.

    Returns:
    None
    """

    df = pd.DataFrame({
        'x': adata.obsm['spatial'][:,0],
        'y': adata.obsm['spatial'][:,1]
    })

    # Create a scatter plot with Seaborn
    if ( colorcat != None ):
        df[colorcat] = list(adata.obs[colorcat])
        df[colorcat] = pd.Categorical(df[colorcat], ordered=True)

    sns.scatterplot(data=df, x='x', y='y', s=pointsize, hue=colorcat, palette=palette)

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # Add labels and title
    plt.xlabel('x')
    plt.ylabel('y')

    if ( title ):
        plt.title(title)

    if ( rect != None ):
        plt.gca().add_patch(rect)

    if ( flip ):
        plt.gca().invert_yaxis()

    # Move the legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., markerscale=1)

    plt.tight_layout()
    plt.savefig(f'{figure_path}/scatterplot_{suffix}.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{figure_path}/scatterplot_{suffix}.pdf', bbox_inches='tight', dpi=300)
    plt.close()


def plot_scatter_density(adata: AnnData, figure_path: str, suffix: str, 
                         scattercat: Optional[str], densitycat: Optional[str],
                         palette: Union[str, dict, None], title: Optional[str],
                         pointsize=1.0, flip=False) -> None:
    
    df = pd.DataFrame({
        'x': adata.obsm['spatial'][:,0],
        'y': adata.obsm['spatial'][:,1]
    })

    # Add scatter category
    if ( scattercat != None ):
        df[scattercat] = list(adata.obs[scattercat])

    # Set up the plot grid
    plt.figure(figsize=(13, 10))
    ax = plt.gca()

    # Overlay a scatter plot
    scatter = sns.scatterplot(data=df, x='x', y='y', s=pointsize, hue=scattercat, palette=palette)

    # Move the legend outside the plot (further to the right)
    handles, labels = scatter.get_legend_handles_labels()

    if handles and labels:
        plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1.6, 1),
                   loc='upper left', borderaxespad=0., markerscale=1)

    # Add density category
    try:
        if densitycat is not None:
            df[densitycat] = list(adata.obs[densitycat])  # Ensure this column exists in df
            xc, yc, z, (xlim, ylim) = fast_kde2d(
                df['x'].values, df['y'].values,
                weights=df[densitycat].values,
                bins=1000,
                bw_adjust=0.5
            )
            if (z == 0).all():
                raise ValueError("[WARNING] Failed to plot density with KDE")
            xx, yy = np.meshgrid(xc, yc)
            cf = plt.contourf(xx, yy, z, levels=20, cmap=_CMAP_DENSITY, alpha=0.7, zorder=2)
            cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, shrink=get_cbar_shrink(df))
            cbar.set_label('Density' if densitycat is None else f'Density (weighted by {densitycat})')
        else:
            xc, yc, z, (xlim, ylim) = fast_kde2d(
                df['x'].values, df['y'].values,
                bins=1000,
                bw_adjust=0.5
            )
            if (z == 0).all():
                raise ValueError("[WARNING] Failed to plot density with KDE")
            xx, yy = np.meshgrid(xc, yc)
            cf = plt.contourf(xx, yy, z, levels=20, cmap=_CMAP_DENSITY, alpha=0.7, zorder=2)
            cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, shrink=get_cbar_shrink(df))
            cbar.set_label('Density' if densitycat is None else f'Density (weighted by {densitycat})')
    except Exception as e:
        print(f"Probably because not enough data points for your chosen density category: {e}")
        print("Just plotting scatter plot.")
        handles, labels = scatter.get_legend_handles_labels()
        if handles and labels:
            plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1.6, 1), 
                    loc='upper left', borderaxespad=0., markerscale=1)

    if ( title ):
        plt.title(title)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(np.min(df['x']), np.max(df['x']))
    plt.ylim(np.min(df['y']), np.max(df['y']))
    ax.set_aspect('equal', adjustable='box')

    if ( flip ):
        plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{figure_path}/scatterplot_densityplot_{suffix}.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{figure_path}/scatterplot_densityplot_{suffix}.pdf', bbox_inches='tight', dpi=300)
    plt.close()


def plot_scatter_density_df(df: pd.DataFrame, figure_path: str, suffix: str,
                         scattercat: Optional[str], densitycat: Optional[str],
                         palette: Union[str, dict, None], title: Optional[str],
                         pointsize=1.0, flip=False) -> None:
    
    plt.figure(figsize=(13, 10))
    ax = plt.gca()
    handles, labels = None, None

    scatter = sns.scatterplot(data=df, x='x', y='y', s=pointsize, hue=scattercat, palette=palette)

    # Create a kernel density estimate (KDE) plot
    try:
        xc, yc, z, (xlim, ylim) = fast_kde2d(
            df['x'].values, df['y'].values,
            weights=df[densitycat].values,
            bins=1000,
            bw_adjust=0.5
        )
        if (z == 0).all():
                raise ValueError("[WARNING] Failed to plot density with KDE")
        xx, yy = np.meshgrid(xc, yc)
        cf = plt.contourf(xx, yy, z, levels=20, cmap=_CMAP_DENSITY, alpha=0.7, zorder=2)
        cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, shrink=get_cbar_shrink(df))
        cbar.set_label('Density' if densitycat is None else f'Density (weighted by {densitycat})')
    except Exception as e:
        print(f"Probably because not enough data points for your chosen density category: {e}")
        print("Just plotting scatter plot.")
        handles, labels = scatter.get_legend_handles_labels()
        if handles and labels:
            plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1.6, 1), 
                    loc='upper left', borderaxespad=0., markerscale=1)

    if handles == None and labels == None:
        ax.get_legend().remove()

    if ( title ):
        plt.title(title)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(np.min(df['x']), np.max(df['x']))
    plt.ylim(np.min(df['y']), np.max(df['y']))
    ax.set_aspect('equal', adjustable='box')

    if ( flip ):
        plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{figure_path}/scatterplot_densityplot_{suffix}.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{figure_path}/scatterplot_densityplot_{suffix}.pdf', bbox_inches='tight', dpi=300)
    plt.close()


def plot_original_image_cell_circles(sdata, figure_path, suffix):
    sdata.pl.render_shapes(elements="cell_circles", scale=0.3).pl.show(dpi=300)
    plt.savefig(f'{figure_path}/image_cell_circles_{suffix}.png')
    plt.savefig(f'{figure_path}/image_cell_circles_{suffix}.pdf')
    plt.close()

def min_value_shift(data: Union[np.ndarray, list]) -> np.ndarray:
    """
    Shift data to ensure all values are non-negative while preserving their relative relationships.

    This is useful in cases where negative values cannot be used (e.g., as weights).

    Parameters:
    data (Union[np.ndarray, list]): Input data, either as a NumPy array or a list of numeric values.

    Returns:
    np.ndarray: Shifted data with all values non-negative.
    """

    # Find the minimum value
    min_value = np.min(data)
    # Shift the data if the minimum value is negative
    if min_value < 0:
        shifted_data = data + abs(min_value)
    else:
        shifted_data = data
    return shifted_data


def points_within_radius(df: pd.DataFrame, radius: float, num: bool) -> List[Union[int, List[int]]]:
    """
    Get points within a given radius for each point in a DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing at least two columns, 'x' and 'y', representing coordinates of points.
    radius (float): The radius within which to search for points.
    num (bool): If True, return the number of points within the radius for each point. 
                If False, return the indices of the points within the radius.

    Returns:
    List[Union[int, List[int]]]:
        A list where each element corresponds to a point in `df`:
        - If `num` is True, the element is the count of points within the radius.
        - If `num` is False, the element is a list of indices of points within the radius.
    """
    points_in_radius = []
    
    for i, point in df.iterrows():
        x1, y1 = point['x'], point['y']
        
        # Calculate the distance from this point to all other points
        distances = np.sqrt((df['x'] - x1)**2 + (df['y'] - y1)**2)
        
        # Get the indices of points within the given radius (excluding the point itself)
        close_points = df[distances <= radius].index.tolist()
        close_points.remove(i)  # Remove the point itself from the list
        
        # Append the list of close points to the result.
        if ( num ):
            points_in_radius.append(len(close_points))
        else:
            points_in_radius.append(close_points)
    
    return points_in_radius


def euclidean_distance(point1: Sequence[float], point2: Sequence[float]) -> float:
    """
    Computes the Euclidean distance between two points in n-dimensional space.

    Args:
        point1 (Sequence[float]): The first point, represented as a sequence of coordinates (e.g., list, tuple).
        point2 (Sequence[float]): The second point, represented as a sequence of coordinates (e.g., list, tuple).

    Returns:
        float: The Euclidean distance between the two points.

    The distance is computed as:
        sqrt(sum((coord_1 - coord_2)^2 for each pair of coordinates))
    """
    return np.sqrt(sum((coord_1 - coord_2) ** 2 for coord_1, coord_2 in zip(point1, point2)))



def add_manual_legend(legend_dict, points=None):
    # If you also want an entry for the red points overlay:
    include_points_in_legend = points is not None
    points_label = "Sample points"   # change as you wish

    # Where to place the legend:
    legend_title = None
    legend_loc = "upper right"       # e.g. 'upper left', 'lower center', etc.
    legend_ncol = 1                  # columns in legend
    legend_frame = True              # show box around legend

    ax = plt.gca()

    # Build handles: squares for raster categories
    handles = [Patch(facecolor=color, edgecolor='black', label=label) 
            for label, color in legend_dict.items()]

    # Optionally add a handle for the scatter points
    if include_points_in_legend:
        handles.append(Line2D([0], [0],
                            marker='o', linestyle='None', markersize=6,
                            markerfacecolor='red', markeredgecolor='red',
                            label=points_label))

    # Add the legend
    leg = ax.legend(handles=handles, title=legend_title, loc=legend_loc,
                    ncols=legend_ncol, frameon=legend_frame)

    ax.legend(handles=handles, title=legend_title, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)


def plot_pixels(
        figure_path,
        image,
        imagedim,
        suffix,
        title,
        cmap,
        axis_off,
        baroff,
        *,
        points=None,
        legend_dict=None,
        flip=False
):

    plt.figure(figsize=(12, 6))
    plt.title(title)

    if ( flip ):
        image = np.flipud(image)

    plt.imshow(
        image , # I need to flip the image
        cmap=cmap,
        extent=[imagedim.bb_xmin, imagedim.bb_xmax, imagedim.bb_ymin, imagedim.bb_ymax], # get the bounding box
        aspect='equal'
    )

    # Overlay red dots if points is provided
    if points is not None:
        plt.scatter(points.iloc[:,0], points.iloc[:,1], c='red', s=2)

    if ( not baroff):
        plt.colorbar()
    if ( axis_off ):
        plt.axis('off')

    if ( legend_dict ):
        add_manual_legend(legend_dict, points)

    plt.savefig(f'{figure_path}/imageplot_{suffix}.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{figure_path}/imageplot_{suffix}.pdf', bbox_inches='tight', dpi=300)
    plt.close()


def KL(a, b):
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def apply_general_plotly_layout(fig, showlegend):
    fig.update_layout(
        plot_bgcolor='white',  # Set background of the plot area to white
        paper_bgcolor='white',  # Set the background of the entire figure to white  
        showlegend=showlegend
    )
    
    fig.update_yaxes(
        showline=True,                        # Ensure the axis line is visible
        linecolor='black',                    # Set the axis line color to black
        ticks='outside',                      # Display ticks outside the axis line
        tickwidth=2,                          # Set the width of the ticks
        tickcolor='black'                     # Set the color of the ticks to black
    )

    fig.update_xaxes(
        showline=True,                        # Ensure the axis line is visible
        linecolor='black',                    # Set the axis line color to black
        ticks='outside',                      # Display ticks outside the axis line
        tickwidth=2,                          # Set the width of the ticks
        tickcolor='black'                     # Set the color of the ticks to black
    )


def leiden_silhouette(adata, resolution, rs, n_clusters):
    '''
    returns a silhouette score based on a given resolution and random state
    '''

    sc.tl.leiden(adata, resolution=resolution, key_added='temp_leiden', random_state=rs)

    score = silhouette_score(adata.obsm['X_umap'], adata.obs['temp_leiden'])
    num_clusters = len(list(set(adata.obs['temp_leiden'])))

    adata.obs.drop(columns=['temp_leiden'], inplace=True)

    return (score, num_clusters == n_clusters, num_clusters)

# This is for checking which leiden cluster resoltuion would work the best.
# Pick the one with the highest silhouette score but not the one from the beginning.
def test_resolutions_leiden(rna, figure_path, n_clusters):
    resolutions = np.linspace(0, 2, num = 21)[1:]

    resolutions

    out = []
    for res in tqdm(resolutions):
        for rs in [0,1,2]: #since the random seed will change result - multiple samples per resolution
            score, hit_n_clusters, num_clusters = leiden_silhouette(rna, res, rs, n_clusters)
            out.append([res, score, rs, hit_n_clusters, num_clusters])

    ss = pd.DataFrame(out, columns = ['res','ss', 'rs', 'hit_n_clusters', 'num_clusters'])

    # Set up the plot grid
    plt.figure(figsize=(15, 5))

    ax = sns.lineplot(data = ss, x = 'res', y = 'ss')

    # Set y-axis limit
    plt.ylim(min(ss['ss']) * 0.9, max(ss['ss']) * 1.1)  # Adding a little padding (10%)

    # Adding labels
    plt.xlabel('resolutions')
    plt.ylabel('silhouettescore')

    # Adding vertical grey lines for each step in 'res'
    for res_value in ss['res'].unique():  # Assuming 'res' contains the breakpoints
        plt.axvline(x=res_value, color='grey', linestyle='--', alpha=0.7)  # Adding vertical lines

    # Setting the breaks on the x-axis
    plt.xticks(ss['res'].unique())  # Ensure all 'res' values are shown on the x-axis

    # Saving the figure
    plt.savefig(f'{figure_path}/test_resolutions_leiden_clustering.png')
    plt.savefig(f'{figure_path}/test_resolutions_leiden_clustering.pdf')
    plt.close()

    return ss

def min_max_normalize(array):
    array = np.array(array)
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def get_stuff_from_image_around_coords(pixel_coords, radius, xy_image_feature, imagedim):
    total_stuff = 0
    stuff_list = []
    xy_image_feature = np.flipud(xy_image_feature) # I have to flip, see plot_pixels function

    # These list I need later because the image matrix has not the same index range as the centroid coords.
    x_idx = [i for i in range(int(imagedim.bb_xmin), int(imagedim.bb_xmax))]
    y_idx = [i for i in range(int(imagedim.bb_ymin), int(imagedim.bb_ymax))]

    for (nx, ny) in pixel_coords:
        nx = int(nx)
        ny = int(ny)
        pixel_stuff = 0
        # Define a circular region around (nx, ny)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x_i = nx + dx
                y_i = ny + dy
                if imagedim.bb_xmin <= x_i < imagedim.bb_xmax and imagedim.bb_ymin <= y_i < imagedim.bb_ymax:
                    # here I have to take x as y and y as x, because x in the image as len(y) and vice versa
                    pixel_stuff += xy_image_feature[y_idx.index(y_i), x_idx.index(x_i)]
        total_stuff += pixel_stuff
        stuff_list.append(pixel_stuff)
    return(total_stuff, stuff_list)


def check_centroids(figure_path, image_df, imagedim, dim_x, dim_y, cells_centroid_coords, nuclei_centroid_coords):
    
    plot_pixels(
        figure_path,
        np.array(image_df['intensity']).reshape(dim_x, dim_y),
        imagedim,
        'check_cell_pixel_coordinates',
        'Check cell coordinates to pixel coordinates scale', 
        'gray',
        False,
        False,
        points=cells_centroid_coords
    )

    plot_pixels(
        figure_path,
        np.array(image_df['intensity']).reshape(dim_x, dim_y),
        imagedim,
        'check_nuclei_pixel_coordinates', 
        'Check nuclei coordinates to pixel coordinates', 
        'gray',
        False,
        False,
        points=nuclei_centroid_coords
    )


def thread_split_list(data, t):
    """Splits the data into t sublists for threading or parallel processing."""
    avg = len(data) // t
    remainder = len(data) % t
    result = []
    start = 0
    for i in range(t):
        end = start + avg + (1 if i < remainder else 0)
        result.append(data[start:end])
        start = end
    return result



def dummyplot(figure_path, suffix):
    plt.figure(figsize=(13, 10))
    scatter = sns.scatterplot(x=[1,2], y=[1,2])
    plt.savefig(f'{figure_path}/dummy_plot_{suffix}.png', bbox_inches='tight')
    plt.savefig(f'{figure_path}/dummy_plot_{suffix}.pdf', bbox_inches='tight')
    plt.close()


def values_to_hex_gradient(values, cmap_name='hot', reverse=False):
    """
    Maps a list of numeric values to hexadecimal color strings using a color map.

    Args:
        values (list of float): The list of values to map.
        cmap_name (str): The name of the colormap to use (default: 'hot').
        reverse (bool): If True, reverse the colormap (default: False).

    Returns:
        list of str: A list of hex color strings corresponding to each value.
    """
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
    cmap = matplotlib.colormaps[cmap_name]
    if reverse:
        cmap = cmap.reversed()
    hex_colors = [mcolors.to_hex(cmap(norm(v))) for v in values]
    return hex_colors


def sdata_obs_to_parquet(sdata, figure_path, spoqc_tmp_folder, suffix, obs_columns):
    new_columns = [x for x in sdata['table'].obs.columns if x not in obs_columns]
    write_df = sdata['table'].obs.loc[:,new_columns]
    write_df.index = sdata['table'].obs.index
    write_df.to_parquet(f"{spoqc_tmp_folder}/{figure_path.split('/')[-2]}_output_{suffix}.parquet")
    return(obs_columns + new_columns)

def read_sdata_parquet_tmp_files(sdata, spoqc_tmp_folder, suffix):
    try:
        tmp_files = [f'{spoqc_tmp_folder}/{file}' for file in os.listdir(spoqc_tmp_folder) \
                     if file.endswith(f'{suffix}.parquet')]
        for tmp_file in tmp_files:
            print(f'[NOTE] read in {tmp_file}')
            tmp_data = pd.read_parquet(tmp_file)
            tmp_data.index = [str(x) for x in tmp_data.index]
            sdata['table'].obs.index = [str(x) for x in sdata['table'].obs.index]
            sdata['table'].obs = sdata['table'].obs.join(tmp_data, how='left')
    except Exception as e:
        print(f"[ERROR] Failed to read parquet files from {spoqc_tmp_folder} most likely because" + \
              f"the data was already loaded in: {e}")
        return None

def nparr_to_parquet(np_arr, prefix, spoqc_tmp_folder, suffix):
    outfile = f"{spoqc_tmp_folder}/{prefix}_output_{suffix}.parquet"
    table = pa.Table.from_arrays([pa.array(np_arr)], names=[prefix])
    pq.write_table(table, outfile)

def df_to_parquet(df, prefix, spoqc_tmp_folder, obs_columns, suffix):
    outfile = f"{spoqc_tmp_folder}/{prefix}_output_{suffix}.parquet"
    new_columns = [x for x in df.columns if x not in obs_columns]
    write_df = df.loc[:,new_columns]
    index = np.array(df.index)
    write_df.index = index
    write_df.to_parquet(outfile)
    return(obs_columns + new_columns)


def ddf_to_parquet(
    ddf: "dd.DataFrame",
    prefix: str,
    spoqc_tmp_folder: str,
    obs_columns: Sequence[str],
    suffix: str,
    *,
    partition_on: str = None,
    include_index: bool = True,
    overwrite: bool = True,
    engine: str = "pyarrow",
) -> List[str]:
    """
    Write the non-observation columns of a Dask DataFrame to Parquet.

    Parameters
    ----------
    df : dask.dataframe.DataFrame
        Input Dask DataFrame.
    prefix, spoqc_tmp_folder, suffix : str
        Used to form the output path: {spoqc_tmp_folder}/{prefix}_output_{suffix}.parquet
    obs_columns : Sequence[str]
        Columns to exclude from the Parquet write.
    include_index : bool, default True
        Whether to persist the index into Parquet.
    partition_on: str, default None
        Give partition key to make use of directory based disk partition.
    overwrite : bool, default True
        Overwrite existing output.
    engine : str, default "pyarrow"
        Parquet engine.

    Returns
    -------
    List[str]
        Column order list: obs_columns + new_columns
    """
    path = f"{spoqc_tmp_folder}/{prefix}_output_{suffix}"

    # Compute columns to write (Dask-friendly; no data materialized)
    obs_set = set(obs_columns)
    new_columns = [c for c in ddf.columns if c not in obs_set]

    # Select only needed columns lazily
    write_ddf = ddf[new_columns]

    # Write to Parquet (this triggers computation)
    write_ddf.to_parquet(
        path,
        engine=engine,
        write_index=include_index,
        overwrite=overwrite,
        partition_on=partition_on,
        compute=True,
    )


def read_df_parquet_tmp_files(intensities, spoqc_tmp_folder, suffix):
    read_image_df = pd.DataFrame({
        'pid': range(len(intensities)),
        'intensity': intensities
    })

    try:
        tmp_files = [f'{spoqc_tmp_folder}/{file}' for file in os.listdir(spoqc_tmp_folder) \
                     if file.endswith(f'{suffix}.parquet')]
        for tmp_file in tmp_files:
            print(f'[NOTE] read in {tmp_file}')
            tmp_data = pd.read_parquet(tmp_file)
            tmp_data['pid'] = range(len(intensities))
            read_image_df = pd.merge(read_image_df, tmp_data, on='pid', how='left')
            del tmp_file
            del tmp_data
            gc.collect()
        return read_image_df
    except Exception as e:
        print(f"[ERROR] Failed to read parquet files from {spoqc_tmp_folder} because of {e}")
        return None


def read_df_parquet_tmp_files_daskified(num_values_image, spoqc_tmp_folder, suffix):
    try:
        # Create base DataFrame
        read_image_df = pd.DataFrame({
            'pid': np.arange(num_values_image)
        })

        # Convert to Dask and set 'pid' as index
        read_image_ddf = dd.from_pandas(read_image_df, npartitions=4).set_index('pid')

        # List all matching parquet files
        tmp_files = [
            os.path.join(spoqc_tmp_folder, file)
            for file in os.listdir(spoqc_tmp_folder)
            if file.endswith(f'{suffix}.parquet')
        ]

        for tmp_file in tmp_files:
            print(f'[NOTE] Reading in {tmp_file}')
            tmp_ddf = dd.read_parquet(tmp_file)
            read_image_ddf = read_image_ddf.join(tmp_ddf, how='left')

        return read_image_ddf

    except Exception as e:
        print(f"[ERROR] Failed to read parquet files from {spoqc_tmp_folder} because of {e}")
        return None


def read_df_parquet_tmp_files_scorify(cluster_df, spoqc_tmp_folder, suffix):
    try:
        tmp_files = [f'{spoqc_tmp_folder}/{file}' for file in os.listdir(spoqc_tmp_folder) \
                     if file.endswith(f'{suffix}.parquet')]
        for tmp_file in tmp_files:
            print(f'[NOTE] read in {tmp_file}')
            tmp_data = pd.read_parquet(tmp_file)

            # Ensure tmp_data has the correct number of rows
            if ( len(tmp_data) != len(cluster_df)) :
                raise ValueError(f"Row count mismatch: expected {len(cluster_df)}, got {len(tmp_data)}")

            if ( tmp_data.isna().any().sum() ):
                print("[DEBUG] Number of NaNs in each column:")
                print(tmp_data.isna().sum().compute())

            # Sum across each row and add to 'score'
            cluster_df['score'] += tmp_data.sum(axis=1).values

    except Exception as e:
        print(f"[ERROR] Failed to read parquet files from {spoqc_tmp_folder} because of {e}")
        return None


def plot_histogram_for_array(array, nbins, figure_path, title, suffix):
    sns.histplot(array, bins=nbins)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("frequency")
    plt.savefig(f'{figure_path}/histogram_{suffix}.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{figure_path}/histogram_{suffix}.pdf', bbox_inches='tight', dpi=300)
    plt.close()


def dask_df_add_sequential_index(ddf, index_name="new_index"):

    lengths = ddf.map_partitions(len).compute()
    starts = [0] + list(np.cumsum(lengths[:-1]))

    def assign_index(part, start):
        part[index_name] = np.arange(start, start + len(part))
        return part

    parts = [dask.delayed(assign_index)(part, start) 
             for part, start in zip(ddf.to_delayed(), starts)]

    meta = ddf._meta.copy()
    meta[index_name] = np.int64()
    ddf_with_index = dd.from_delayed(parts, meta=meta)

    return ddf_with_index.set_index(index_name)
