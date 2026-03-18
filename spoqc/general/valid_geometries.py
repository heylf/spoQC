import numpy as np
import matplotlib.pyplot as plt

from .. import helperfuncs

# I tested geopandas.GeoSeries.make_valid instead of of the convex hull, but it still looked weird.
def correct_for_valid_geometries(sdata):
    for obj_type in ['cell', 'nucleus']:
        # Because of invalid geometris I have to take for those the convex hull.
        geometries = np.array(sdata[f'{obj_type}_boundaries']['geometry'])
        for i, obj in enumerate(geometries):
            if( not obj.is_valid ):
                geometries[i] = obj.convex_hull
        sdata[f'{obj_type}_boundaries']['geometry'] = geometries


def check_for_valid_geometries(sdata, figure_path):
    for obj_type in ['cell', 'nucleus']:
        geometries = np.array(sdata[f'{obj_type}_boundaries']['geometry'])
        print(obj_type)

        # This can happen if you have nucleus-free cells
        if ( len(geometries) < sdata['table'].n_obs ):
            sdata['table'].obs[f'valid_{obj_type}_geometry'] = [True] * sdata['table'].n_obs
            sdata['table'].obs[f'wvalid_{obj_type}_geometry'] = [0] * sdata['table'].n_obs
            boundary_indices = np.array([str(x) for x in sdata[f'{obj_type}_boundaries'].index])

            sdata.table.obs.loc[boundary_indices, f'valid_{obj_type}_geometry'] = \
                [True if x.is_valid else False for x in geometries]

            sdata.table.obs.loc[boundary_indices, f'wvalid_{obj_type}_geometry'] = \
                [0 if x.is_valid else 1 for x in geometries]
        else:
            sdata['table'].obs[f'valid_{obj_type}_geometry'] = [True if x.is_valid else False for x in geometries]
            sdata['table'].obs[f'wvalid_{obj_type}_geometry'] = [0 if x.is_valid else 1 for x in geometries]

        helperfuncs.plot_scatter_density(
                sdata['table'],
                figure_path,
                f'invalid_{obj_type}_geometry',
                f'valid_{obj_type}_geometry',
                f'wvalid_{obj_type}_geometry',
                ['red', 'lightblue'],
                f'Density of invalid {obj_type} geometries'
        )

        # Because of invalid geometris I have to take for those the convex hull.
        num_invalid_examples = 0
        for i, obj in enumerate(geometries):
            if( not obj.is_valid ):

                if ( num_invalid_examples < 10 ):
                    x, y = obj.exterior.xy

                    fig, ax = plt.subplots()
                    ax.fill(x, y, alpha=0.7, fc='red', ec='black')  # fill polygon with blue, outline in black
                    ax.set_aspect('equal')
                    ax.axis('off')
                    plt.savefig(f"{figure_path}/invalid_{obj_type}_geomtry_{num_invalid_examples}.png",
                                bbox_inches='tight', pad_inches=0, dpi=300)
                    plt.close()
                    num_invalid_examples += 1

                geometries[i] = obj.convex_hull
        sdata[f'{obj_type}_boundaries']['geometry'] = geometries



