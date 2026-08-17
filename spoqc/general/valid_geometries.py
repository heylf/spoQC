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


    # In[]
        geometries = np.array(sdata[f'{obj_type}_boundaries']['geometry'])

        # This can happen if you have nucleus-free cells
        if ( len(geometries) < sdata['table'].n_obs ):
            sdata['table'].obs[f'valid_{obj_type}_geometry'] = [True] * sdata['table'].n_obs
            sdata['table'].obs[f'wvalid_{obj_type}_geometry'] = [0] * sdata['table'].n_obs
            boundary_indices = np.array([str(x) for x in sdata[f'{obj_type}_boundaries'].index])

            sdata['table'].obs.loc[boundary_indices, f'valid_{obj_type}_geometry'] = \
                [True if x.is_valid else False for x in geometries]

            sdata['table'].obs.loc[boundary_indices, f'wvalid_{obj_type}_geometry'] = \
                [0 if x.is_valid else 1 for x in geometries]
        elif ( len(geometries) > sdata['table'].n_obs ):
            # This can happen if a cell has more than one nucleus.
            sdata['table'].obs[f'valid_{obj_type}_geometry'] = [True] * sdata['table'].n_obs
            sdata['table'].obs[f'wvalid_{obj_type}_geometry'] = [0] * sdata['table'].n_obs

            boundary_indices = np.array([str(x) for x in sdata[f'nucleus_boundaries']['cell_id']])
            is_valid = np.array([x.is_valid for x in geometries])

            # A cell is invalid if at least one of its nuclei is invalid.
            validity_per_cell = {}
            for idx, valid in zip(boundary_indices, is_valid):
                validity_per_cell[idx] = validity_per_cell.get(idx, True) and valid

            unique_indices = np.array(list(validity_per_cell.keys()))
            unique_valid = np.array(list(validity_per_cell.values()))

            sdata['table'].obs.loc[unique_indices, f'valid_{obj_type}_geometry'] = unique_valid
            sdata['table'].obs.loc[unique_indices, f'wvalid_{obj_type}_geometry'] = (~unique_valid).astype(int)
        else:
            sdata['table'].obs[f'valid_{obj_type}_geometry'] = [True if x.is_valid else False for x in geometries]
            sdata['table'].obs[f'wvalid_{obj_type}_geometry'] = [0 if x.is_valid else 1 for x in geometries]

    # In[]
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
                    plt.savefig(f"{figure_path}/invalid_{obj_type}_geomtry_{num_invalid_examples}.pdf",
                                bbox_inches='tight', pad_inches=0, dpi=300)
                    plt.close()
                    num_invalid_examples += 1

                geometries[i] = obj.convex_hull
        sdata[f'{obj_type}_boundaries']['geometry'] = geometries



