import matplotlib.pyplot as plt

from .. import helperfuncs
from .. import subworkflows
from .. import metrics
from .. import general

def quick_viz_images(figure_path, image_names, sdata):

    for i, image in enumerate(image_names):
        sdata.pl.render_images(image).pl.show(title=image, dpi=300)
        plt.gca().invert_yaxis()
    plt.savefig(f'{figure_path}/all_images.png')
    plt.savefig(f'{figure_path}/all_images.pdf')
    plt.close()

    for i in image_names:
        sdata.pl.render_images(i).pl.show(title=i)
        plt.gca().invert_yaxis()
        plt.savefig(f'{figure_path}/{i}.png')
        plt.savefig(f'{figure_path}/{i}.pdf')
        plt.close()

def run_qc_sc(sdata, figure_path, CONST, obs_columns):

    helperfuncs.plot_original_image_cell_circles(sdata, figure_path, '1')

    quick_viz_images(figure_path, list(sdata.images), sdata)
    metrics.segmentation.sc_metrics.calc_sc_metrics(sdata, figure_path, CONST.ANNOTATION_FILE, CONST.ANNOTATION_KEY)
    general.normalizations.cell_area_normalization(sdata)
    general.valid_geometries.check_for_valid_geometries(sdata, figure_path)

    print("[NOTE] Write results")
    obs_columns = helperfuncs.sdata_obs_to_parquet(sdata, figure_path, CONST.TMP_PATH, 'hqcr', obs_columns)
    print("[finish]")

    return obs_columns