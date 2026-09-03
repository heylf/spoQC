import matplotlib.pyplot as plt

from .. import helperfuncs
from .. import subworkflows
from .. import metrics
from .. import general

def quick_viz_images(figure_path, image_names, sdata, flip=True):

    for i, image in enumerate(image_names):
        ax = sdata.pl.render_images(image).pl.show(title=image, dpi=300, return_ax=True, show=False)
        if flip:
            ax.invert_yaxis()
        plt.savefig(f'{figure_path}/all_images.png')
        plt.savefig(f'{figure_path}/all_images.pdf')
        plt.close()

    for i in image_names:
        ax = sdata.pl.render_images(i).pl.show(title=i, return_ax=True, show=False)
        if flip:
            ax.invert_yaxis()
        plt.savefig(f'{figure_path}/{i}.png')
        plt.savefig(f'{figure_path}/{i}.pdf')
        plt.close()

def run_qc_sc(enterprise):
    if ( enterprise.args.step in ['all', 'unittest', 'generalqc'] ):
        print('[NOTE] General QC')
        figure_path = f'{enterprise.args.output_dir}/generalqc/'
        
        helperfuncs.plot_original_image_cell_circles(enterprise.cargo.sdata, figure_path, '1')

        quick_viz_images(figure_path, list(enterprise.cargo.sdata.images), enterprise.cargo.sdata)
        metrics.segmentation.sc_metrics.calc_sc_metrics(
            enterprise.cargo.sdata,
            figure_path,
            enterprise.args.annotation_file,
            enterprise.cargo.celltype_annotation.annotation_key,
        )
        general.normalizations.cell_area_normalization(enterprise.cargo.sdata)
        general.valid_geometries.check_for_valid_geometries(enterprise.cargo.sdata, figure_path)

        print("[NOTE] Write results")
        helperfuncs.sdata_obs_to_parquet(
            enterprise,
            figure_path,
            'hqcr'
        )
        print("[finish]")