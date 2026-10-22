import matplotlib.pyplot as plt

from .. import helperfuncs

def start_exploration(enterprise, flip=True):
    if ( enterprise.args.step in ['all', 'unittest', 'generalqc'] ):
        print('[NOTE] Check images')
        figure_path = f'{enterprise.args.output_dir}/generalqc/'
        
        helperfuncs.plot_original_image_cell_circles(enterprise.cargo.sdata, figure_path, '1')

        for i, image in enumerate(list(enterprise.cargo.sdata.images)):
            ax = enterprise.cargo.sdata.pl.render_images(image).pl.show(title=image, dpi=300, return_ax=True, show=False)
            if flip:
                ax.invert_yaxis()
            plt.savefig(f'{figure_path}/all_images.png')
            plt.savefig(f'{figure_path}/all_images.pdf')
            plt.close()

        for i in list(enterprise.cargo.sdata.images):
            ax = enterprise.cargo.sdata.pl.render_images(i).pl.show(title=i, return_ax=True, show=False)
            if flip:
                ax.invert_yaxis()
            plt.savefig(f'{figure_path}/{i}.png')
            plt.savefig(f'{figure_path}/{i}.pdf')
            plt.close()
