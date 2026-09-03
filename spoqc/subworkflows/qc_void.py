
from .. import helperfuncs
from .. import metrics

def run_qc_void(enterprise):
    if enterprise.args.step in ['all', 'unittest', 'voidqc']:
        figure_path = f'{enterprise.args.output_dir}/voidqc/'

        print("[NOTE] Void QC")
        # You can also provide contaminants with contaminant_list=['CD3D', 'CD14', 'CD68']:
        metrics.segmentation.void.calc_void(
            enterprise.cargo.sdata,
            figure_path,
            enterprise.args.tmp_dir,
        )

        print("[NOTE] Write results")
        helperfuncs.sdata_obs_to_parquet(
            enterprise,
            figure_path,
            'hqcr'
        )
        print("[finish]")

        # helperfuncs.plot_scatter_density(
        #     sdata['table'], figure_path, None, 
        #     1, None, 'convexhull_all_transcripts', None, 'Density of convexhull'
        # )