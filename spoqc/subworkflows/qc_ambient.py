
from .. import metrics

def start_qc_ambient(enterprise):

    if ( enterprise.args.step in ['all', 'hqtr', 'unittest', 'ambientqc'] ):
        figure_path = f'{enterprise.args.output_dir}/ambientqc/'

        enterprise.cargo.sdata['table'].X = enterprise.cargo.sdata['table'].layers['normlog']

        print("[Note] Investigate global ambient contamination")
        metrics.transcript_density.global_moran_I.calculate_global_moran_I_values(
            enterprise.cargo.sdata,
            figure_path,
            enterprise.args.tmp_dir,
        )

        enterprise.cargo.sdata['table'].X = enterprise.cargo.sdata['table'].layers['raw']
        print("[finish]")