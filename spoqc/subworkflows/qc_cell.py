
from .. import helperfuncs
from .. import subworkflows
from .. import metrics

def run_qc_cell(enterprise):
    if enterprise.args.step in ['all', 'unittest', 'cellqc']:
        figure_path = f'{enterprise.args.output_dir}/cellqc/'

        timer = helperfuncs.Timer()

        print("[NOTE] Convexity QC")
        timer.start()
        metrics.segmentation.convexity.calc_convexity(enterprise.cargo.sdata, figure_path)
        timer.stop()

        print("[NOTE] Multinuclei QC")
        timer.start()
        metrics.segmentation.nuclei_count.count_nuclei(enterprise.cargo.sdata, figure_path)
        timer.stop()

        print("[NOTE] Border cell inspection")
        timer.start()
        metrics.segmentation.border_score.define_border_cells(
            enterprise.cargo.sdata,
            figure_path,
            enterprise.args.nthreads,
        )
        timer.stop()

        print("[NOTE] Cell island inspection")
        timer.start()
        metrics.segmentation.island_score.calc_island_score(enterprise.cargo.sdata, figure_path)
        timer.stop()

        print("[NOTE] Low transcript quality inspection")
        timer.start()
        transcript_df = enterprise.cargo.sdata['transcripts'].compute()
        subworkflows.qc_transcript.get_low_qc_transcript_count(transcript_df, enterprise.cargo.sdata, figure_path)
        timer.stop()

        print("[NOTE] Write results")
        helperfuncs.sdata_obs_to_parquet(
            enterprise,
            figure_path,
            'hqcr'
        )
        print("[finish]")