
from .. import helperfuncs
from .. import subworkflows
from .. import metrics

def run_qc_cell(sdata, figure_path, CONST, obs_columns):

    timer = helperfuncs.Timer()

    print("[NOTE] Convexity QC")
    timer.start()
    metrics.segmentation.convexity.calc_convexity(sdata, figure_path)
    timer.stop()

    print("[NOTE] Multinuclei QC")
    # TODO check this again in contradicts the convexity analysis
    timer.start()
    metrics.segmentation.nuclei_count.count_nuclei(sdata, figure_path)
    timer.stop()

    print("[NOTE] Border cell inspection")
    #TODO combine island score and border score to really just pick border cells and not smalle cell islands.
    timer.start()
    metrics.segmentation.border_score.define_border_cells(sdata, figure_path, 1.0, 50, 10, CONST.THREADS)
    timer.stop()

    print("[NOTE] Cell island inspection")
    timer.start()
    metrics.segmentation.island_score.calc_island_score(sdata, figure_path, 15, 10)
    timer.stop()

    print("[NOTE] Low transcript quality inspection")
    timer.start()
    transcript_df = sdata['transcripts'].compute()
    # at 10x Genomics they use a threshold of qv < 20 (see 10x Baysor tutorial)
    subworkflows.qc_transcript.get_low_qc_transcript_count(transcript_df, sdata, 20, figure_path)
    timer.stop()

    print("[NOTE] Write results")
    obs_columns = helperfuncs.sdata_obs_to_parquet(sdata, figure_path, CONST.TMP_PATH, 'hqcr', obs_columns)
    print("[finish]")