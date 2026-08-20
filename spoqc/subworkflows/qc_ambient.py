
from .. import metrics

def start_qc_ambient(sdata, figure_path, spoqc_tmp_folder):
    sdata['table'].X = sdata['table'].layers['normlog']

    print("[Note] Investigate global ambient contamination")
    global_ambient = metrics.transcript_density.global_moran_I.calculate_global_moran_I_values(
        sdata,
        figure_path,
        spoqc_tmp_folder
    )

    print("[finish]")
    return global_ambient