import numpy as np
import pandas as pd

from ... import helperfuncs
from ... import core

from dask_ml.preprocessing import MinMaxScaler

def _calc_prior(
        spoqc_tmp_folder,
        figure_path,
        imagedim,
        dim_x,
        dim_y,
        chunk_size,
        *,
        thresh=0.4,
        std=1,
        tail="left",
        col="qv_density"
    ):


    if std <= 0:
        raise ValueError("std must be > 0")

    inv_std = 1.0 / std
    norm_const = inv_std / np.sqrt(2.0 * np.pi)

    # Load data
    tmp_files = f'{spoqc_tmp_folder}/qv_density_output_hqtr'
    image_ddf = helperfuncs.read_data_as_ddf(tmp_files, chunk_size)

    def _part(part: pd.DataFrame) -> pd.Series:
        x = part[col].to_numpy()
        # Gaussian PDF centered at `thresh`
        z = (x - thresh) * inv_std
        pdf = norm_const * np.exp(-0.5 * z * z)

        # Tail overwrite to norm_const (then we'll invert below).
        # You have to use norm_const because it is ultimately where the peak height of the Guassian is.
        # Do not use np.max(pdf) here because we deal with Dask partitions and each partition has its own distribution.
        # Thus the constant here is given by the Gaussian shape.
        if tail == "left":
            pdf = np.where(x < thresh, norm_const, pdf)
        elif tail == "right":
            pdf = np.where(x > thresh, norm_const, pdf)
        # else: no tail tweak

        # Because the tailing sets values to norm_const we have substract norm_const
        # to create 0 which is the extreme case of the worst probability.
        # Keep in mind that you deal with pdfs here not probabilities.
        out = norm_const - pdf

        # We have no densities which we still have to turn into probabilities!
        return pd.Series(out, index=part.index, name=col)

    p_series = image_ddf.map_partitions(_part, meta=(col, "f8"))
    image_ddf = image_ddf.assign(**{col: p_series})

    # Min-Max normalize using dask-ml
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(image_ddf[[col]])
    scaled_series = scaled_df.iloc[:, 0]

    image_ddf = image_ddf.assign(**{f"prob_{col}": scaled_series})

    helperfuncs.plot_histogram_for_array(
        image_ddf[f"prob_{col}"].compute().to_numpy(),
        100,
        figure_path,
        f"{col}: t={np.round(thresh, 3)} with {1} x {np.round(std, 3)} std",
        f"prob_qv_density",
        t=thresh,
        std=std,
        nstds=1,
    )

    helperfuncs.plot_pixels(
        figure_path,
        image_ddf[f"prob_{col}"].compute().to_numpy().reshape(dim_x, dim_y),
        imagedim,
        'prob_qv_density',
        'Probability of QV Density Pixel', 
        'hot',
        False,
        False
    )

    return image_ddf


def init_prior(enterprise):

    # These have to be defined.
    name = "qv_density"
    modality = "hqtr"
    tmp_path = None
    needs_metrics = ["qv_density"]

    # These are given by your prior calc function.
    args = [enterprise.args.tmp_dir, f"{enterprise.args.output_dir}/hqtr/hqtr_qv/", 
            enterprise.cargo.imagedim, enterprise.cargo.dim_x, enterprise.cargo.dim_y, 
            enterprise.args.chunk_size]
    kwargs = None

    prior = core.prior.Prior(
        _calc_prior, 
        name,
        modality,
        needs_metrics = needs_metrics,
        tmp_path = tmp_path,
        args = args,
        kwargs = kwargs,
    )    
    
    return prior
