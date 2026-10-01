import numpy as np
import pandas as pd

from ... import helperfuncs

from dask_ml.preprocessing import MinMaxScaler

def calc_prob_pixel_stuff_v2(image_ddf, figure_path, thresh, std, tail, col):

    if std <= 0:
        raise ValueError("std must be > 0")

    inv_std = 1.0 / std
    norm_const = inv_std / np.sqrt(2.0 * np.pi)

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
        return pd.Series(out, index=part.index, name=f"d_{col}")

    p_series = image_ddf.map_partitions(_part, meta=(f"d_{col}", "f8"))
    image_ddf = image_ddf.assign(**{f"d_{col}": p_series})

    # Min-Max normalize using dask-ml
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(image_ddf[[f'd_{col}']])
    scaled_series = scaled_df.iloc[:, 0]

    image_ddf = image_ddf.assign(
        **{f'norm_p_{col}': scaled_series}
    )

    helperfuncs.plot_histogram_for_array(
        image_ddf[col].compute().to_numpy(),
        100,
        figure_path,
        f"{col}: t={np.round(thresh, 3)} with {1} x {np.round(std, 3)} std",
        f"{col}_prior",
        t=thresh,
        std=std,
        nstds=1,
    )

    return image_ddf