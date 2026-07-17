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

        # Tail overwrite to 1.0 (then we'll invert below)
        if tail == "left":
            pdf = np.where(x < thresh, 1.0, pdf)
        elif tail == "right":
            pdf = np.where(x > thresh, 1.0, pdf)
        # else: no tail tweak

        out = 1.0 - pdf
        return pd.Series(out, index=part.index, name=f"p_{col}")

    p_series = image_ddf.map_partitions(_part, meta=(f"p_{col}", "f8"))
    image_ddf = image_ddf.assign(**{f"p_{col}": p_series})

    # Min-Max normalize using dask-ml
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(image_ddf[[f'p_{col}']])
    scaled_series = scaled_df.iloc[:, 0]

    image_ddf = image_ddf.assign(
        **{f'norm_p_{col}': scaled_series}
    )

    helperfuncs.plot_histogram_for_array(
        image_ddf[col].compute().to_numpy(),
        20,
        figure_path,
        f"{col}: t={np.round(thresh, 3)} with {1} x {np.round(std, 3)} std",
        f"{col}_prior",
        t=thresh
    )

    return image_ddf