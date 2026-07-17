import pandas as pd
import numpy as np

from ... import helperfuncs

from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from dask_ml.preprocessing import MinMaxScaler

def calc_probs_pixel_score(pixel_scores, figure_path, gmm_mod=3, nstds=3, t=None, std=None):
    mix = GaussianMixture(n_components=gmm_mod, tol=1e-8, max_iter=int(1e4))
    mix.fit(pixel_scores.reshape(-1, 1))
    means = mix.means_
    cov = mix.covariances_
    stds = [ np.sqrt(  np.trace(cov[i])/gmm_mod) for i in range(0,gmm_mod) ]
    max_std = stds[np.argmax(means)]

    max_mean = -1
    if ( t ):
        max_mean = t
        max_std = 1.0  # Since mean is hard picked, we will use unit variance.
    else:
        max_mean = np.max(means)

    if ( std ):
        max_std = std

    print(f'Using std {max_std} and mean {max_mean} for pixel prior')

    helperfuncs.plot_histogram_for_array(
        pixel_scores,
        20,
        figure_path,
        f"Pixel scores: t={np.round(max_mean, 3)} with {nstds} x {np.round(max_std, 3)} std",
        "pixel_scores_prior",
        t=max_mean
    )

    # Calculate the probability density at x for each pixel clusters.
    return norm.pdf(pixel_scores, loc=max_mean, scale=nstds*max_std)


def calc_prob_pixel_stuff(image_ddf, thresh, std, tail, col):

    # Calculate PDF values directly
    image_ddf = image_ddf.assign(
        **{f'p_{col}': image_ddf[col].map_partitions(
            lambda s: norm.pdf(s, loc=thresh, scale=std), meta=(f'p_{col}', 'f8'))}
    )

    # Apply tail logic using map_partitions (1.0 overwrite rule)
    if tail == 'left':
        image_ddf = image_ddf.assign(
            **{f'p_{col}': image_ddf[[col, f'p_{col}']].map_partitions(
                lambda df: df.apply(lambda row: 1.0 if row[col] < thresh else row[f'p_{col}'], axis=1),
                meta=(f'p_{col}', 'f8'))}
        )
    elif tail == 'right':
        image_ddf = image_ddf.assign(
            **{f'p_{col}': image_ddf[[col, f'p_{col}']].map_partitions(
                lambda df: df.apply(lambda row: 1.0 if row[col] > thresh else row[f'p_{col}'], axis=1),
                meta=(f'p_{col}', 'f8'))}
        )

    # Invert the probabilities
    image_ddf = image_ddf.assign(
        **{f'p_{col}': 1.0 - image_ddf[f'p_{col}']}
    )

    # Min-Max normalize using dask-ml
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(image_ddf[[f'p_{col}']])
    scaled_series = scaled_df.iloc[:, 0]

    image_ddf = image_ddf.assign(
        **{f'norm_p_{col}': scaled_series}
    )

    return image_ddf


def calc_prob_pixel_stuff_v2(image_ddf, thresh, std, tail, col):

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

    return image_ddf