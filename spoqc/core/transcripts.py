"""The one transcript loader of a spoQC run.

The columns of the transcripts element that cli.py sets up (deduplicated index,
cell_id mapped to the obs index, feature_name recast to sorted categories, and the
--dev_test crop) are computed once, kept until `release()`, and handed to every
step that needs them. cli.py's dask setup is the only implementation of those
transforms; this module only materialises its result, as polars.
`feature_name` becomes a `pl.Enum` over the same categories, in the same order.
Caches are keyed weakly by the SpatialData object, so they die with it.
"""

from __future__ import annotations

import weakref

import numpy as np
import polars as pl
from spatialdata.models import get_axes_names
from spatialdata.transformations import (
    Identity,
    MapAxis,
    Scale,
    Translation,
    get_transformation,
)
from xarray import DataArray

_frames: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_indexes: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_global_coordinates: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

# Elementwise transformations: numpy gives the same bits as spatialdata's dask path.
# Affine (and Sequence, which composes to one) goes through a matrix product whose
# result differs between numpy and chunked dask BLAS (measured 1e-14), so it is refused.
_EXACT_TRANSFORMATIONS = (Identity, Scale, Translation, MapAxis)


def _read(sdata, columns: list[str]) -> pl.DataFrame:
    computed = sdata.points["transcripts"][columns].compute()
    if sdata not in _indexes:
        _indexes[sdata] = computed.index.to_numpy()
    frame = pl.from_pandas(computed)
    if "feature_name" in columns:
        categories = list(computed["feature_name"].cat.categories)
        frame = frame.with_columns(pl.col("feature_name").cast(pl.Enum(categories)))
    return frame.rechunk()


def load_transcripts(sdata, columns: list[str]) -> pl.DataFrame:
    """Return `columns` of the transcripts, in index order; each column is computed once per run."""
    frame = _frames.get(sdata)
    missing = [c for c in columns if frame is None or c not in frame.columns]
    if missing:
        new = _read(sdata, missing)
        frame = new if frame is None else frame.hstack(new)
        _frames[sdata] = frame
    return frame.select(columns)


def transcript_index(sdata) -> np.ndarray:
    """The dask element's index labels, row for row with `load_transcripts` (not 0..n-1 after a crop)."""
    if sdata not in _indexes:
        _indexes[sdata] = sdata.points["transcripts"].index.compute().to_numpy()
    return _indexes[sdata]


def lookup_by_code(feature_name: pl.Series, values: dict, default, dtype) -> np.ndarray:
    """An array indexed by the Enum code of `feature_name`: `values[name]`, or `default` for other names."""
    return np.array(
        [values.get(name, default) for name in feature_name.dtype.categories],
        dtype=dtype,
    )


def global_coordinates(sdata) -> pl.DataFrame:
    """The coordinates in the 'global' coordinate system, equal to `sd.get_centroids(transcripts, 'global').compute()`.

    Computed once per run with spatialdata's own transformation code, and kept until `release()`.
    """
    if sdata in _global_coordinates:
        return _global_coordinates[sdata]
    transformation = get_transformation(sdata.points["transcripts"], "global")
    if not isinstance(transformation, _EXACT_TRANSFORMATIONS):
        raise NotImplementedError(
            f"global_coordinates is bit-identical to get_centroids only for "
            f"{[t.__name__ for t in _EXACT_TRANSFORMATIONS]}, not {type(transformation).__name__}"
        )
    axes = list(get_axes_names(sdata.points["transcripts"]))
    coords = load_transcripts(sdata, axes)
    data = DataArray(
        coords.to_numpy(), coords={"points": range(coords.height), "dim": axes}
    )
    # Private API, the same code get_centroids runs. spatialdata is pinned to 0.7.3 in
    # pyproject.toml and requirements.txt; re-check this call when the pin moves.
    moved = transformation._transform_coordinates(data)
    _global_coordinates[sdata] = pl.DataFrame(
        {ax: moved.sel(dim=ax).data for ax in axes}
    )
    return _global_coordinates[sdata]


def release(sdata) -> None:
    """Drop the cached transcripts of this SpatialData (a no-op if no step loaded them)."""
    for cache in (_frames, _indexes, _global_coordinates):
        cache.pop(sdata, None)
