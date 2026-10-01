"""Synthetic Xenium-like SpatialData and loaders for the verbatim origin/dev modules in tests/legacy/."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import anndata as ad
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import shapely
import spatialdata as sd
from spatialdata.models import Image2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Identity, Scale

from spoqc import helperfuncs

LEGACY = Path(__file__).parent / "legacy"
N_CELLS = 150
N_TRANSCRIPTS = 6_000
N_PARTS = 12  # part.10 and part.11 sort after part.2 only numerically
IMAGE_SHAPE = (60, 80)  # (y, x) pixels in the global coordinate system
POINT_SCALE = 2.0  # points -> global
# unsorted, with an unused category and names outside the gene panel
FEATURES = [
    "SEC11C",
    "NegControlCodeword_0502",
    "ACTA2",
    "BLANK_0001",
    "NegControlProbe_00042",
    "KRT7",
    "UNUSED",
    "CD4",
]
GENES = ["SEC11C", "ACTA2", "KRT7", "CD4", "EPCAM"]


def load_legacy(name: str, package: str):
    """Import tests/legacy/<name>.py as a sibling module of `package`, so its relative imports resolve."""
    spec = importlib.util.spec_from_file_location(
        f"{package}._legacy_{name}", LEGACY / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_sdata(rng: np.random.Generator) -> sd.SpatialData:
    # points space; x2 gives global pixels inside the 80 x 60 image
    cell_xy = rng.uniform((0, 0), (38, 28), size=(N_CELLS, 2))
    cell_ids = rng.permutation(np.arange(1, 3 * N_CELLS))[:N_CELLS].astype(np.int32)

    features = rng.choice(FEATURES[:-2] + FEATURES[-1:], size=N_TRANSCRIPTS).astype(
        object
    )
    features[rng.random(N_TRANSCRIPTS) < 0.01] = None
    assigned = rng.random(N_TRANSCRIPTS) < 0.7
    owner = rng.integers(0, N_CELLS, N_TRANSCRIPTS)
    cell_id = np.where(assigned, cell_ids[owner], 0).astype(np.int32)
    cell_id[rng.random(N_TRANSCRIPTS) < 0.02] = (
        10_000  # a cell that is not in the table
    )
    xy = np.where(
        assigned[:, None],
        cell_xy[owner] + rng.normal(0, 1.5, (N_TRANSCRIPTS, 2)),
        rng.uniform((0, 0), (39, 29), (N_TRANSCRIPTS, 2)),
    )
    xy = np.clip(xy, 0, (39.9, 29.9)).astype(np.float32)
    frame = pd.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            "z": rng.uniform(0, 20, N_TRANSCRIPTS).astype(np.float32),
            "feature_name": pd.Categorical(features, categories=FEATURES),
            "cell_id": cell_id,
            "transcript_id": np.arange(N_TRANSCRIPTS, dtype=np.uint64) * 7,
            "overlaps_nucleus": rng.integers(0, 2, N_TRANSCRIPTS).astype(np.uint8),
            "qv": rng.uniform(0, 40, N_TRANSCRIPTS).astype(np.float32),
        }
    )
    points = PointsModel.parse(
        dd.from_pandas(frame, npartitions=N_PARTS),
        feature_key="feature_name",
        instance_key="cell_id",
        transformations={
            "global": Scale([POINT_SCALE, POINT_SCALE, 1.0], axes=("x", "y", "z"))
        },
    )

    def polygons(radius):
        shapes = gpd.GeoDataFrame(
            {"cell_id": cell_ids},
            geometry=[
                shapely.Point(x, y).buffer(radius) for x, y in cell_xy * POINT_SCALE
            ],
            index=cell_ids.astype(np.int64),
        )
        return ShapesModel.parse(shapes, transformations={"global": Identity()})

    image = Image2DModel.parse(
        rng.integers(0, 1000, (1, *IMAGE_SHAPE), dtype=np.uint16),
        dims=("c", "y", "x"),
        scale_factors=[2],
        transformations={"global": Identity()},
    )

    counts = sp.csr_matrix(rng.poisson(2.0, (N_CELLS, len(GENES))).astype(np.float32))
    table = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(
            {
                "cell_id": cell_ids,
                "region": pd.Categorical(["nucleus_boundaries"] * N_CELLS),
            },
            index=cell_ids.astype(str),  # like Xenium: obs index '17' for cell_id 17
        ),
        var=pd.DataFrame(index=GENES),
    )
    table.obsm["spatial"] = cell_xy * POINT_SCALE
    # annotating the nuclei lets the --dev_test crop (a bounding-box query) keep the table
    table = TableModel.parse(
        table, region="nucleus_boundaries", region_key="region", instance_key="cell_id"
    )
    return sd.SpatialData(
        points={"transcripts": points},
        shapes={"nucleus_boundaries": polygons(1.0), "cell_boundaries": polygons(2.5)},
        images={"morphology_focus": image},
        tables={"table": table},
    )


@pytest.fixture(scope="session")
def synthetic_zarr(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("spoqc") / "synthetic.zarr"
    _synthetic_sdata(np.random.default_rng(7)).write(path)
    parts = sorted(
        p.name
        for p in (path / "points" / "transcripts" / "points.parquet").glob(
            "part.*.parquet"
        )
    )
    assert len(parts) == N_PARTS, parts
    return path


def cli_sdata(path: Path, crop: tuple | None = None) -> sd.SpatialData:
    """Read the zarr and apply the transcript setup of cli.py (origin/dev db00d98, lines 379-430) verbatim."""
    sdata = sd.read_zarr(f"{path}")
    sdata.points["transcripts"] = PointsModel.parse(
        helperfuncs.deduplicate_dask_index(sdata.points["transcripts"])
    )
    if crop:  # the --dev_test crop, (xmin, ymin, xmax, ymax) in global coordinates
        sdata, _, _ = helperfuncs.image_crop(sdata, *crop, "global")
    sdata["table"].obs.index = [int(i) for i in range(len(sdata["table"].obs.index))]
    mapping = dict(zip(sdata["table"].obs["cell_id"], sdata["table"].obs.index))
    sdata.points["transcripts"]["cell_id"] = (
        sdata.points["transcripts"]["cell_id"]
        .map(mapping, meta=("cell_id", int))
        .fillna(-1)
        .astype(int)
    )
    sdata.points["transcripts"]["feature_name"] = (
        sdata.points["transcripts"]["feature_name"]
        .astype("string")
        .fillna("NaN")
        .astype("category")
    )
    return sdata


@pytest.fixture
def sdata(synthetic_zarr) -> sd.SpatialData:
    return cli_sdata(synthetic_zarr)


def assert_same_array(a, b, what: str) -> None:
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype, f"{what}: dtype {a.dtype} != {b.dtype}"
    assert a.shape == b.shape, f"{what}: shape {a.shape} != {b.shape}"
    if a.dtype == object:
        assert a.tolist() == b.tolist(), what
    else:
        assert a.tobytes() == b.tobytes(), f"{what}: values differ"


def parquet_rows(path):
    """A per-pixel part directory as one pyarrow table, parts in dask's (natural) order; every
    part must have the first part's schema, pandas metadata included."""
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq
    from dask.utils import natural_sort_key

    tables = [pq.read_table(f"{path}/{f}") for f in sorted(os.listdir(path), key=natural_sort_key)]
    assert all(t.schema.equals(tables[0].schema, check_metadata=True) for t in tables), path
    return pa.concat_tables(tables)
