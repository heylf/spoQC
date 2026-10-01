"""The hqtr/hqpr mask directories (mask_raw, mask_smoothed_raw) in core.parquet.PART_ROWS-row parts.

origin/dev wrote them as 10,000-row dask parts (91,296 files per directory at 913 Mpx); they are
now written in PART_ROWS-row parts. Only the layout changes: every reader must see the same rows,
values, dtypes and index. Each reader of these directories is compared on the two layouts, the
dask ones by their verbatim previous code (tests/legacy/) reading the old layout against
the current code reading the new one, and mutants of the new layout (part order, row count,
dtype) must fail the comparison.
"""

import os
import types

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import load_legacy, parquet_rows
from legacy.parquet_writer import ddf_to_parquet  # origin/dev's writer: the old layout
from spoqc import helperfuncs, hqr, subworkflows
from spoqc.additional_analysis import analysis_funcs
from spoqc.core import parquet, raster
from spoqc.hqr import combine_masks_zoom
from spoqc.image_analysis import celltype_analysis, pixel_scoring_refinement

DIM_X, DIM_Y = 40, 60
N = DIM_X * DIM_Y
OLD_PARTS = 16  # more than 10, so part.10 sorts after part.2 only naturally
NEW_PART_ROWS = 700  # stands in for PART_ROWS (4 parts, the last one short)
METRICS = [
    "edge_strength",
    "energy",
    "relevance",
    "entropy",
    "homogenity",
    "uniformity",
]


def mask_columns(prefix, rng):
    """The columns pixel_scoring_dask writes to {prefix}_output_mask_raw and pixel_scoring_refinement
    to {prefix}_output_mask_smoothed_raw, with origin/dev's dtypes; NaN beliefs (stored as null)."""
    beliefs = rng.random(N)
    beliefs[rng.random(N) < 0.02] = np.nan
    raw = {
        "cluster": rng.integers(0, 100, N).astype(np.int32),
        "s_score": rng.gamma(2.0, 5.0, N).astype(np.float32),
        "as_score": rng.standard_normal(N).astype(np.float32),
        "intensity": (
            rng.integers(0, 65536, N).astype(np.uint16)
            if prefix != "hqtr"
            else rng.gamma(2.0, 3.0, N).astype(np.float32)
        ),
        "p_informative_pixel": rng.random(N),
    }
    if prefix == "hqtr":
        raw |= {
            "norm_p_pixel_score": rng.random(N),
            "pixel_score_mask": rng.integers(0, 2, N),
        }
    raw |= {
        f"{prefix}_beliefs": beliefs,
        f"{prefix}_mask": rng.integers(0, 2, N).astype(
            np.int8 if prefix == "hqtr" else np.int64
        ),
    }
    smoothed = {
        f"{prefix}_beliefs": beliefs,
        f"{prefix}_beliefs_smoothed": rng.random(N).astype(np.float32),
        f"{prefix}_mask_smoothed": rng.integers(0, 2, N).astype(np.int8),
    }
    return {"mask_raw": raw, "mask_smoothed_raw": smoothed}


def write_dir(root, prefix, kind, columns, layout):
    if layout == "old":
        ddf_to_parquet(
            dd.from_pandas(pd.DataFrame(columns), npartitions=OLD_PARTS),
            prefix,
            root,
            [],
            kind,
        )
    else:
        parquet.write_parts(
            f"{root}/{prefix}_output_{kind}",
            N,
            parquet.columns_of(columns),
            range(0, N, NEW_PART_ROWS),
            3,
        )


def write_tmp(root, layout, seed=0):
    """A spoQC tmp folder with every file the mask readers open; the hqpr/hqtr mask directories
    in `layout`, the rest (hqcr files, metrics, priors) identical for both layouts."""
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    for prefix in ("hqpr_0", "hqtr"):
        for kind, columns in mask_columns(prefix, rng).items():
            write_dir(root, prefix, kind, columns, layout)
    for suf in ("", "_smoothed"):
        pd.DataFrame(
            {
                f"hqcr_beliefs{suf}": rng.random(N),
                f"hqcr_mask{suf}": rng.integers(0, 2, N).astype(np.uint8),
            }
        ).to_parquet(f"{root}/hqcr_output_mask{suf}_raw.parquet")
    for folder, suffix in (("metrices/hqpr/0", "hqpr_0"), ("metrices/hqtr", "hqtr")):
        os.makedirs(f"{root}/{folder}", exist_ok=True)
        for metric in METRICS:
            pq.write_table(
                pa.table({metric: rng.random(N).astype(np.float32)}),
                f"{root}/{folder}/{metric}_output_{suffix}.parquet",
            )
    for prior in ("qv", "ac"):
        parquet.write_parts(
            f"{root}/hqtr_output_{prior}_prob",
            N,
            parquet.columns_of({f"{prior}_density": rng.random(N)}),
            range(0, N, 1_000),
            2,
        )
    return root


@pytest.fixture(scope="module")
def tmps(tmp_path_factory):
    base = tmp_path_factory.mktemp("layouts")
    return {layout: write_tmp(str(base / layout), layout) for layout in ("old", "new")}


DIRS = [
    f"{p}_output_{k}"
    for p in ("hqpr_0", "hqtr")
    for k in ("mask_raw", "mask_smoothed_raw")
]


def assert_same_rows(old, new):
    """What every generic reader of a part directory sees: dask's frame (values, dtypes, index),
    the unittests' row hash, read_pixel_columns, and the parts' schema."""
    old_frame, new_frame = (
        dd.read_parquet(old).compute(),
        dd.read_parquet(new).compute(),
    )
    pd.testing.assert_frame_equal(old_frame, new_frame, check_exact=True)
    for column in old_frame:
        assert (
            old_frame[column].to_numpy().tobytes()
            == new_frame[column].to_numpy().tobytes()
        ), column
    assert old_frame.index.to_numpy().tobytes() == new_frame.index.to_numpy().tobytes()

    def row_hash(path):  # spoqc/unittests/test_all.py: assert_dask_equal
        return (
            dd.read_parquet(path, engine="pyarrow")
            .map_partitions(lambda df: pd.util.hash_pandas_object(df, index=True).sum())
            .compute()
            .sum()
        )

    assert row_hash(old) == row_hash(new)
    columns = list(old_frame.columns)
    a = raster.read_pixel_columns(old, columns, N, 3)
    b = raster.read_pixel_columns(new, columns, N, 4)
    for column in columns:
        assert a[column].dtype == b[column].dtype == old_frame[column].dtype, column
        assert (
            a[column].tobytes()
            == b[column].tobytes()
            == old_frame[column].to_numpy().tobytes()
        ), column
    assert parquet_rows(old).schema.equals(
        parquet_rows(new).schema, check_metadata=True
    )


@pytest.mark.parametrize("name", DIRS)
def test_every_generic_reader_sees_the_same_rows(tmps, name):
    old, new = f"{tmps['old']}/{name}", f"{tmps['new']}/{name}"
    assert len(os.listdir(old)) == OLD_PARTS and len(os.listdir(new)) == -(
        -N // NEW_PART_ROWS
    )
    assert_same_rows(old, new)


@pytest.mark.parametrize("name", DIRS)
def test_only_the_dask_divisions_change(tmps, name):
    """No spoQC reader uses divisions; they follow the parts (PART_ROWS apart, then the last row)."""
    new = dd.read_parquet(f"{tmps['new']}/{name}", calculate_divisions=True)
    assert new.divisions == (*range(0, N, NEW_PART_ROWS), N - 1)
    old = dd.read_parquet(f"{tmps['old']}/{name}", calculate_divisions=True)
    assert old.known_divisions and old.divisions[0] == 0 and old.divisions[-1] == N - 1


def relayout(src, dst, edit):
    """A copy of the new-layout directory src with one part table edited: edit(i, table) -> table."""
    os.makedirs(dst)
    for name in os.listdir(src):
        i = int(name.split(".")[1])
        pq.write_table(edit(i, pq.read_table(f"{src}/{name}")), f"{dst}/{name}")


def cast(table, column, type_):
    i = table.schema.get_field_index(column)
    return table.set_column(
        i, table.schema.field(i).with_type(type_), table.column(i).cast(type_)
    )


LAYOUT_MUTANTS = {
    "parts swapped": lambda path: [
        os.rename(f"{path}/part.{a}.parquet", f"{path}/part.{b}.parquet")
        for a, b in ((0, "x"), (1, 0), ("x", 1))
    ],
    "one row dropped": lambda path: [
        pq.write_table(
            pq.read_table(f"{path}/part.2.parquet").slice(1), f"{path}/part.2.parquet"
        )
    ],
    "one row duplicated": lambda path: [
        pq.write_table(pa.concat_tables([t, t.slice(0, 1)]), f"{path}/part.1.parquet")
        for t in [pq.read_table(f"{path}/part.1.parquet")]
    ],
    "one part's dtype": lambda path: [
        pq.write_table(
            cast(pq.read_table(f"{path}/part.3.parquet"), "hqtr_mask", pa.int16()),
            f"{path}/part.3.parquet",
        )
    ],
    "every part's dtype": lambda path: [
        pq.write_table(
            cast(pq.read_table(f"{path}/{n}"), "hqtr_mask", pa.int16()), f"{path}/{n}"
        )
        for n in os.listdir(path)
    ],
}


@pytest.mark.parametrize("name", sorted(LAYOUT_MUTANTS))
def test_layout_mutant_is_caught(tmps, tmp_path, name):
    src = f"{tmps['new']}/hqtr_output_mask_raw"
    dst = str(tmp_path / "hqtr_output_mask_raw")
    relayout(src, dst, lambda i, table: table)
    assert_same_rows(
        f"{tmps['old']}/hqtr_output_mask_raw", dst
    )  # the unedited copy passes
    LAYOUT_MUTANTS[name](dst)
    with pytest.raises((AssertionError, ValueError)):
        assert_same_rows(f"{tmps['old']}/hqtr_output_mask_raw", dst)


def test_production_part_size_is_several_row_groups(tmp_path):
    """PART_ROWS-row parts are pq.write_table's 1,048,576-row row groups, which read_pixel_columns
    decodes in parallel; dask reads the same rows."""
    n = 2 * parquet.PART_ROWS + 12_345
    rng = np.random.default_rng(1)
    columns = {
        "hqtr_mask_smoothed": rng.integers(0, 2, n).astype(np.int8),
        "hqtr_beliefs_smoothed": rng.random(n),
    }
    path = str(tmp_path / "hqtr_output_mask_smoothed_raw")
    parquet.write_parts(
        path, n, parquet.columns_of(columns), range(0, n, parquet.PART_ROWS), 3
    )
    groups = [
        pq.ParquetFile(f"{path}/part.{i}.parquet").metadata.num_row_groups
        for i in range(3)
    ]
    assert sorted(os.listdir(path)) == [
        "part.0.parquet",
        "part.1.parquet",
        "part.2.parquet",
    ]
    assert groups == [4, 4, 1]
    got = raster.read_pixel_columns(path, list(columns), n, 4)
    frame = dd.read_parquet(path).compute()
    for name, values in columns.items():
        assert (
            got[name].dtype == values.dtype and got[name].tobytes() == values.tobytes()
        ), name
        assert frame[name].to_numpy().tobytes() == values.tobytes(), name
    assert frame.index.to_numpy().tobytes() == np.arange(n).tobytes()


# ---------------------------------------------------------------------------
# Each dask reader, verbatim before (old layout) vs now (new layout)
# ---------------------------------------------------------------------------
def labels_record(labels):
    """What a figure/cell-mapping call receives: values, dtype and (for a Series) its index."""
    index = labels.index if isinstance(labels, pd.Series) else None
    values = np.asarray(labels)
    return (
        values.dtype,
        values.tobytes(),
        None
        if index is None
        else (index.name, index.dtype, index.to_numpy().tobytes()),
    )


def record_cell_mappings(monkeypatch, stop_at=None):
    calls = []

    def map_values_to_cells(
        sdata, polys, image_type, resolution, labels, res_col, figure_path, mode, **kw
    ):
        calls.append((res_col, mode, str(kw), labels_record(labels)))
        if stop_at and res_col.endswith(stop_at):
            raise StopIteration(res_col)

    monkeypatch.setattr(subworkflows.hqcr, "map_values_to_cells", map_values_to_cells)
    monkeypatch.setattr(
        subworkflows.hqcr, "create_polygon_dataframe", lambda *a, **k: None
    )
    return calls


def test_map_modality_metrics_to_cells_hands_on_the_same_values(tmps, monkeypatch):
    legacy = load_legacy("analysis_funcs", "spoqc.additional_analysis")
    runs = {}
    for layout, module in (("old", legacy), ("new", analysis_funcs)):
        calls = record_cell_mappings(monkeypatch)
        umap_cats = module.map_modality_metrics_to_cells(
            None, None, "img", "scale0", tmps[layout], "raw", DIM_X, DIM_Y, ["0"], "fig"
        )
        runs[layout] = (umap_cats, calls)
    assert len(runs["old"][1]) == len(runs["new"][1]) > 30
    for i, (a, b) in enumerate(zip(runs["old"][1], runs["new"][1])):
        assert a == b, f"call {i}: {a[0]}"
    assert runs["old"][0] == runs["new"][0]


@pytest.mark.parametrize("modality,staining", [("hqpr", "0"), ("hqtr", None)])
def test_celltype_analysis_hands_on_the_same_values(
    tmps, monkeypatch, modality, staining
):
    legacy = load_legacy("celltype_analysis", "spoqc.image_analysis")
    obs = pd.DataFrame({"celltype": ["a", "b"], "wnucleus_free": [0, 1]})
    sdata = {"table": types.SimpleNamespace(obs=obs)}
    monkeypatch.setattr(helperfuncs, "plot_pixels", lambda *a, **k: None)
    monkeypatch.setattr(
        helperfuncs, "read_sdata_parquet_tmp_files", lambda *a, **k: None
    )
    monkeypatch.setattr(
        subworkflows.hqcr,
        "load_cell_df",
        lambda counts, sdata: pd.DataFrame(index=obs.index),
    )
    monkeypatch.setattr(
        subworkflows.hqcr, "cell_artefact_assignment", lambda *a, **k: None
    )
    runs = {}
    for layout, module in (("old", legacy), ("new", celltype_analysis)):
        monkeypatch.setattr(
            module,
            "px",
            types.SimpleNamespace(
                violin=lambda *a, **k: types.SimpleNamespace(
                    update_layout=lambda **k: None
                )
            ),
        )
        monkeypatch.setattr(module, "save_figure", lambda *a, **k: None)
        calls = record_cell_mappings(monkeypatch, stop_at="_class")
        with pytest.raises(StopIteration):
            module.start_image_celltype_analysis(
                sdata,
                "fig",
                tmps[layout],
                modality,
                "img",
                "scale0",
                None,
                DIM_X,
                DIM_Y,
                "celltype",
                False,
                staining=staining,
            )
        runs[layout] = calls
    assert [c[0] for c in runs["new"]] == [
        "as_score",
        "s_score",
        "intensity",
        f"{modality}{'_0' if staining else ''}_class",
    ]
    assert runs["old"] == runs["new"]


def old_from_pandas_starts(n_rows, n_partitions):
    """The previous core.parquet.from_pandas_starts, which the legacy refinement writes with."""
    size, residual = divmod(n_rows, n_partitions)
    sizes = [size + (i < residual) for i in range(n_partitions)]
    starts = np.cumsum([0, *sizes[:-1]]).tolist()
    return [s for s, n in zip(starts, sizes) if n > 0]


@pytest.mark.parametrize("modality,staining", [("hqpr", "0"), ("hqtr", None)])
def test_refinement_reads_and_writes_the_same_rows(
    tmps, tmp_path, monkeypatch, modality, staining
):
    """The refinement's standalone read of mask_raw (beliefs_raw=None) and its mask_smoothed_raw:
    the old code on the old layout against the new code, which writes PART_ROWS-row parts."""
    legacy = load_legacy("pixel_scoring_refinement", "spoqc.image_analysis")
    monkeypatch.setattr(
        parquet, "from_pandas_starts", old_from_pandas_starts, raising=False
    )
    monkeypatch.setattr(parquet, "PART_ROWS", NEW_PART_ROWS)
    inputs = []

    def mrf(beliefs_raw, beta, max_iter, normalize):
        inputs.append(labels_record(beliefs_raw))
        return (np.nan_to_num(beliefs_raw) * 0.5).astype(np.float32), (
            np.nan_to_num(beliefs_raw) > 0.5
        ).astype(np.int8)

    monkeypatch.setattr(
        hqr.markov_random_field_zarr_parallel,
        "first_version_loopy_belief_propagation_parallel",
        mrf,
    )
    monkeypatch.setattr(
        hqr.markov_random_field_zarr_parallel,
        "visualize_markov_calculation",
        lambda *a: None,
    )
    prefix = f"{modality}_{staining}" if staining else modality
    out = {}
    for layout, module in (("old", legacy), ("new", pixel_scoring_refinement)):
        tmp = tmp_path / layout
        tmp.mkdir()
        os.symlink(
            f"{tmps[layout]}/{prefix}_output_mask_raw",
            tmp / f"{prefix}_output_mask_raw",
        )
        module.start_pixel_mask_refinement(
            "fig", str(tmp), modality, DIM_X, DIM_Y, 1.5, 15, staining=staining
        )
        out[layout] = f"{tmp}/{prefix}_output_mask_smoothed_raw"
    assert inputs[0] == inputs[1]
    assert len(os.listdir(out["old"])) == 1  # origin/dev: ceil(N / 10,000) parts
    assert len(os.listdir(out["new"])) == -(-N // NEW_PART_ROWS)
    assert_same_rows(out["old"], out["new"])


def test_combine_masks_zoom_builds_the_same_frames(tmps, monkeypatch):
    """The zoom step's hqpr/hqtr reads, as the frames it builds from them. The step then dies on
    its pre-existing KeyError (test_bounding_boxes_equivalence pins it), on both layouts."""
    legacy = load_legacy("combine_masks_zoom", "spoqc.hqr")
    monkeypatch.setattr(helperfuncs, "plot_pixels", lambda *a, **k: None)
    runs = {}
    for layout, module in (("old", legacy), ("new", combine_masks_zoom)):
        frames = []

        def frame(data):
            frames.append([(k, labels_record(v)) for k, v in data.items()])
            return pd.DataFrame(data)

        monkeypatch.setattr(
            module,
            "pd",
            types.SimpleNamespace(DataFrame=frame, read_parquet=pd.read_parquet),
        )
        with pytest.raises(KeyError, match="hqcr_beliefs_smoothed"):
            module.start_combining_masks(
                None,
                "fig",
                tmps[layout],
                "img",
                "scale0",
                helperfuncs.ImageDimStruct(0, 0, DIM_Y, DIM_X),
                DIM_X,
                DIM_Y,
                "0",
                3,
            )
        runs[layout] = frames
    assert len(runs["new"]) == 2 and runs["old"] == runs["new"]
