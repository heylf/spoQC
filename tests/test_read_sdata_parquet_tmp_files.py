"""helperfuncs.read_sdata_parquet_tmp_files vs the verbatim original (origin/dev db00d98).

The original compared the parquet schema's names, which include the stored pandas index
('index'), with obs.columns. So when a step's columns were already in obs (every -s all run)
it did not skip the file, the join raised "columns overlap", and the warning swallowed it.
The fixed reader reads only the columns obs does not have (skipping files with none), and\nraises on any failure.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from spoqc import helperfuncs

N_CELLS = 50


def original_read_sdata_parquet_tmp_files(sdata, spoqc_tmp_folder, suffix):
    import os

    import pyarrow.parquet as pq

    try:
        tmp_files = [
            f"{spoqc_tmp_folder}/{file}"
            for file in os.listdir(spoqc_tmp_folder)
            if file.endswith(f"{suffix}.parquet")
        ]
        sdata["table"].obs.index = [str(x) for x in sdata["table"].obs.index]
        for tmp_file in tmp_files:
            # Check the on-disk schema (cheap, no data read) so files already joined in a
            # previous call are skipped instead of being re-read from disk every time.
            columns = pq.ParquetFile(tmp_file).schema.names
            if all(col in sdata["table"].obs.columns for col in columns):
                print(f"[NOTE] skip {tmp_file}, already loaded in")
                continue
            print(f"[NOTE] read in {tmp_file}")
            tmp_data = pd.read_parquet(tmp_file)
            tmp_data.index = [str(x) for x in tmp_data.index]
            sdata["table"].obs = sdata["table"].obs.join(tmp_data, how="left")
    except Exception as e:
        print(f"[WARN] Failed to read parquet files from {spoqc_tmp_folder}: {e}")
        return None


def base_table():
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {"cell_area": rng.random(N_CELLS), "cell_id": [f"c{i}" for i in range(N_CELLS)]}
    )
    obs.index = pd.Index(
        [str(i) for i in range(N_CELLS)], name="index"
    )  # cli.py names it 'index'
    return ad.AnnData(obs=obs)


def step_columns():
    """What three steps add to obs, as sdata_obs_to_parquet writes it."""
    rng = np.random.default_rng(1)
    return {
        "generalqc": pd.DataFrame(
            {
                "pct_counts_mt": rng.random(N_CELLS),
                "valid_cell_geometry": rng.random(N_CELLS) > 0.1,
            }
        ),
        "doubletqc": pd.DataFrame(
            {
                "doublet": rng.integers(0, 2, N_CELLS),
                "doublet_distance": rng.random(N_CELLS),
            }
        ),
        "voidqc": pd.DataFrame(
            {"convexhull_outside_trnascripts": rng.integers(0, 9, N_CELLS)}
        ),
    }


@pytest.fixture
def tmp_folder(tmp_path):
    table = base_table()
    obs_columns = list(table.obs.columns)
    for step, columns in step_columns().items():
        for col in columns:
            table.obs[col] = columns[col].to_numpy()
        obs_columns = helperfuncs.sdata_obs_to_parquet(
            {"table": table}, f"figures/{step}/", str(tmp_path), "hqcr", obs_columns
        )
    # hqcr's own tmp file, written with an unnamed index ('__index_level_0__')
    pd.DataFrame(
        {"hqcr_traffic_light": ["green"] * N_CELLS}, index=table.obs.index.rename(None)
    ).to_parquet(tmp_path / "traffic_light_output_hqcr.parquet")
    (tmp_path / "hqcr_output_mask_raw.parquet").write_bytes(
        b"not read: no 'hqcr.parquet' suffix"
    )
    return tmp_path, table.obs.copy()


def in_process_table(in_memory_obs):
    """-s all: the steps ran in this process, so their columns are already in obs."""
    return ad.AnnData(obs=in_memory_obs.copy())


def test_in_process_run_reads_nothing_like_the_original(tmp_folder, capsys):
    folder, in_memory_obs = tmp_folder
    in_memory_obs["hqcr_traffic_light"] = "green"
    old, new = (
        {"table": in_process_table(in_memory_obs)},
        {"table": in_process_table(in_memory_obs)},
    )
    original_read_sdata_parquet_tmp_files(old, str(folder), "hqcr")
    assert (
        "[WARN] Failed to read parquet files" in capsys.readouterr().out
    )  # the silent failure
    helperfuncs.read_sdata_parquet_tmp_files(new, str(folder), "hqcr")
    out = capsys.readouterr().out
    assert "read in" not in out and out.count("skip") == 4
    pd.testing.assert_frame_equal(new["table"].obs, old["table"].obs)
    pd.testing.assert_frame_equal(
        new["table"].obs, in_memory_obs.set_axis(in_memory_obs.index.rename(None))
    )


def test_step_run_on_its_own_reads_every_file(tmp_folder):
    folder, in_memory_obs = tmp_folder
    old, new = {"table": base_table()}, {"table": base_table()}
    original_read_sdata_parquet_tmp_files(old, str(folder), "hqcr")
    helperfuncs.read_sdata_parquet_tmp_files(new, str(folder), "hqcr")
    pd.testing.assert_frame_equal(new["table"].obs, old["table"].obs)
    expected = in_memory_obs.assign(hqcr_traffic_light="green")
    got = new["table"].obs[expected.columns]
    pd.testing.assert_frame_equal(got, expected.set_axis(expected.index.rename(None)))


def test_step_on_its_own_keeps_recomputed_columns_and_reads_the_rest(tmp_folder):
    """A step run on its own recomputes some of a file's columns (cli's mandatory block sets
    the valid-geometry columns of generalqc's file); only the other columns are read."""
    folder, in_memory_obs = tmp_folder
    table = base_table()
    table.obs["valid_cell_geometry"] = in_memory_obs["valid_cell_geometry"].to_numpy()
    sdata = {"table": table}
    helperfuncs.read_sdata_parquet_tmp_files(sdata, str(folder), "hqcr")
    expected = in_memory_obs.assign(hqcr_traffic_light="green")
    got = sdata["table"].obs[expected.columns]
    pd.testing.assert_frame_equal(got, expected.set_axis(expected.index.rename(None)))


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        helperfuncs.read_sdata_parquet_tmp_files(
            {"table": base_table()}, str(tmp_path / "missing"), "hqcr"
        )


def test_column_dropped_after_the_read_is_read_again(tmp_folder):
    """With an annotation file, write_out_anndata('overview') drops nuclei_idxs from obs and
    load_cell_metrices reads the tmp files again: only that column is read back (origin/dev
    swallowed the overlap and read nothing)."""
    folder, in_memory_obs = tmp_folder
    in_memory_obs["hqcr_traffic_light"] = "green"
    sdata = {"table": in_process_table(in_memory_obs.drop(columns=["doublet_distance"]))}
    helperfuncs.read_sdata_parquet_tmp_files(sdata, str(folder), "hqcr")
    expected = in_memory_obs.set_axis(in_memory_obs.index.rename(None))
    pd.testing.assert_frame_equal(sdata["table"].obs[expected.columns], expected)


def test_column_in_two_files_raises(tmp_folder):
    folder, _ = tmp_folder
    pd.DataFrame({"doublet_distance": np.zeros(N_CELLS)}, index=base_table().obs.index).to_parquet(
        folder / "extraqc_output_hqcr.parquet"
    )
    with pytest.raises(ValueError, match=r"more than one tmp file.*doublet_distance.*extraqc_output_hqcr"):
        helperfuncs.read_sdata_parquet_tmp_files({"table": base_table()}, str(folder), "hqcr")


def test_file_without_pandas_metadata_raises(tmp_folder):
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder, _ = tmp_folder
    pq.write_table(pa.table({"x": np.zeros(N_CELLS)}), folder / "plain_output_hqcr.parquet")
    with pytest.raises(ValueError, match="plain_output_hqcr.parquet has no pandas metadata"):
        helperfuncs.read_sdata_parquet_tmp_files({"table": base_table()}, str(folder), "hqcr")
