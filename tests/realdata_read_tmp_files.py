"""Real-data check (not collected by pytest): read_sdata_parquet_tmp_files on a real run's tmp files.

Usage:
    python tests/realdata_read_tmp_files.py <spatialdata.zarr> <tmp_folder>

<tmp_folder> holds a run's *_hqcr.parquet files. Checks, against the verbatim original:
- a step run on its own (obs without the steps' columns) reads every file, as the original did,
  also when cli's mandatory block has already set generalqc's valid-geometry columns;
- in -s all the steps' columns are already in obs: the fixed reader skips every file and
  leaves obs as it was, which is what the original's swallowed failure also left.
"""

import sys

import anndata as ad
import pandas as pd

from spoqc import helperfuncs
from test_read_sdata_parquet_tmp_files import original_read_sdata_parquet_tmp_files

zarr_path, tmp_folder = sys.argv[1], sys.argv[2]


def table():
    adata = ad.read_zarr(f"{zarr_path}/tables/table")
    adata.obs.index = [int(i) for i in range(adata.n_obs)]  # cli.py
    adata.obs.index = adata.obs.index.astype(str)
    adata.obs.index.name = "index"
    return {"table": adata}


old, new = table(), table()
original_read_sdata_parquet_tmp_files(old, tmp_folder, "hqcr")
helperfuncs.read_sdata_parquet_tmp_files(new, tmp_folder, "hqcr")
pd.testing.assert_frame_equal(new["table"].obs, old["table"].obs)
print(
    f"on its own: {new['table'].obs.shape[1]} obs columns, identical to the original's read"
)

in_memory = new["table"].obs.copy()
old = {"table": ad.AnnData(obs=in_memory.copy())}
new = {"table": ad.AnnData(obs=in_memory.copy())}
original_read_sdata_parquet_tmp_files(old, tmp_folder, "hqcr")
helperfuncs.read_sdata_parquet_tmp_files(new, tmp_folder, "hqcr")
pd.testing.assert_frame_equal(new["table"].obs, old["table"].obs)
pd.testing.assert_frame_equal(new["table"].obs, in_memory)
print(
    "in-process (-s all): every file skipped, obs unchanged and identical to the original's"
)

# a step on its own: cli's mandatory block (correct_for_valid_geometries) set these first
full = in_memory
geometry = [f"{w}valid_{o}_geometry" for o in ("cell", "nucleus") for w in ("", "w")]
mandatory = table()
for col in geometry:
    mandatory["table"].obs[col] = full[col].to_numpy()
helperfuncs.read_sdata_parquet_tmp_files(mandatory, tmp_folder, "hqcr")
pd.testing.assert_frame_equal(mandatory["table"].obs[full.columns], full)
print(f"on its own with {geometry} already set: the other columns read, identical to the full read")

# with an annotation file: write_out_anndata('overview') drops nuclei_idxs, then
# load_cell_metrices reads the tmp files again. Only that column is read back, from cellqc's file.
dropped = {"table": ad.AnnData(obs=in_memory.drop(columns=["nuclei_idxs"]))}
original = {"table": ad.AnnData(obs=in_memory.drop(columns=["nuclei_idxs"]))}
original_read_sdata_parquet_tmp_files(original, tmp_folder, "hqcr")
assert "nuclei_idxs" not in original["table"].obs  # origin/dev swallowed the overlap
helperfuncs.read_sdata_parquet_tmp_files(dropped, tmp_folder, "hqcr")
pd.testing.assert_frame_equal(dropped["table"].obs[in_memory.columns], in_memory)
print("after write_out_anndata's drop: nuclei_idxs read back (origin/dev: not read), the rest unchanged")
