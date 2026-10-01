"""helperfuncs.ddf_to_parquet at origin/dev (db00d98), verbatim: the reference writer for core.parquet.write_parts."""

from typing import List, Sequence


def ddf_to_parquet(
    ddf: "dd.DataFrame",
    prefix: str,
    spoqc_tmp_folder: str,
    obs_columns: Sequence[str],
    suffix: str,
    *,
    partition_on: str = None,
    include_index: bool = True,
    overwrite: bool = True,
    engine: str = "pyarrow",
) -> List[str]:
    """
    Write the non-observation columns of a Dask DataFrame to Parquet.

    Parameters
    ----------
    df : dask.dataframe.DataFrame
        Input Dask DataFrame.
    prefix, spoqc_tmp_folder, suffix : str
        Used to form the output path: {spoqc_tmp_folder}/{prefix}_output_{suffix}.parquet
    obs_columns : Sequence[str]
        Columns to exclude from the Parquet write.
    include_index : bool, default True
        Whether to persist the index into Parquet.
    partition_on: str, default None
        Give partition key to make use of directory based disk partition.
    overwrite : bool, default True
        Overwrite existing output.
    engine : str, default "pyarrow"
        Parquet engine.

    Returns
    -------
    List[str]
        Column order list: obs_columns + new_columns
    """
    path = f"{spoqc_tmp_folder}/{prefix}_output_{suffix}"

    # Compute columns to write (Dask-friendly; no data materialized)
    obs_set = set(obs_columns)
    new_columns = [c for c in ddf.columns if c not in obs_set]

    # Select only needed columns lazily
    write_ddf = ddf[new_columns]

    # Write to Parquet (this triggers computation)
    write_ddf.to_parquet(
        path,
        engine=engine,
        write_index=include_index,
        overwrite=overwrite,
        partition_on=partition_on,
        compute=True,
    )
