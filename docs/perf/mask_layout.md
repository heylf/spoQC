# hqtr/hqpr mask directories: larger parts, decoded in parallel

The mask directories are written in larger parts, and read by decoding their row groups in parallel.
Values, dtypes, the index and the row order are unchanged; only the file layout changes.

## Layout

| Directory | origin/dev | Now |
| --- | --- | --- |
| `hqtr_output_mask_raw` | 10,000-row parts (the union of `chunk_size` and the priors' 10,000-row divisions): 91,296 files at 913 Mpx | `core.parquet.PART_ROWS` = 4,194,304-row parts: 218 files |
| `hqpr_{s}_output_mask_raw` | `--pixel_qc_chunk_size` parts (200,000 by default): 4,565 files | 218 files |
| `{hqtr,hqpr_{s}}_output_mask_smoothed_raw` | `dd.from_pandas(npartitions=ceil(n / 10,000))`: 91,296 files | 218 files |

Each part is still the file dask's `to_parquet` writes for its rows (`core.parquet.write_parts`): the same schema and pandas metadata, the int64 pixel index `__null_dask_index__`, NaN stored as null.
A part holds four 1,048,576-row row groups (`pq.write_table`'s default); `raster.read_pixel_columns` decodes row groups in parallel.
`--pixel_qc_chunk_size` no longer sets the hqpr mask_raw layout (it still sets the clustering chunks), and the refinement's unused `chunk_size` argument is gone.
`ORIGIN_PRIOR_PART_ROWS` and `core.parquet.from_pandas_starts`, which only reproduced the old layouts, are deleted.

Dask divisions follow the parts (`0, PART_ROWS, 2 PART_ROWS, ..., n - 1`); no spoQC reader uses them.

## Reader: `core.raster.read_pixel_columns(path, columns, n_rows, threads)`

One implementation for every per-pixel parquet read.
The footers are read on `threads` threads; then every row group is decoded once for all requested columns (`read_row_group(use_threads=False)`, which releases the GIL) and copied into preallocated arrays, one row group per task on `threads` threads.
It replaces the Arrow-scanner version, whose single consumer thread copied every batch and so held the pool near 3.1 of 4 cores.
Nulls, type checks and the row-count check behave as before.

The dask readers of these directories now call it (one read per directory for all columns, instead of one `.compute()` per column):

| Reader | Before | Now |
| --- | --- | --- |
| `hqr/combine_masks` | `read_pixel_columns` | + `threads` |
| `image_analysis/bounding_boxes` | `read_pixel_columns` | + `threads` |
| `core/raster.load_intensity_image` (hqtr density) | `read_pixel_columns` | + `threads` |
| `image_analysis/pixel_scoring_refinement` (standalone step: `beliefs_raw=None`) | `dd.read_parquet` | `read_pixel_columns` |
| `image_analysis/celltype_analysis` (annotation only) | two `dd.read_parquet` of mask_raw | one `read_pixel_columns`; frames with the same index |
| `additional_analysis/analysis_funcs.map_modality_metrics_to_cells` (annotation only) | four `dd.read_parquet`, 2-3 `.compute()` each | four `read_pixel_columns` |
| `hqr/combine_masks_zoom` | two `dd.read_parquet` | two `read_pixel_columns` |
| `unittests/test_all.py` (`-s unittest`) | row-hash sum over dask partitions | unchanged: the sum does not depend on the partitions (tested) |

Still on dask, not part of this change: the qv/ac prior reads in `analysis_funcs` and `priors/combine_priors`, and the metric parquets (whose writer, `helperfuncs.nparr_to_parquet`, is changed separately).

## Choosing the part size (measured)

A 64,000,000-row subset (the first 6,400 parts) of the tmp folder of a real full-slide run, rewritten at each part size with `write_parts`.
Measured with `taskset -c 16-19`, 4 threads (`pa.cpu_count() == 4`), warm page cache, 3 runs; cores busy = process CPU time / wall.
A heavy run was using other cores of the host; the one write round it disturbed (cores 1-2) is discarded.

Read, `hqtr_mask` + `hqtr_beliefs` (what combine_masks reads):

| Part rows | Files (64 M) | Scanner reader (before) | Row-group reader (now) |
| --- | --- | --- | --- |
| 10,000 (origin/dev) | 6,400 | 1.16-1.21 s, 2.8-2.9 cores | 2.31-2.42 s, 2.2 cores |
| 200,000 | 320 | 0.20-0.22 s, 3.3-3.5 cores | 0.21 s, 3.5 cores |
| 1,000,000 | 64 | 0.19-0.20 s, 3.1-3.3 cores | 0.16 s, 3.7-3.8 cores |
| 4,194,304 | 16 | 0.19-0.21 s, 3.0-3.2 cores | 0.15 s, 3.8-3.9 cores |
| 8,388,608 | 8 | 0.20 s, 3.1 cores | 0.15 s, 3.9 cores |

From 1 M rows up the read is decode-bound and flat.
Write (`write_parts`, all columns, 4 threads), wall and peak-RSS increase over the loaded columns:

| Part rows | mask_raw (9 columns) | mask_smoothed_raw (3 columns) |
| --- | --- | --- |
| 10,000 | 4.6-4.8 s, 3.6-3.7 cores, +0.01-0.05 GB | 3.3-3.5 s, 3.4-3.5 cores, +0.03-0.06 GB |
| 1,000,000 | 2.5 s, 3.9 cores, +0.20-0.25 GB | 1.2-1.3 s, 3.8-4.0 cores, +0.27-0.30 GB |
| 4,194,304 | 2.7 s, 3.7-3.8 cores, +0.32-0.36 GB | 1.2-1.35 s, 3.3-3.8 cores, +0.34-0.36 GB |
| 8,388,608 | 2.7 s, 3.8 cores, +0.46 GB | 1.2 s, 3.8 cores, +0.46 GB |

`PART_ROWS = 1 << 22` because:

- the read is decode-bound (flat from 1 M to 8 M rows) and the write is flat from 1 M rows;
- 218 parts at 913 Mpx keep 4 x 30 threads busy, which 8 M-row parts (109) would not;
- the write's extra memory is one part per thread in flight: +0.34 GB at 4 threads, about +2.5 GB at 30 threads (extrapolated), against about +1.9 GB at 1 M rows;
- the per-pixel metric parquets (`helperfuncs.nparr_to_parquet`) use the same `parquet.PART_ROWS`: one constant serves both.
  The hqtr qv/ac prior parquets keep their own layout (`priors.hqtr.ac_or_qv.PART_ROWS = 1_000_000`, see `docs/perf/hqtr_ambient.md`).

## Before and after (4 threads)

Full slide = 912,953,648 pixels; the full-slide columns are linear extrapolations (x 14.26) from the 64 M-row subset, labelled "extrapolated".

| Step | Before (64 M) | After (64 M) | Full slide before (extrapolated) | Full slide after (extrapolated) |
| --- | --- | --- | --- | --- |
| combine_masks read of hqtr mask_raw (2 columns) | 1.16-1.21 s, 2.8 cores | 0.15 s, 3.8 cores | 17 s | 2.1 s |
| mask_raw, all 9 columns | 2.0-2.1 s, 3.2-3.3 cores | 0.71 s, 3.7-3.9 cores | 29 s | 10 s |
| mask_smoothed_raw, 3 columns | 1.07 s, 3.1 cores | 0.21 s, 3.7-3.9 cores | 15 s | 3.0 s |
| hqtr mask_raw write | 4.6-4.8 s | 2.5-2.7 s | 67 s | 38 s |
| mask_smoothed_raw write | 3.3-3.5 s | 1.2-1.35 s | 48 s | 18 s |
| hqtr mask_raw files / size | 6,400 / 983 MB | 16 / 758 MB | 91,296 / 14.0 GB | 218 / 10.8 GB |

hqcr's single 913 M-row file (871 row groups), `hqcr_mask`, measured at full size: 0.89-0.91 s at 2.3 cores before, 0.45-0.47 s at 3.7-3.8 cores now.
Peak RSS of the reads is unchanged within 0.3 GB (the output arrays dominate), measured on the new layout only.
A stale tmp folder in the old layout is different: `read_pixel_columns` keeps every part's footer in its `metas` list, about 34 KB per file, so the 91,296-file hqtr mask_raw holds about 3.1 GB of footers during the read.

The row-group reader is slower than the scanner on the old 10,000-row layout (2.3 s against 1.2 s per 64 M rows: per-file Python overhead).
spoQC no longer writes that layout; a tmp folder from an older spoQC is still read correctly, just slower.

## Exactness

- `tests/test_mask_layout.py`: every generic reader (dask's frame with dtypes and index, the unittests' row hash, `read_pixel_columns`, the parts' schema with pandas metadata) sees the same bytes on origin/dev's layout and the new one; the dask readers' previous code (verbatim in `tests/legacy/`) on the old layout hands the same values, dtypes and index to `map_values_to_cells`, the refinement's MRF and the zoom frames as the new code on the new layout; the refinement writes the same rows in PART_ROWS parts.
  Layout mutants (two parts swapped, a row dropped or duplicated, one or every part's dtype changed) fail it. Hand-applied mutants of the reader wiring (a dtype cast in analysis_funcs, a renamed index in celltype_analysis, reversed rows in combine_masks_zoom, a cast in the refinement) each failed it too.
- `tests/test_mask_layout.py::test_production_part_size_is_several_row_groups`: real PART_ROWS parts have 4 row groups and read back exactly.
- `tests/test_pixel_scoring_differential.py`: `start_pixel_qc`'s mask_raw against origin/dev's pipeline, row for row, in 5,000-row parts; `write_parts` at 3,000 to 4,194,304-row parts against origin/dev's frame, byte for byte per row, schema included.
- `tests/test_hqtr_ambient_differential.py`: the refinement's mask_smoothed_raw layout against `dd.from_pandas` + origin/dev's writer.
- `tests/test_read_pixel_columns_differential.py`: the reader against the verbatim previous (Arrow-scanner) reader on every layout, at 1, 2 and 7 threads; 10 reader mutants (part order, row-group order, first row group only, row count check, type check, dtype, null promotion, null values, `out` ignored, integer nulls cast into `out`) are caught.
