# Input

## SpatialData format

Before using spoQC, please make sure your data is stored in the [SpatialData](https://github.com/scverse/spatialdata) format.

When your SpatialData bundle is loaded in Python, it should look similar to this:

```
SpatialData object, with associated Zarr store: spatialdata
├── Images
│     └── 'morphology_focus': DataTree[cyx] (5, 23912, 34154), (5, 11956, 17077), (5, 5978, 8538), (5, 2989, 4269), (5, 1494, 2134)
├── Labels
│     ├── 'cell_labels': DataTree[yx] (23912, 34154), (11956, 17077), (5978, 8538), (2989, 4269), (1494, 2134)
│     └── 'nucleus_labels': DataTree[yx] (23912, 34154), (11956, 17077), (5978, 8538), (2989, 4269), (1494, 2134)
├── Points
│     └── 'transcripts': DataFrame with shape: (<Delayed>, 13) (3D points)
├── Shapes
│     ├── 'cell_boundaries': GeoDataFrame shape: (63173, 1) (2D shapes)
│     ├── 'cell_circles': GeoDataFrame shape: (63173, 2) (2D shapes)
│     └── 'nucleus_boundaries': GeoDataFrame shape: (63036, 1) (2D shapes)
└── Tables
      └── 'table': AnnData (63173, 5006)
with coordinate systems:
    ▸ 'global', with elements:
        morphology_focus (Images), cell_labels (Labels), nucleus_labels (Labels), transcripts (Points), cell_boundaries (Shapes), cell_circles (Shapes), nucleus_boundaries (Shapes)
```

**Please make sure that:**

- Unassigned transcripts in `sdata.points['transcripts']['cell_id']` are marked with:

  ```
  -1
  ```

- The following entries use the same cell IDs:

  - `sdata['table'].obs.index`
  - `sdata['cell_circles'].index`
  - `sdata['cell_boundaries'].index`
  - `sdata['nucleus_boundaries']['cell_id]`
  - `sdata.points['transcripts']['cell_id']`

Matching cell IDs are required so that spoQC can connect transcripts, cell boundaries, cell circles, and the AnnData table correctly.

## Annotation

You can provide a cell type annotation as a `.csv` or `.tsv` file.

The annotation file must have the following format:

- tab-separated or comma-separated file (`.tsv` or `.csv`)
- 2 columns: `Barcode` and `Cluster`
- a header row must be included

Example:

```
Barcode	Cluster
cell_1	T cell
cell_2	B cell
cell_3	Macrophage
```

## Cell cycle gene file

spoQC currently provides a default cell cycle reference file.

This behavior will change in a future version. Users will then need to provide their own cell cycle gene file.