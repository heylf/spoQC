# Input

## SpatialData format
Please read upon the [SpatialData](https://github.com/scverse/spatialdata) format in order to use this tool. Your SpatialData bundle should look like this when it is read into python:

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

* Special strings in `sdata.points['transcripts']['cell_id']`:
  * Unassigned transcript have to be get: `-1`
* The following entries require the same cell ids:
  * `sdata['table'].index`
  * `sdata['cell_circles'].index`
  * `sdata['cell_boundaries'].index`
  * `sdata.points['transcripts']['cell_id']`

## Annotation

You can provide an annotation as .csv file.
The file should have the format:

* tab or comma seperate (.tsv/.csv)
* 2 columns (`Barcode` and `Cluster`)
* Header needs to be provided

## Cell cycle gene file:

The tool provides a cell cycle reference file, but this will change soon and the user has to provide one.
