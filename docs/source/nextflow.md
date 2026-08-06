
# Nextflow Subworkflow

<img src="./_static/figures/logo/plus_splialaxe.png" width="400">

spoQC can be executed sequentially, but processing a full-resolution spatial transcriptomics dataset typically takes **4–5 days** to complete.

To significantly reduce runtime, we provide a dedicated Nextflow subworkflow that parallelizes many of the processing steps. Using the Nextflow workflow can reduce the total runtime to approximately **1–2 days**, depending on the available computational resources.

The workflow is available on the **spoQC branch** of [nf-core/spatialaxe](https://github.com/nf-core/spatialaxe/tree/dev).

```{note}
Processing a full-resolution spatial transcriptomics (SRT) dataset with spoQC typically requires access to an HPC (High Performance Computing) environment.
```

If an HPC system is not available, you may still be able to run spoQC locally by:

- Using a lower-resolution dataset.
- Processing a subset of your data.
- Running selected workflow components instead of the complete pipeline.