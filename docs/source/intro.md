<img src="./_static/figures/logo/complex.png" width="1000">

<div style="height: 20px;"></div>

spoQC is a modular framework for multimodal quality control (QC) of imaging-based spatially resolved transcriptomics (SRT). It independently evaluates cell segmentation, imaging, and transcript data to identify high-quality regions (HQRs) across entire tissue sections. In addition, spoQC uses Markov random fields (MRFs) to incorporate spatial dependencies and generate spatially refined QC masks.

```{note}
SpoQC is currently under active development and is still in the alpha phase. You may encounter bugs, incomplete features, or unexpected behavior. If you are testing spoQC and run into any issues, please contact the development team or open an issue in the repository. Feedback, bug reports, and pull requests are highly appreciated and help us improve the project.
```

```{note}
Processing a full-resolution spatial transcriptomics (SRT) dataset with spoQC typically requires access to an HPC (High Performance Computing) environment. For smaller datasets, reduced-resolution data, or data subsets, it may be possible to run spoQC locally.
```

<img src="./_static/figures/logo/plus_splialaxe.png" width="400">

To reduce runtime and improve scalability, we recommend running spoQC with Nextflow. We are continuously working on improving performance and making local execution easier.

# Supported Spatial Transcriptomics Technologies

Currently supported:

- 10x Xenium (XOA v4.0 or lower)

```{note}
Atera support is currently under development and is not yet available.
```

# Cite

If you use spoQC in your work, please cite:

```
@software{spoqc,
  author  = {Heyl, Florian and
             Sen, Ezgi and
             Müller-Bötticher, Niklas and
             Kher, Sameesh and
             He, Dongze and
             Long, Brian and
             Ishaque, Naveed and
             Stegle, Oliver},
  title   = {spoQC},
  url     = {https://bio.tools/spoqc},
  note    = {bio.tools identifier: biotools:spoqc},
  urldate = {2026-08-17}
}
```

# Collaborators

This tool was developed in collaboration with the following institutions:

- German Cancer Research Center (DKFZ, Heidelberg, Germany)
- Centro Nacional de Análisis Genómico (CNAG, Barcelona, Spain)
- Center for Quantitative Analysis of Molecular and Cellular Biosystems (BioQuant, Heidelberg, Germany)
- Berlin Institute of Health at Charité (Berlin, Germany)
- Altos Labs San Diego Institute of Technology (San Diego, USA)
- Allen Institute for Brain Science (Seattle, USA)
- European Molecular Biology Laboratory (EMBL, Heidelberg, Germany)

# Contributors

The following people contributed directly or indirectly through supervision, code review, and the development of concepts and ideas:

- Florian Heyl
- Ezgi Sen
- Niklas Müller-Bötticher
- Sameesh Kher
- Dongze He
- Brian Long
- Naveed Ishaque
- Oliver Stegle