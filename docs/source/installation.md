# Installation

## Pip

```
pip install spoqc
```

Once istalled, run spoQC exactly as described in [Run](run.md).

## Docker

spoQC also provides a prebuilt Docker image (`quay.io/heylf/spoqc:0.1.0`) on Quay.io with all
dependencies already installed. This is useful if you do not want to set up a local Python
environment.

### Using the Docker container

Start a container and drop into a shell, bind-mounting the directory that holds your data so the
container can read and write it:

```
docker run -ti -v /path/to/data:/data quay.io/heylf/spoqc:0.1.0 bash
```

Once inside the container, run spoQC exactly as described in [Run](run.md), for example:

```
python3 -m spoqc -s all -i [input_spatial_data_bundle] -o [output_folder] -t [spoqc_tmp_folder] -n [n_cores] -a [annotation_file]
```

### Using Singularity / Apptainer

Many HPC clusters do not allow running Docker directly, but support Singularity or Apptainer instead. Both can pull and run Docker images directly, so you can use the same spoQC image on such systems.

Pull the Docker image and convert it to a Singularity image file:

```
singularity pull spoqc.sif docker://quay.io/heylf/spoqc:0.1.0
```

Then run spoQC through the image, bind-mounting your data directory:

```
singularity exec -B /path/to/data:/data spoqc.sif \
    python3 -m spoqc -s all -i [input_spatial_data_bundle] -o [output_folder] -t [spoqc_tmp_folder] -n [n_cores] -a [annotation_file]
```

On clusters using the newer Apptainer branding, `apptainer` is a drop-in replacement for
`singularity`, thus the same commands work by substituting `apptainer` for `singularity`.