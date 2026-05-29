# Nextflow subworkflow

You can use the tool sequential, but it will take 4-5 days to complete everything. In order to speed things up, we provide a nextflow subworkflow that will reduce the time to 1-2 days. You can find the subworkflow under [nf-core/spatialxe](https://github.com/nf-core/spatialxe/tree/dev) in the spoQC branch. SpoQC needs an HPC infrastructure to perform all tasks on a full SRT datset with full resolution. You might be able to perform spoQC locally with a lower resolution or with subsetting your data.

> [!NOTE]
> We are developing a Nextflow SRT QC sub-workflow repository to support all technologies. If new data modalities are added to spoQC they will also be present in this repository: [soon to come]().