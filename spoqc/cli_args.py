from __future__ import annotations

import argparse


def positive_int(value: str) -> int:
    """argparse type: an integer >= 1."""
    number = int(value)
    if number < 1 or str(number) != value.strip():
        raise argparse.ArgumentTypeError(f"must be an integer >= 1, got {value!r}")
    return number


def build_parser() -> argparse.ArgumentParser:

    print("[NOTE] Use arguments")

    tool_description = """
    """

    # parse command line arguments
    parser = argparse.ArgumentParser(description=tool_description, formatter_class=argparse.RawDescriptionHelpFormatter)

    # version
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 0.1.0")

    # mandatory
    parser.add_argument(
        "-i", "--input",
        dest="input",
        type=str, 
        help="Path to the input directory containing Xenium data.",
        required=True
    )
    parser.add_argument(
        "-o", "--output",
        dest="output",
        type=str, 
        help="Path to the output directory containing the report.",
        required=True
    )
    parser.add_argument(
        "-t",
        dest="tmp",
        type=str, 
        help="Path to the tmp directory where spoQC saves tmp files.",
        required=True
    )

    # optional
    parser.add_argument(
        "-n", "--threads",
        dest="threads",
        type=int,
        default=1, 
        help="Number of cores to be used.",
        required=False
    )
    parser.add_argument(
        "-a", "--annotation",
        dest="annotation",
        type=str,
        help="Path to the annotation file.",
        required=False
    )
    parser.add_argument(
        "--reference",
        dest="reference",
        type=str,
        help="Path to a transcript reference file for the transcript QC.",
        required=False
    )
    parser.add_argument(
        "--cellcycle_gene_file",
        dest="cellcycle_gene_file",
        type=str,
        default='',
        help='Path to a JSON file with "S" and "G2M" keys listing S-phase and G2M-phase gene names.',
        required=False
    )
    parser.add_argument(    
        "-s", "--step",
        dest="step",
        type=str,
        default="all",
        help="Steps to run for QC.",
        required=False
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action='store_true',
        help="Overwriting temporary files.",
        required=False
    )
    parser.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
        help="This is used for to apply standardization to spatial data for the data used in the publication.",
        required=False
    )
    parser.add_argument(
        "--staining",
        dest="staining",
        type=str,
        help="Number of cores to be used.",
        default='0',
        required=False
    )
    parser.add_argument(
        "--thresh_prior_pixel",
        dest="thresh_prior_pixel",
        type=float,
        default=None,
        help="You can set a prior threshold for the pixel prior distribution. Please read the documentation to understand what this threshold does before you set it.",
        required=False
    )
    parser.add_argument(
        "--nstds_prior_pixel",
        dest="nstds_prior_pixel",
        type=float,
        default=6,
        help="You can set the number of stds for the pixel prior distribution. Please read the documentation to understand what this does before you set it.",
        required=False
    )
    parser.add_argument(
        "--pixel_qc_chunk_size",
        dest="pixel_qc_chunk_size",
        type=int,
        default=200_000,
        help="Row-chunk size for the pixel-level QC dask arrays/dataframes (hqpr/hqtr clustering and scoring). Larger values reduce dask task-graph overhead but increase peak memory per chunk.",
        required=False
    )
    parser.add_argument(
        "--kmeans_sample_size",
        dest="kmeans_sample_size",
        type=int,
        default=5_000_000,
        help="Number of pixels randomly subsampled to fit the pixel-cluster MiniBatchKMeans model (hqpr/hqtr). The full dataset is then labeled in parallel using the fitted model.",
        required=False
    )
    parser.add_argument(
        "--gmm_n_init",
        dest="gmm_n_init",
        type=positive_int,
        default=1,
        help="Number of k-means starts for each GaussianMixture prior fit (hqcr negative probes, hqpr/hqtr pixel scores); the fit with the highest likelihood is kept. The 3-component pixel-score fit can have several optima. Default 1 (one seeded start).",
        required=False
    )
    parser.add_argument(
        "--doublet_prior_std",
        dest="doublet_prior_std",
        type=int,
        default=100,
        help="The std for the doublet prior estimation. If you increase it then the impact of doublet events increaes, that means doublets events will impact more cells and give them lower quality.",
        required=False
    )
    parser.add_argument(
        "--cluster_celltype",
        dest="cluster_celltype",
        type=str,
        default=None,
        help="Name of the cluster cell type you want to specifically analyse.",
        required=False
    )
    parser.add_argument(
        "--spatial_smoothing",
        dest="spatial_smoothing",
        action="store_true",
        help="Turn on spatial smoothing for prior beliefs.",
        required=False
    )
    parser.add_argument(
        "--dev_test",
        dest="dev_test",
        action="store_true",
        help="This is just for developing and testing the tool.",
        required=False
    )
    parser.add_argument(
        "--dev_report",
        dest="dev_report",
        action="store_true",
        help="This is just for developing and testing the tool (report).",
        required=False
    )
    

    return parser
