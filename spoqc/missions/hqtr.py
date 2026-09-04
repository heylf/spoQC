
from .. import image_analysis
from .. import metrics

def get_hqtr(enterprise):

    if enterprise.args.step in ['all', 'unittest', 'hqtr', 'hqtr_metrices']:

        image_analysis.structure_analysis.start_image_struc_analyis(
            enterprise.cargo.sdata,
            enterprise.args.output_dir,
            enterprise.args.tmp_dir,
            'hqtr',
            enterprise.args.image_type,
            enterprise.args.resolution,
            enterprise.cargo.imagedim,
            enterprise.cargo.dim_x,
            enterprise.cargo.dim_y,
            enterprise.args.overwrite,
        )

        print('[finish]')

    if enterprise.args.step in ['all', 'unittest', 'hqtr', 'hqtr_qv']:

        metrics.transcript_density.qv_image.transcript_qv_image(
            enterprise.cargo.sdata,
            enterprise.args.output_dir,
            enterprise.args.tmp_dir,
            'hqtr',
            enterprise.cargo.imagedim,
            enterprise.cargo.dim_x,
            enterprise.cargo.dim_y,
        )

        print('[finish]')

    if enterprise.args.step in ['all', 'unittest', 'hqtr', 'hqtr_ac']:

        metrics.transcript_density.ac_image.transcript_ac_image(
            enterprise.cargo.sdata,
            enterprise.args.output_dir,
            enterprise.args.tmp_dir,
            'hqtr',
            enterprise.args.nthreads,
            enterprise.cargo.imagedim,
            enterprise.cargo.dim_x,
            enterprise.cargo.dim_y,
        )

        print('[finish]')

    if enterprise.args.step in ['all', 'unittest', 'hqtr', 'hqtr_clustering']:

        image_analysis.pixel_scoring_dask.start_pixel_qc(
            enterprise.cargo.sdata,
            enterprise.args.output_dir,
            enterprise.args.tmp_dir,
            'hqtr',
            enterprise.args.image_type,
            enterprise.args.resolution,
            enterprise.cargo.imagedim,
            enterprise.cargo.dim_x,
            enterprise.cargo.dim_y,
            enterprise.args.seed,
            enterprise.args.nthreads,
            chunk_size=enterprise.args.pixel_qc_chunk_size,
            sample_size=enterprise.args.kmeans_sample_size,
            thresh_p=enterprise.args.thresh_prior_pixel,
            nstds_p=enterprise.args.nstds_prior_pixel,
        )

        print("[finish]")


    if enterprise.args.step in ['all', 'unittest', 'hqtr', 'hqtr_refinement']:

        image_analysis.pixel_scoring_refinement.start_pixel_mask_refinement (
                enterprise.args.output_dir,
                enterprise.args.tmp_dir,
                'hqtr',
                enterprise.cargo.dim_x,
                enterprise.cargo.dim_y,
        )

        print('[finish]')

    if enterprise.args.step in ['all', 'unittest', 'hqtr', 'hqtr_bounding_box']:

        image_analysis.bounding_boxes.define_bounding_boxes(
            enterprise.cargo.sdata,
            enterprise.args.output_dir,
            enterprise.args.tmp_dir,
            'hqtr',
            enterprise.args.image_type,
            enterprise.args.resolution,
            enterprise.cargo.imagedim,
            enterprise.cargo.dim_x,
            enterprise.cargo.dim_y,
            'raw',
            dilation_radius=1
        )

        print('[finish]')


    if enterprise.args.step in ['all', 'hqtr_celltype']:
        
        if enterprise.args.annotation_file :

            image_analysis.celltype_analysis.start_image_celltype_analysis(
                enterprise.cargo.sdata,
                enterprise.args.output_dir,
                enterprise.args.tmp_dir,
                'hqtr',
                enterprise.args.image_type,
                enterprise.args.resolution,
                enterprise.cargo.imagedim,
                enterprise.cargo.dim_x,
                enterprise.cargo.dim_y,
                enterprise.cargo.celltype_annotation.annotation_key,
                enterprise.args.canorm,
            )

            print("[finish]")

        else:
            print("[NOTE] No annotation file provided so I will not perform celltype_refinement_of_hqtr")