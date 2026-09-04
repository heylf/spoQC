from .. import image_analysis

def get_hqpr(enterprise):

    # Memory depends on threads. The more threads you choose the more memory you need.
    if enterprise.args.step in ['all', 'unittest', 'hqpr', 'hqpr_metrices'] :
        
        image_analysis.structure_analysis.start_image_struc_analyis(
            enterprise.cargo.sdata,
            enterprise.args.output_dir,
            enterprise.args.tmp_dir,
            'hqpr',
            enterprise.args.image_type,
            enterprise.args.resolution,
            enterprise.cargo.imagedim,
            enterprise.cargo.dim_x,
            enterprise.cargo.dim_y,
            enterprise.args.overwrite,
            staining=enterprise.args.staining,
        )

        print('[finish]')


    if enterprise.args.step in ['all', 'unittest', 'hqpr', 'hqpr_clustering']:

        image_analysis.pixel_scoring_dask.start_pixel_qc(
            enterprise.cargo.sdata,
            enterprise.args.output_dir,
            enterprise.args.tmp_dir,
            'hqpr',
            enterprise.args.image_type,
            enterprise.args.resolution,
            enterprise.cargo.imagedim,
            enterprise.cargo.dim_x,
            enterprise.cargo.dim_y,
            enterprise.args.seed,
            enterprise.args.nthreads,
            chunk_size=enterprise.args.pixel_qc_chunk_size,
            sample_size=enterprise.args.kmeans_sample_size,
            staining=enterprise.args.staining,
            thresh_p=enterprise.args.thresh_prior_pixel,
            nstds_p=enterprise.args.nstds_prior_pixel,
        )

        print('[finish]')   

    if enterprise.args.step in ['all', 'unittest', 'hqpr', 'hqpr_refinement']:

        image_analysis.pixel_scoring_refinement.start_pixel_mask_refinement (
                enterprise.args.output_dir,
                enterprise.args.tmp_dir,
                'hqpr',
                enterprise.cargo.dim_x,
                enterprise.cargo.dim_y,
                staining=enterprise.args.staining,
        )

        print('[finish]')


    if enterprise.args.step in ['all', 'unittest', 'hqpr', 'hqpr_bounding_box']:

        image_analysis.bounding_boxes.define_bounding_boxes(
            enterprise.cargo.sdata,
            enterprise.args.output_dir,
            enterprise.args.tmp_dir,
            'hqpr',
            enterprise.args.image_type,
            enterprise.args.resolution,
            enterprise.cargo.imagedim,
            enterprise.cargo.dim_x,
            enterprise.cargo.dim_y,
            'raw',
            staining=enterprise.args.staining,
        )

        print('[finish]')


    if enterprise.args.step in ['all', 'hqpr_celltype']:

        if enterprise.args.annotation_file:

            image_analysis.celltype_analysis.start_image_celltype_analysis(
                enterprise.cargo.sdata,
                enterprise.args.output_dir,
                enterprise.args.tmp_dir,
                'hqpr',
                enterprise.args.image_type,
                enterprise.args.resolution,
                enterprise.cargo.imagedim,
                enterprise.cargo.dim_x,
                enterprise.cargo.dim_y,
                enterprise.cargo.celltype_annotation.annotation_key,
                enterprise.args.canorm,
                staining=enterprise.args.staining,
            )

            print("[finish]")

        else:
            print("[NOTE] No annotation file provided so I will not perform celltype_refinement_of_hqpr")