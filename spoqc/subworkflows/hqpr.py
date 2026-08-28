from .. import image_analysis

def get_hqpr(
        sdata,
        spoqc_tmp_folder,
        imagedim,
        dim_x,
        dim_y,
        CONST,
        seed,
        *,
        thresh_p=None,
        nstds_p=None,
    ):

    # Memory depends on threads. The more threads you choose the more memory you need.
    if ( CONST.STEP in ['all', 'unittest', 'hqpr', 'hqpr_metrices'] ):
        
        image_analysis.structure_analysis.start_image_struc_analyis(
            sdata,
            CONST.FIGURE_PATH,
            spoqc_tmp_folder,
            'hqpr',
            CONST.IMAGE_TYPE,
            CONST.RESOLUTION,
            imagedim,
            dim_x,
            dim_y,
            CONST.OVERWRITE,
            staining=CONST.STAINING,
        )

        print('[finish]')


    if ( CONST.STEP in ['all', 'unittest', 'hqpr', 'hqpr_clustering'] ):

        image_analysis.pixel_scoring_dask.start_pixel_qc(
            sdata,
            CONST.FIGURE_PATH,
            spoqc_tmp_folder,
            'hqpr',
            CONST.IMAGE_TYPE,
            CONST.RESOLUTION,
            dim_x,
            dim_y,
            imagedim,
            seed,
            CONST.THREADS,
            chunk_size=CONST.PIXEL_QC_CHUNK_SIZE,
            sample_size=CONST.KMEANS_SAMPLE_SIZE,
            staining=CONST.STAINING,
            thresh_p=thresh_p,
            nstds_p=nstds_p,
        )

        print('[finish]')   

    if ( CONST.STEP in ['all', 'unittest', 'hqpr', 'hqpr_refinement'] ):

        image_analysis.pixel_scoring_refinement.start_pixel_mask_refinement (
                CONST.FIGURE_PATH,
                spoqc_tmp_folder,
                'hqpr',
                dim_x,
                dim_y,
                1.5,
                15,
                staining=CONST.STAINING,
        )

        print('[finish]')


    if ( CONST.STEP in ['all', 'unittest', 'hqpr', 'hqpr_bounding_box'] ):

        image_analysis.bounding_boxes.define_bounding_boxes(
            sdata,
            CONST.FIGURE_PATH,
            spoqc_tmp_folder,
            'hqpr',
            CONST.IMAGE_TYPE,
            CONST.RESOLUTION,
            dim_x,
            dim_y,
            imagedim,
            'raw',
            staining=CONST.STAINING,
        )

        print('[finish]')

# In[]

def celltype_refinement_of_hqpr(sdata, spoqc_tmp_folder, imagedim, dim_x, dim_y, CONST):

    if ( CONST.STEP in ['all', 'hqpr_celltype'] ):
        
        image_analysis.celltype_analysis.start_image_celltype_analysis(
            sdata,
            CONST.FIGURE_PATH,
            spoqc_tmp_folder,
            'hqpr',
            CONST.IMAGE_TYPE,
            CONST.RESOLUTION,
            imagedim,
            dim_x,
            dim_y,
            CONST.ANNOTATION_KEY,
            CONST.CANORM,
            staining=CONST.STAINING,
        )

        print("[finish]")