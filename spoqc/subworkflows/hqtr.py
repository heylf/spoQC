
from .. import image_analysis
from .. import metrics

def get_hqtr(
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

    if ( CONST.STEP in ['all', 'unittest', 'hqtr', 'hqtr_metrices'] ):

        image_analysis.structure_analysis.start_image_struc_analyis(
            sdata,
            CONST.FIGURE_PATH,
            spoqc_tmp_folder,
            'hqtr',
            CONST.IMAGE_TYPE,
            CONST.RESOLUTION,
            imagedim,
            dim_x,
            dim_y,
            CONST.OVERWRITE
        )

        print('[finish]')

    if ( CONST.STEP in ['all', 'unittest', 'hqtr', 'hqtr_qv'] ):

        metrics.transcript_density.qv_image.transcript_qv_image(
            sdata,
            CONST.FIGURE_PATH,
            spoqc_tmp_folder,
            'hqtr',
            CONST.IMAGE_TYPE,
            CONST.RESOLUTION,
            dim_x,
            dim_y,
            imagedim,
        )

        print('[finish]')

    if ( CONST.STEP in ['all', 'unittest', 'hqtr', 'hqtr_ac'] ):

        metrics.transcript_density.ac_image.transcript_ac_image(
            sdata,
            CONST.FIGURE_PATH,
            spoqc_tmp_folder,
            'hqtr',
            CONST.THREADS,
            CONST.IMAGE_TYPE,
            CONST.RESOLUTION,
            dim_x,
            dim_y,
            imagedim,
        )

        print('[finish]')

    if ( CONST.STEP in ['all', 'unittest', 'hqtr', 'hqtr_clustering'] ):

        image_analysis.pixel_scoring_dask.start_pixel_qc(
            sdata,
            CONST.FIGURE_PATH,
            spoqc_tmp_folder,
            'hqtr',
            CONST.IMAGE_TYPE,
            CONST.RESOLUTION,
            dim_x,
            dim_y,
            imagedim,
            seed,
            CONST.THREADS,
            chunk_size=CONST.PIXEL_QC_CHUNK_SIZE,
            sample_size=CONST.KMEANS_SAMPLE_SIZE,
            thresh_p=thresh_p,
            nstds_p=nstds_p,
        )

        print("[finish]")


# In[]

    if ( CONST.STEP in ['all', 'unittest', 'hqtr', 'hqtr_refinement'] ):

        image_analysis.pixel_scoring_refinement.start_pixel_mask_refinement (
                CONST.FIGURE_PATH,
                spoqc_tmp_folder,
                'hqtr',
                dim_x,
                dim_y,
                1.5,
                15
        )

        print('[finish]')

    if ( CONST.STEP in ['all', 'unittest', 'hqtr', 'hqtr_bounding_box'] ):

        image_analysis.bounding_boxes.define_bounding_boxes(
            sdata,
            CONST.FIGURE_PATH,
            spoqc_tmp_folder,
            'hqtr',
            CONST.IMAGE_TYPE,
            CONST.RESOLUTION,
            dim_x,
            dim_y,
            imagedim,
            'raw',
            dilation_radius=1
        )

        print('[finish]')


def celltype_refinement_of_hqtr(sdata, spoqc_tmp_folder, imagedim, dim_x, dim_y, CONST):

    if ( CONST.STEP in ['all', 'hqtr_celltype'] ):
        
        image_analysis.celltype_analysis.start_image_celltype_analysis(
            sdata,
            CONST.FIGURE_PATH,
            spoqc_tmp_folder,
            'hqtr',
            CONST.IMAGE_TYPE,
            CONST.RESOLUTION,
            imagedim,
            dim_x,
            dim_y,
            CONST.ANNOTATION_KEY,
            CONST.CANORM
        )

        print("[finish]")