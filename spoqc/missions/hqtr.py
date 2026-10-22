
from .. import image_analysis
from .. import missions

def start_exploration(enterprise):

    if enterprise.args.step in ['all', 'unittest', 'hqtr', 'hqtr_clustering']:

        print("[NOTE] Calculate Priors and combine them")
        missions.combine_priors.combine_priors_hqtr(enterprise)
        print('[finish]')   


    if enterprise.args.step in ['all', 'unittest', 'hqtr', 'hqtr_refinement']:

        image_analysis.pixel_scoring_refinement.start_pixel_mask_refinement (
                enterprise.args.output_dir,
                enterprise.args.tmp_dir,
                'hqtr',
                enterprise.cargo.dim_x,
                enterprise.cargo.dim_y,
                enterprise.args.chunk_size,
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
            enterprise.args.overwrite,
            enterprise.args.chunk_size,
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
                enterprise.hqcr_set.cell_clustering_df,
            )

            print("[finish]")

        else:
            print("[NOTE] No annotation file provided so I will not perform celltype_refinement_of_hqtr")