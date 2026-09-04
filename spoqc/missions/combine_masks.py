
from .. import hqr

def run_combine_masks(enterprise):
    if enterprise.args.step in ['all', 'combine_masks']:
        figure_path = f"{enterprise.args.output_dir}/combine_masks/{enterprise.args.staining}"
        
        hqr.combine_masks.start_combining_masks(
            figure_path,
            enterprise.args.tmp_dir,
            enterprise.cargo.imagedim,
            enterprise.cargo.dim_x,
            enterprise.cargo.dim_y,
            enterprise.args.staining,
            celltype_refined=False
        )

        print("[finish]")


    if enterprise.args.step in ['combine_masks_zoom']:

        hqr.combine_masks_zoom.start_combining_masks(
            enterprise.cardo.sdata,
            figure_path,
            enterprise.args.tmp_dir,
            enterprise.cargo.imagedim,
            enterprise.cargo.dim_x,
            enterprise.cargo.dim_y,
            enterprise.args.staining,
            enterprise.args.image_type,
            enterprise.args.resolution,
            celltype_refined=False
        )

        print('[finish]')
