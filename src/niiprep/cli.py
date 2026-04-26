import argparse
import ants
import numpy as np
from .resample import resample
from .registration import register
from .nii2mp4 import nii_to_mp4
from .round import round_nifti
from .denoise_mp2rage import robust_combination
from .crop import crop
from .autocrop import autocrop
from .mask import rotation_mask, otsu_mask, main as mask_main
from .bias_correct_spm import spm_bias_correct, DEFAULT_SPM_PATH
from .patchify_nii import patchify_nii, unpatchify_nii
from .matlab_denoise import mdenoise


def _compute_ants_mask(ants_image, use_rotation=True, rot_axes="z", n_angles=36, threshold=None,
                       threshold_scale=1.0, closing_radius=0, fill_holes=True, blur_sigma=0.0,
                       edge_shrink=False, edge_blur_sigma=1.0, grad_threshold=None):
    """Compute a binary mask and return it as an ANTs image."""
    vol = ants_image.numpy()
    if use_rotation:
        print(f"Computing rotation mask (rot_axes={rot_axes}, n_angles={n_angles})...")
        mask_arr = rotation_mask(vol, n_angles=n_angles, threshold=threshold,
                                 threshold_scale=threshold_scale,
                                 closing_radius=closing_radius, rot_axes=rot_axes,
                                 fill_holes=fill_holes, blur_sigma=blur_sigma,
                                 edge_shrink=edge_shrink, edge_blur_sigma=edge_blur_sigma,
                                 grad_threshold=grad_threshold)
    else:
        print("Computing Otsu mask...")
        mask_arr = otsu_mask(vol)
    return ants.new_image_like(ants_image, mask_arr.astype(np.float32))

def resample_cli():
    parser = argparse.ArgumentParser(description='Resample NIfTI image to specified resolution')
    parser.add_argument('-i', '--input', required=True,
                      help='Path to input NIfTI file')
    parser.add_argument('-o', '--output', required=True,
                      help='Path to save resampled NIfTI file')
    parser.add_argument('-s', '--spacing', nargs=3, type=float, default=[1.0, 1.0, 1.0],
                      help='Target voxel spacing in mm (x y z), default: 1.0 1.0 1.0')
    parser.add_argument('--interpolation', choices=['linear', 'nearest', 'bspline'], 
                      default='linear',
                      help='Interpolation method (default: linear)')
    
    args = parser.parse_args()
    
    resample(
        input_path=args.input,
        output_path=args.output,
        target_spacing=tuple(args.spacing),
        interpolation=args.interpolation
    )

def crop_cli():
    parser = argparse.ArgumentParser(description='Crop Or Pad NIfTI image to specified shape')
    parser.add_argument('-i', '--input', required=True,
                      help='Path to input NIfTI file')
    parser.add_argument('-o', '--output', required=True,
                      help='Path to save resampled NIfTI file')
    parser.add_argument('-s', '--shape', nargs=3, type=float, default=[256, 256, 256],
                      help='Target image shape, default: 256 256 256')
    
    args = parser.parse_args()
    
    crop(
        input_path=args.input,
        output_path=args.output,
        target_shape=tuple(args.shape),
    )

def register_cli():
    parser = argparse.ArgumentParser(description='Register moving image to fixed image')
    parser.add_argument('-f', '--fixed', required=True,
                      help='Path to fixed/reference NIfTI file')
    parser.add_argument('-m', '--moving', required=True,
                      help='Path to moving NIfTI file')
    parser.add_argument('-o', '--output', required=True,
                      help='Path to save registered NIfTI file')
    parser.add_argument('-t', '--type', choices=['rigid', 'affine', 'syn'], 
                      default='syn',
                      help='Registration type (default: syn)')
    parser.add_argument('--interpolation', default='linear',
                      help='Interpolation type (default: linear)')
    parser.add_argument('--norm', action='store_true',
                      help='Min–max normalize intensities to 0–255 and round before saving')
    
    args = parser.parse_args()
    
    register(
        fixed_path=args.fixed,
        moving_path=args.moving,
        output_path=args.output,
        reg_type=args.type,
        interpolation=args.interpolation,
        norm=args.norm,
    ) 

def nii_to_mp4_cli():
    parser = argparse.ArgumentParser(description='Convert NIfTI file to MP4 video')
    parser.add_argument('-i', '--input', required=True,
                      help='Path to input NIfTI file')
    parser.add_argument('-o', '--output', required=True,
                      help='Path to save MP4 file')
    parser.add_argument('-d', '--dimension', type=int, default=2, choices=[0, 1, 2],
                      help='Dimension to slice along (0: sagittal, 1: coronal, 2: axial (default))')
    parser.add_argument('--fps', type=int, default=10,
                      help='Frames per second (default: 10)')
    parser.add_argument('--no-normalize', action='store_false', dest='normalize',
                      help='Disable intensity normalization')
    
    args = parser.parse_args()
    
    # Ensure output path has .mp4 extension
    output_path = args.output
    if not output_path.endswith('.mp4'):
        output_path += '.mp4'
    
    nii_to_mp4(
        input_path=args.input,
        output_path=output_path,
        dimension=args.dimension,
        fps=args.fps,
        normalize=args.normalize
    ) 

def round_cli():
    parser = argparse.ArgumentParser(description='Round NIfTI image pixel values')
    parser.add_argument('-i', '--input', required=True,
                      help='Path to input NIfTI file (will be overwritten)')
    
    args = parser.parse_args()
    
    round_nifti(args.input) 

def denoise_mp2rage():

    parser = argparse.ArgumentParser(description='MP2RAGE robust combination processing')
    
    parser.add_argument('--uni', type=str, required=True,
                        help='Path to UNI image (.nii or .nii.gz)')
    parser.add_argument('--inv1', type=str, required=True,
                        help='Path to INV1 image (.nii or .nii.gz)')
    parser.add_argument('--inv2', type=str, required=True,
                        help='Path to INV2 image (.nii or .nii.gz)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output path for processed image')
    parser.add_argument('--regularization', '-r', type=float, default=None,
                        help='Noise regularization factor (default: None for interactive mode)')

    args = parser.parse_args()

    mp2rage_data = {
        'filenameUNI': args.uni,
        'filenameINV1': args.inv1,
        'filenameINV2': args.inv2,
        'filenameOUT': args.output
    }
    _, _ = robust_combination(mp2rage_data, regularization=args.regularization,)
def denoise_cli():
    parser = argparse.ArgumentParser(description='Denoise a NIfTI image using ANTs')
    parser.add_argument('-i', '--input', required=True,
                      help='Path to input NIfTI file')
    parser.add_argument('-o', '--output', required=True,
                      help='Path to save denoised NIfTI file')
    parser.add_argument('--norm', action='store_true',
                      help='Min–max normalize intensities to 0–255 and round before saving')
    parser.add_argument('--ants-mask', action='store_true',
                      help='Use ANTs built-in mask instead of the default rotation-based mask')
    parser.add_argument('--no-rotation', action='store_true',
                      help='Use fast Otsu mask instead of rotation-based mask')
    parser.add_argument('--axes', type=str, default='x,y,z',
                      help='Axes to rotate around (e.g. z, or xyz for all 3)')
    parser.add_argument('--n_angles', type=int, default=36,
                      help='Number of rotation angles for mask (default: 36)')
    parser.add_argument('--threshold', type=float, default=None,
                      help='Intensity threshold for mask (default: Otsu)')
    parser.add_argument('--threshold_scale', type=float, default=0.2,
                      help='Scale factor applied to the mask threshold (default: 1.0). '
                           'Use < 1.0 (e.g. 0.5) for 7T data with low signal in the neck region.')
    parser.add_argument('--closing_radius', type=int, default=0,
                      help='Morphological closing radius after rotation mask (default: 0 = off). '
                           'Bridges Z-axis signal dropout gaps at the bottom of the neck.')
    parser.add_argument('--no-fill-holes', action='store_true',
                      help='Disable filling holes and extracting largest connected component in final mask')
    parser.add_argument('--blur_sigma', type=float, default=3.0,
                      help='Gaussian blur sigma for smoothing the mask before saving (default: 0 = off)')
    parser.add_argument('--edge-shrink', action='store_true', help='Shrink the mask inwards until hitting a strong gradient spike (true skin boundary).')
    parser.add_argument('--edge-blur-sigma', type=float, default=1.0, help='Blur applied before gradient calculation for shrinking. Default: 1.0.')
    parser.add_argument('--grad-threshold', type=float, default=0.1, help='Gradient threshold for shrink stop condition. Default is auto Otsu.')
    args = parser.parse_args()

    image = ants.image_read(args.input)

    if args.ants_mask:
        mask = None
    else:
        mask = _compute_ants_mask(image, use_rotation=not args.no_rotation,
                                  rot_axes=args.axes,
                                  n_angles=args.n_angles, threshold=args.threshold,
                                  threshold_scale=args.threshold_scale,
                                  closing_radius=args.closing_radius,
                                  fill_holes=not args.no_fill_holes,
                                  blur_sigma=args.blur_sigma,
                                  edge_shrink=args.edge_shrink,
                                  edge_blur_sigma=args.edge_blur_sigma,
                                  grad_threshold=args.grad_threshold)

    denoised = ants.denoise_image(image, mask=mask)

    # Round the image
    arr = denoised.numpy()
    arr = np.round(arr)

    # Normalize if requested
    if args.norm:
        a_min, a_max = float(arr.min()), float(arr.max())
        if a_max > a_min:
            arr = (arr - a_min) / (a_max - a_min) * 255.0
        else:
            arr = np.zeros_like(arr, dtype=np.float64)
        arr = np.round(arr)

    arr = arr.astype(np.float32)
    denoised = ants.new_image_like(denoised, arr)
    ants.image_write(denoised, args.output)


def mdenoise_cli():
    parser = argparse.ArgumentParser(description='Denoise a NIfTI image using the MATLAB denoising routine')
    parser.add_argument('-i', '--input', required=True,
                      help='Path to input NIfTI file')
    parser.add_argument('-o', '--output', required=True,
                      help='Path to save denoised NIfTI file')

    args = parser.parse_args()

    mdenoise(args.input, args.output)

def biascorrect_cli():
    parser = argparse.ArgumentParser(description='N4 bias field correction using ANTs')
    parser.add_argument('-i', '--input', required=True,
                      help='Path to input NIfTI file')
    parser.add_argument('-o', '--output', required=True,
                      help='Path to save bias-corrected NIfTI file')
    parser.add_argument('--norm', action='store_true',
                      help='Min–max normalize intensities to 0–255 and round before saving')
    parser.add_argument('--ants-mask', action='store_true',
                      help='Use ANTs built-in mask instead of the default rotation-based mask')
    parser.add_argument('--no-rotation', action='store_true',
                      help='Use fast Otsu mask instead of rotation-based mask')
    parser.add_argument('--axes', type=str, default='x,y,z',
                      help='Axes to rotate around (e.g. z, or xyz for all 3)')
    parser.add_argument('--n_angles', type=int, default=36,
                      help='Number of rotation angles for mask (default: 36)')
    parser.add_argument('--threshold', type=float, default=None,
                      help='Intensity threshold for mask (default: Otsu)')
    parser.add_argument('--threshold_scale', type=float, default=0.2,
                      help='Scale factor applied to the mask threshold (default: 1.0). '
                           'Use < 1.0 (e.g. 0.5) for 7T data with low signal in the neck region.')
    parser.add_argument('--closing_radius', type=int, default=0,
                      help='Morphological closing radius after rotation mask (default: 0 = off). '
                           'Bridges Z-axis signal dropout gaps at the bottom of the neck.')
    parser.add_argument('--no-fill-holes', action='store_true',
                      help='Disable filling holes and extracting largest connected component in final mask')
    parser.add_argument('--blur_sigma', type=float, default=3.0,
                      help='Gaussian blur sigma for smoothing the mask before saving (default: 0 = off)')
    parser.add_argument('--edge-shrink', action='store_true', help='Shrink the mask inwards until hitting a strong gradient spike (true skin boundary).')
    parser.add_argument('--edge-blur-sigma', type=float, default=1.0, help='Blur applied before gradient calculation for shrinking. Default: 1.0.')
    parser.add_argument('--grad-threshold', type=float, default=0.1, help='Gradient threshold for shrink stop condition. Default is auto Otsu.')

    args = parser.parse_args()

    image = ants.image_read(args.input)

    if args.ants_mask:
        mask = None
    else:
        mask = _compute_ants_mask(image, use_rotation=not args.no_rotation,
                                  rot_axes=args.axes,
                                  n_angles=args.n_angles, threshold=args.threshold,
                                  threshold_scale=args.threshold_scale,
                                  closing_radius=args.closing_radius,
                                  fill_holes=not args.no_fill_holes,
                                  blur_sigma=args.blur_sigma,
                                  edge_shrink=args.edge_shrink,
                                  edge_blur_sigma=args.edge_blur_sigma,
                                  grad_threshold=args.grad_threshold)

    corrected = ants.n4_bias_field_correction(image, mask=mask)

    # Round the image
    arr = corrected.numpy()
    arr = np.round(arr)

    # Normalize if requested
    if args.norm:
        a_min, a_max = float(arr.min()), float(arr.max())
        if a_max > a_min:
            arr = (arr - a_min) / (a_max - a_min) * 255.0
        else:
            arr = np.zeros_like(arr, dtype=np.float64)
        arr = np.round(arr)

    arr = arr.astype(np.float32)
    corrected = ants.new_image_like(corrected, arr)
    ants.image_write(corrected, args.output)

def mask_cli():
    parser = argparse.ArgumentParser(description='Generate a binary brain/object mask from a NIfTI image')
    parser.add_argument('-i', '--input', required=True,
                      help='Path to input NIfTI file')
    parser.add_argument('-o', '--output', default=None,
                      help='Path to output mask NIfTI file (default: <input>_mask.nii)')
    parser.add_argument('--no-rotation', action='store_true',
                      help='Use fast Otsu mask instead of rotation-based mask')
    parser.add_argument('--axes', type=str, default='x,y,z',
                      help='Axes to rotate around (e.g. z, or xyz for all 3)')
    parser.add_argument('--n_angles', type=int, default=36,
                      help='Number of rotation angles (default: 36)')
    parser.add_argument('--threshold', type=float, default=None,
                      help='Intensity threshold (default: Otsu)')
    parser.add_argument('--threshold_scale', type=float, default=0.2,
                      help='Scale factor applied to the threshold (default: 1.0). '
                           'Use < 1.0 (e.g. 0.5) for 7T data with low signal in the neck region.')
    parser.add_argument('--closing_radius', type=int, default=0,
                      help='Morphological closing radius after rotation mask (default: 0 = off). '
                           'Bridges Z-axis signal dropout gaps at the bottom of the neck.')
    parser.add_argument('--no-fill-holes', action='store_true',
                      help='Disable filling holes and extracting largest connected component in final mask')
    parser.add_argument('--blur_sigma', type=float, default=3.0,
                      help='Gaussian blur sigma for smoothing the mask before saving (default: 0 = off)')
    parser.add_argument('--edge-shrink', action='store_true', help='Shrink the mask inwards until hitting a strong gradient spike (true skin boundary).')
    parser.add_argument('--edge-blur-sigma', type=float, default=1.0, help='Blur applied before gradient calculation for shrinking. Default: 1.0.')
    parser.add_argument('--grad-threshold', type=float, default=None, help='Gradient threshold for shrink stop condition. Default is auto Otsu.')

    args = parser.parse_args()
    
    mask_main(
        input_path=args.input,
        output_path=args.output,
        use_rotation=not args.no_rotation,
        rot_axes=args.axes,
        n_angles=args.n_angles,
        threshold=args.threshold,
        threshold_scale=args.threshold_scale,
        closing_radius=args.closing_radius,
        fill_holes=not args.no_fill_holes,
        blur_sigma=args.blur_sigma,
        edge_shrink=args.edge_shrink,
        edge_blur_sigma=args.edge_blur_sigma,
        grad_threshold=args.grad_threshold,
    )


def mbiascorrect_cli():
    parser = argparse.ArgumentParser(description='SPM12 bias field correction for NIfTI images')
    parser.add_argument('-i', '--input', required=True,
                      help='Path to input NIfTI file (.nii or .nii.gz)')
    parser.add_argument('-o', '--output', required=True,
                      help='Path to save bias-corrected NIfTI file (.nii or .nii.gz)')
    parser.add_argument('--spm-path', default=DEFAULT_SPM_PATH,
                      help=f'Path to SPM12 installation (default: {DEFAULT_SPM_PATH})')
    parser.add_argument('--window-size', type=int, default=60,
                      help='Bias FWHM in mm (default: 60)')
    parser.add_argument('--sphere-size', type=int, default=50,
                      help='Sphere size for brain mask in mm (default: 50)')
    parser.add_argument('--include-c3', action='store_true',
                      help='Include CSF (c3) tissue class in segmentation')
    parser.add_argument('--save-masks', action='store_true',
                      help='Save SPM tissue segmentation masks (c1-c5) alongside output')
    parser.add_argument('--round', action='store_true',
                      help='Round output intensity values to nearest integer')

    args = parser.parse_args()

    spm_bias_correct(
        input_path=args.input,
        output_path=args.output,
        spm_path=args.spm_path,
        window_size=args.window_size,
        sphere_size=args.sphere_size,
        include_c3=args.include_c3,
        save_masks=args.save_masks,
        round_output=args.round,
    )


def autocrop_cli():
    parser = argparse.ArgumentParser(description='Autocrop NIfTI image to remove empty space and pad to multiple of N')
    parser.add_argument('-i', '--input', required=True,
                      help='Path to input NIfTI file')
    parser.add_argument('-o', '--output', required=True,
                      help='Path to output NIfTI file')
    parser.add_argument('-n', type=int, default=None,
                      help='Integer multiple for output measurement (default: None, crop tightly)')
    parser.add_argument('-s', '--shape', nargs=3, type=int, default=None,
                      help='Target shape (x y z). If specified with -n, dimensions will be adjusted to closest smaller multiple of n.')

    args = parser.parse_args()

    autocrop(
        input_path=args.input,
        output_path=args.output,
        n=args.n,
        target_shape=tuple(args.shape) if args.shape else None
    )


def patchify_nii_cli():
    parser = argparse.ArgumentParser(description='Split a 3D NIfTI image into patches')
    parser.add_argument('-i', '--input', required=True,
                      help='Path to input NIfTI file')
    parser.add_argument('-o', '--output', required=True,
                      help='Output directory for patches and metadata')
    parser.add_argument('-p', '--patch-size', nargs=3, type=int, required=True,
                      metavar=('PX', 'PY', 'PZ'),
                      help='Patch size in voxels (x y z)')
    parser.add_argument('-s', '--step', type=int, default=None,
                      help='Step size between patches in voxels (default: min(patch_size), non-overlapping)')
    parser.add_argument('--no-pad', action='store_true',
                      help='Disable automatic zero-padding to compatible dimensions (raises error instead)')
    parser.add_argument('--norm', action='store_true',
                      help='Normalize image intensities to [0, 1] before patchifying')
    parser.add_argument('--min-nonzero-frac', type=float, default=0.01,
                      help='Skip patches with fewer non-zero voxels than this fraction (default: 0.01)')
    parser.add_argument('--min-intensity-range', type=float, default=0.0,
                      help='Skip patches whose max-min intensity is <= this value (default: 0.0, disabled)')
    parser.add_argument('--foreground-threshold', type=float, default=None,
                      help='Intensity threshold above which a voxel counts as foreground '
                           '(default: auto Otsu on the full image)')

    args = parser.parse_args()

    patchify_nii(
        input_path=args.input,
        output_dir=args.output,
        patch_size=tuple(args.patch_size),
        step=args.step,
        pad=not args.no_pad,
        norm=args.norm,
        min_nonzero_frac=args.min_nonzero_frac,
        min_intensity_range=args.min_intensity_range,
        foreground_threshold=args.foreground_threshold,
    )


def unpatchify_nii_cli():
    parser = argparse.ArgumentParser(description='Reconstruct a 3D NIfTI image from patches')
    parser.add_argument('-i', '--input', required=True,
                      help='Directory containing patches and patches_meta.json')
    parser.add_argument('-o', '--output', required=True,
                      help='Path to save reconstructed NIfTI file')

    args = parser.parse_args()

    unpatchify_nii(
        input_dir=args.input,
        output_path=args.output,
    )
