import argparse
import ants
import json
import numpy as np
import tempfile
import os
from .resample import resample, resample_seg
from .registration import register
from .nii2mp4 import nii_to_mp4
from .round import round_nifti
from .denoise_mp2rage import robust_combination
from .crop import crop
from .autocrop import autocrop, apply_crop_bounds
from .mask import rotation_mask, otsu_mask, main as mask_main
from .bias_correct_spm import spm_bias_correct, DEFAULT_SPM_PATH
from .patchify_nii import patchify_nii, unpatchify_nii
from .matlab_denoise import mdenoise
from .mip import mip, rotating_mip


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
                      help='Interpolation method for scalar images (default: linear). '
                           'Ignored when --seg is set.')
    parser.add_argument('--seg', action='store_true',
                      help='Treat input as a discrete-label segmentation. Uses per-label '
                           'one-hot + smoothing + argmax to produce anti-aliased boundaries '
                           'without inventing fractional labels.')
    parser.add_argument('--sigma', type=float, default=0.7,
                      help='Gaussian sigma in source voxels for --seg smoothing '
                           '(default: 0.7; 0 disables, larger = smoother).')
    parser.add_argument('--order', type=int, default=1, choices=[0, 1, 2, 3],
                      help='Spline order for --seg interpolation '
                           '(1=linear, 3=cubic; default: 1).')

    args = parser.parse_args()

    if args.seg:
        resample_seg(
            input_path=args.input,
            output_path=args.output,
            target_spacing=tuple(args.spacing),
            sigma=args.sigma,
            order=args.order,
        )
    else:
        resample(
            input_path=args.input,
            output_path=args.output,
            target_spacing=tuple(args.spacing),
            interpolation=args.interpolation,
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
                        help='Path to UNI image (.nii/.nii.gz file or DICOM-series directory)')
    parser.add_argument('--inv1', type=str, required=True,
                        help='Path to INV1 image (.nii/.nii.gz file or DICOM-series directory)')
    parser.add_argument('--inv2', type=str, required=True,
                        help='Path to INV2 image (.nii/.nii.gz file or DICOM-series directory)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output path (.nii/.nii.gz file, or a directory for DICOM output)')

    def parse_reg(value):
        if value.lower() == 'auto':
            return 'auto'
        try:
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid value for regularization: {value}")

    parser.add_argument('--regularization', '-r', type=parse_reg, default='auto',
                        help="Noise regularization factor; a number, or 'auto' to "
                             "auto-detect (default: 'auto')")

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


def mip_cli():
    parser = argparse.ArgumentParser(
        description='Maximum Intensity Projection (MIP) of a NIfTI image '
                    '(e.g. TOF MRA vascular rendering)')
    parser.add_argument('-i', '--input', required=True,
                      help='Path to input NIfTI file')
    parser.add_argument('-o', '--output', required=True,
                      help='Path to save MIP NIfTI file')
    parser.add_argument('-a', '--axis', type=int, default=2, choices=[0, 1, 2],
                      help='Axis to project along '
                           '(0: sagittal, 1: coronal, 2: axial (default))')
    parser.add_argument('--slab', type=int, default=None,
                      help='Sliding-slab thickness in voxels. If omitted, a '
                           'single full projection is produced; if set, a '
                           'same-shape volume of slab MIPs is produced.')
    parser.add_argument('--gif', action='store_true',
                      help='Generate a rotating MIP cine as an animated GIF '
                           'instead of a NIfTI file')
    parser.add_argument('--mp4', action='store_true',
                      help='Generate a rotating MIP cine as an MP4 video '
                           'instead of a NIfTI file')
    parser.add_argument('--spin-axis', type=int, default=2, choices=[0, 1, 2],
                      dest='spin_axis',
                      help='(--gif/--mp4) Axis the volume rotates around '
                           '(default: 2, cranio-caudal for TOF head data)')
    parser.add_argument('--frames', type=int, default=36,
                      help='(--gif/--mp4) Number of frames over a full '
                           '360 deg rotation (default: 36)')
    parser.add_argument('--fps', type=int, default=10,
                      help='(--gif/--mp4) Playback frames per second '
                           '(default: 10)')
    parser.add_argument('--no-gpu', action='store_true', dest='no_gpu',
                      help='(--gif/--mp4) Force the CPU backend even if a '
                           'CUDA GPU is available (GPU is used by default '
                           'when CuPy and a device are present)')

    args = parser.parse_args()

    if args.gif and args.mp4:
        raise SystemExit("error: --gif and --mp4 are mutually exclusive")

    if args.gif or args.mp4:
        fmt = 'gif' if args.gif else 'mp4'
        ext = '.' + fmt
        output_path = args.output
        for nii_ext in ('.nii.gz', '.nii'):
            if output_path.endswith(nii_ext):
                output_path = output_path[:-len(nii_ext)]
                break
        if not output_path.endswith(ext):
            output_path += ext
        rotating_mip(
            input_path=args.input,
            output_path=output_path,
            axis=args.axis,
            spin_axis=args.spin_axis,
            frames=args.frames,
            fps=args.fps,
            fmt=fmt,
            slab=args.slab,
            use_gpu=not args.no_gpu,
        )
    else:
        mip(
            input_path=args.input,
            output_path=args.output,
            axis=args.axis,
            slab=args.slab,
        )

def mdenoise_cli():
    parser = argparse.ArgumentParser(
        description='Denoise a NIfTI image using the LAVI VST + BM4D MATLAB pipeline')
    parser.add_argument('-i', '--input', required=True,
                      help='Path to input NIfTI file')
    parser.add_argument('-o', '--output', required=True,
                      help='Path to save denoised NIfTI file')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--lc', dest='profile', action='store_const', const='lc',
                       help='BM4D less aggressive / fastest')
    group.add_argument('--np', dest='profile', action='store_const', const='np',
                       help='BM4D normal aggressiveness (default)')
    group.add_argument('--mp', dest='profile', action='store_const', const='mp',
                       help='BM4D more aggressive / slowest')
    parser.set_defaults(profile='np')

    parser.add_argument('-c', '--cores', type=int, default=None,
                      help='Number of CPU cores/threads to use '
                           '(default: all available cores)')

    args = parser.parse_args()

    mdenoise(args.input, args.output, profile=args.profile, cores=args.cores)

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
    parser.add_argument('-s', '--step', type=int, nargs='+', default=None, metavar='S',
                      help='Step size between patches in voxels. Pass one int for uniform step '
                           'on all axes, or three ints (sx sy sz) for per-axis step. '
                           '(default: min(patch_size), non-overlapping on all axes)')
    parser.add_argument('--no-pad', action='store_true',
                      help='Disable automatic zero-padding to compatible dimensions (raises error instead)')
    parser.add_argument('--norm', action='store_true',
                      help='Normalize image intensities to [0, 1] before patchifying')
    parser.add_argument('--skip', action='store_true',
                      help='Skip background/noise patches that do not pass the foreground filter')
    parser.add_argument('--min-nonzero-frac', type=float, default=0.01,
                      help='(--skip) Min fraction of foreground voxels to keep a patch (default: 0.01)')
    parser.add_argument('--min-intensity-range', type=float, default=0.0,
                      help='(--skip) Min max-min intensity to keep a patch (default: 0.0, disabled)')
    parser.add_argument('--foreground-threshold', type=float, default=None,
                      help='(--skip) Intensity threshold for foreground voxels (default: auto Otsu)')
    parser.add_argument('--autocrop', action='store_true',
                      help='Autocrop the image to remove empty space before patchifying')
    parser.add_argument('--align-to', type=str, default=None, metavar='DIR',
                      help='Reuse crop bounds and skip indices from a previous patchify output '
                           'directory so patches align exactly with that run')

    args = parser.parse_args()

    if args.autocrop and args.align_to:
        raise SystemExit("error: --autocrop and --align-to are mutually exclusive")

    if args.step is not None:
        if len(args.step) == 1:
            args.step = args.step[0]
        elif len(args.step) != 3:
            raise SystemExit("error: --step takes 1 value (uniform) or 3 values (sx sy sz)")

    input_path = args.input
    tmp_file = None
    crop_bounds = None
    forced_skip_indices = None

    if args.autocrop:
        tmp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        tmp_file.close()
        print(f"Autocropping {input_path} ...")
        crop_bounds = autocrop(input_path=input_path, output_path=tmp_file.name)
        input_path = tmp_file.name

    elif args.align_to:
        meta_path = os.path.join(args.align_to, 'patches_meta.json')
        if not os.path.exists(meta_path):
            raise SystemExit(f"error: patches_meta.json not found in {args.align_to}")
        with open(meta_path) as f:
            ref_meta = json.load(f)
        crop_bounds = ref_meta.get('crop_bounds')
        forced_skip_indices = ref_meta.get('skipped_indices', [])
        if crop_bounds is not None:
            tmp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
            tmp_file.close()
            print(f"Applying crop bounds from {args.align_to} ...")
            apply_crop_bounds(input_path=input_path, output_path=tmp_file.name, bounds=crop_bounds)
            input_path = tmp_file.name
        if forced_skip_indices:
            print(f"Inheriting {len(forced_skip_indices)} skipped patch indices from {args.align_to}")

    try:
        patchify_nii(
            input_path=input_path,
            output_dir=args.output,
            patch_size=tuple(args.patch_size),
            step=args.step,
            pad=not args.no_pad,
            norm=args.norm,
            skip=args.skip,
            min_nonzero_frac=args.min_nonzero_frac,
            min_intensity_range=args.min_intensity_range,
            foreground_threshold=args.foreground_threshold,
            forced_skip_indices=forced_skip_indices,
            crop_bounds=crop_bounds,
        )
    finally:
        if tmp_file is not None and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)


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
