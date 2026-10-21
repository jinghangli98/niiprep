import numpy as np
import ants
import nibabel as nib
from scipy.ndimage import gaussian_filter
from pathlib import Path
from typing import Literal, Optional, Union, Tuple

def register(
    fixed_path: str,
    moving_path: str,
    output_path: str,
    reg_type: Literal['rigid', 'affine', 'syn'] = 'syn',
    interpolation: str = 'linear',
    norm: bool = False,
) -> Tuple[ants.core.ants_image.ANTsImage, dict]:
    """
    Register a moving image to a fixed image using ANTsPyX.
    
    Args:
        fixed_path (str): Path to fixed/reference image
        moving_path (str): Path to moving image to be registered
        output_path (str): Path to save registered image
        reg_type (str): Registration type ('rigid', 'affine', or 'syn')
        interpolation (str): Interpolation type for resampling
        norm (bool): If True, scale intensities to 0–255 (min–max) and round before saving
    
    Returns:
        tuple: (registered_image, transform_params)
    """
    # Load images
    fixed_image = ants.image_read(fixed_path)
    moving_image = ants.image_read(moving_path)
    
    # Set up registration parameters
    if reg_type == 'rigid':
        transform = 'Rigid'
    elif reg_type == 'affine':
        transform = 'Affine'
    elif reg_type == 'syn':
        transform = 'SyN'
    else:
        raise ValueError("reg_type must be 'rigid', 'affine', or 'syn'")
    
    # Perform registration
    registration = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform=transform,
        interpolator=interpolation
    )
    
    # Save registered image
    registered_image = registration['warpedmovout']
    if norm:
        arr = registered_image.numpy()
        a_min, a_max = float(arr.min()), float(arr.max())
        if a_max > a_min:
            arr = (arr - a_min) / (a_max - a_min) * 255.0
        else:
            arr = np.zeros_like(arr, dtype=np.float64)
        arr = np.round(arr).astype(np.float32)
        registered_image = ants.new_image_like(registered_image, arr)
    ants.image_write(registered_image, output_path)

    return registered_image, registration


def register_seg(
    fixed_path: str,
    moving_path: str,
    output_path: str,
    reg_type: Literal['rigid', 'affine', 'syn'] = 'syn',
    sigma: float = 0.7,
) -> Tuple[ants.core.ants_image.ANTsImage, dict]:
    """
    Register a discrete-label segmentation to a fixed image with anti-aliased
    boundaries.

    Each label in the moving segmentation is one-hot encoded, smoothed with a
    Gaussian in the source space, warped to the fixed image with linear
    interpolation, and finally an argmax across labels is taken per output
    voxel. This preserves the original label set (no fractional labels) while
    removing the stair-step artifacts produced by pure nearest-neighbor /
    genericLabel resampling at the registered boundaries.

    Args:
        fixed_path: Path to fixed/reference NIfTI file.
        moving_path: Path to moving segmentation NIfTI file.
        output_path: Path to save the registered segmentation NIfTI file.
        reg_type: Registration type ('rigid', 'affine', or 'syn').
        sigma: Gaussian sigma in moving voxels applied to each one-hot channel
            before warping. 0 disables smoothing (pure linear argmax). Larger
            values give smoother boundaries.

    Returns:
        tuple: (registered_image, transform_params)
    """
    fixed_image = ants.image_read(fixed_path)
    moving_image = ants.image_read(moving_path)

    if reg_type == 'rigid':
        transform = 'Rigid'
    elif reg_type == 'affine':
        transform = 'Affine'
    elif reg_type == 'syn':
        transform = 'SyN'
    else:
        raise ValueError("reg_type must be 'rigid', 'affine', or 'syn'")

    registration = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform=transform,
    )

    moving_data = np.asarray(moving_image.numpy()).astype(np.int32)
    labels = np.unique(moving_data)

    out_shape = tuple(fixed_image.shape)
    best_prob = np.full(out_shape, -np.inf, dtype=np.float32)
    best_label = np.zeros(out_shape, dtype=np.int32)

    for lbl in labels:
        mask = (moving_data == lbl).astype(np.float32)
        if sigma > 0:
            mask = gaussian_filter(mask, sigma=sigma)
        mask_img = ants.from_numpy(
            mask,
            origin=moving_image.origin,
            spacing=moving_image.spacing,
            direction=moving_image.direction,
        )
        warped = ants.apply_transforms(
            fixed=fixed_image,
            moving=mask_img,
            transformlist=registration['fwdtransforms'],
            interpolator='linear',
        )
        prob = warped.numpy().astype(np.float32)
        if lbl == 0:
            # never let background overwrite a real label at numerical ties
            better = prob > best_prob + 1e-6
        else:
            better = prob > best_prob
        np.putmask(best_label, better, lbl)
        np.putmask(best_prob, better, prob)

    ref = nib.load(fixed_path)
    out_img = nib.Nifti1Image(best_label.astype(np.int32), ref.affine, ref.header)
    out_img.set_data_dtype(np.int32)
    nib.save(out_img, output_path)

    registered_image = ants.from_numpy(
        best_label.astype(np.float32),
        origin=fixed_image.origin,
        spacing=fixed_image.spacing,
        direction=fixed_image.direction,
    )
    return registered_image, registration
