import torchio as tio
import nibabel as nib
import numpy as np

def resample(
    input_path: str,
    output_path: str,
    target_spacing: tuple = (1.0, 1.0, 1.0),
    interpolation: str = 'linear'
) -> None:
    """
    Resample a NIfTI image to specified voxel spacing.
    
    Args:
        input_path (str): Path to input NIfTI file
        output_path (str): Path to save resampled NIfTI file
        target_spacing (tuple): Target voxel spacing in mm (x, y, z)
        interpolation (str): Interpolation method ('linear', 'nearest', 'bspline')
    """
    # Load image
    image = tio.ScalarImage(input_path)
    
    # Create resampling transform
    resample = tio.Resample(target_spacing)
    
    # Apply transform
    resampled_image = resample(image)
    
    # Save resampled image
    resampled_image.save(output_path)
