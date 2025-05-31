import torchio as tio
import nibabel as nib
import numpy as np

def crop(
    input_path: str,
    output_path: str,
    target_shape: tuple = (256, 256, 256),
) -> None:
    """
    Crop or pad a NIfTI image to specified shape.
    
    Args:
        input_path (str): Path to input NIfTI file
        output_path (str): Path to save resampled NIfTI file
        target_shape (tuple): Target shape (x, y, z)
    """
    # Load image
    image = tio.ScalarImage(input_path)
    
    # Create resampling transform
    cropOrPad = tio.CropOrPad(target_shape)
    
    # Apply transform
    cropped_image = cropOrPad(image)
    
    # Save resampled image
    cropped_image.save(output_path)
