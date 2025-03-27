import ants
from pathlib import Path
from typing import Literal, Optional, Union, Tuple

def register(
    fixed_path: str,
    moving_path: str,
    output_path: str,
    reg_type: Literal['rigid', 'affine', 'syn'] = 'syn',
    interpolation: str = 'linear'
) -> Tuple[ants.core.ants_image.ANTsImage, dict]:
    """
    Register a moving image to a fixed image using ANTsPyX.
    
    Args:
        fixed_path (str): Path to fixed/reference image
        moving_path (str): Path to moving image to be registered
        output_path (str): Path to save registered image
        reg_type (str): Registration type ('rigid', 'affine', or 'syn')
        interpolation (str): Interpolation type for resampling
    
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
    ants.image_write(registered_image, output_path)
    
    return registered_image, registration
