import nibabel as nib
import numpy as np

def round_nifti(input_path: str) -> None:
    """
    Round pixel values in a NIfTI image and save with the same name.
    
    Args:
        input_path: Path to input NIfTI file
    """
    # Load the image
    img = nib.load(input_path)
    data = img.get_fdata()
    
    # Round the data
    rounded_data = np.round(data)
    
    # Create new NIfTI image with rounded data
    rounded_img = nib.Nifti1Image(rounded_data, img.affine, img.header)
    
    # Save the rounded image (overwrites original)
    nib.save(rounded_img, input_path) 