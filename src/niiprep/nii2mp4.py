import nibabel as nib
import numpy as np
import cv2
from pathlib import Path

def nii_to_mp4(
    input_path: str,
    output_path: str,
    dimension: int = 2,
    fps: int = 10,
    normalize: bool = True
) -> None:
    """
    Convert a NIfTI file to MP4 video along a specified dimension.
    
    Args:
        input_path (str): Path to input NIfTI file
        output_path (str): Path to save MP4 file
        dimension (int): Dimension to slice along (0: sagittal, 1: coronal, 2: axial)
        fps (int): Frames per second for the output video
        normalize (bool): Whether to normalize slice intensities
    """
    # Load NIfTI file
    img = nib.load(input_path)
    data = img.get_fdata()
    
    # Get slices based on dimension
    if dimension == 0:  # sagittal
        n_slices = data.shape[0]
        slices = [np.rot90(data[i, :, :]) for i in range(n_slices)]
    elif dimension == 1:  # coronal
        n_slices = data.shape[1]
        slices = [np.rot90(data[:, i, :]) for i in range(n_slices)]
    else:  # axial (dimension == 2)
        n_slices = data.shape[2]
        slices = [np.rot90(data[:, :, i]) for i in range(n_slices)]
    
    # Normalize and convert to uint8
    if normalize:
        slices = [(((s - s.min()) / (s.max() - s.min())) * 255).astype(np.uint8) for s in slices]
    else:
        slices = [s.astype(np.uint8) for s in slices]
    
    # Get dimensions for the video
    height, width = slices[0].shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    # Write slices to video
    for slice_data in slices:
        out.write(slice_data)
    
    # Release video writer
    out.release()