import nibabel as nib
import numpy as np
import math

def autocrop(input_path: str, output_path: str, n: int = None, target_shape: tuple = None):
    """
    Automatically crops the NIfTI image to the bounding box of detected meaningful content (edges).
    
    If target_shape is provided:
        Dimensions are set to target_shape.
        If n is also provided, dimensions are adjusted to be the largest multiple of n 
        that is less than or equal to the target_shape dimension (floor behavior).
    
    If target_shape is NOT provided:
        If n is provided, pads the result so that the dimensions are the minimal integer multiple of n
        that covers the meaningful content.
        If n is NOT provided, crops tightly to the meaningful content.

    The content is centered in the output image.

    Args:
        input_path (str): Path to input NIfTI file.
        output_path (str): Path to output NIfTI file.
        n (int, optional): The integer multiple for the output dimensions. 
        target_shape (tuple, optional): Explicit target shape (x, y, z).
    """
    # Load the image
    img = nib.load(input_path)
    data = img.get_fdata()
    data = data/np.max(data) * 255

    # Calculate gradient magnitude
    grads = np.gradient(data)
    grad_mag = np.sqrt(sum(g**2 for g in grads))
    
    # We use the gradient magnitude to determine meaningful content
    max_grad = np.max(grad_mag)
    threshold = 0.1 * max_grad
    
    print(f"Max gradient: {max_grad}, Gradient Threshold: {threshold}")

    # Project to 1D profiles of max gradient magnitude
    axes_x = tuple(i for i in range(data.ndim) if i != 0)
    profile_x = np.max(grad_mag, axis=axes_x) 
    valid_x = np.where(profile_x >= threshold)[0]
    
    axes_y = tuple(i for i in range(data.ndim) if i != 1)
    profile_y = np.max(grad_mag, axis=axes_y)
    valid_y = np.where(profile_y >= threshold)[0]
    
    axes_z = tuple(i for i in range(data.ndim) if i != 2)
    profile_z = np.max(grad_mag, axis=axes_z)
    valid_z = np.where(profile_z >= threshold)[0]

    if len(valid_x) == 0 or len(valid_y) == 0 or len(valid_z) == 0:
        print("Warning: No meaningful gradients found. Output will be an empty image.")
        if target_shape:
             final_shape = target_shape
        elif n:
             final_shape = (n, n, n)
        else:
             final_shape = data.shape[:3]

        if data.ndim == 4 and len(final_shape) == 3:
             final_shape = (*final_shape, data.shape[3])

        new_data = np.zeros(final_shape, dtype=data.dtype)
        new_img = nib.Nifti1Image(new_data, img.affine, img.header)
        nib.save(new_img, output_path)
        return

    min_x, max_x = valid_x[0], valid_x[-1]
    min_y, max_y = valid_y[0], valid_y[-1]
    min_z, max_z = valid_z[0], valid_z[-1]

    # Dimensions of the bounding box
    bbox_width = max_x - min_x + 1
    bbox_height = max_y - min_y + 1
    bbox_depth = max_z - min_z + 1
    
    print(f"Content BBox: {bbox_width}x{bbox_height}x{bbox_depth}")

    # Calculate target dimensions
    if target_shape is not None:
        # User specified shape
        t_w, t_h, t_d = target_shape
        if n is not None:
             # Adjust to be compliant with n (smaller compliant size)
             # integer division // n * n floors to nearest multiple
             t_w = (t_w // n) * n
             t_h = (t_h // n) * n
             t_d = (t_d // n) * n
             print(f"Adjusted target shape to multiples of {n}: {t_w}x{t_h}x{t_d}")
        
        target_width, target_height, target_depth = t_w, t_h, t_d
        
    elif n is not None:
        # No strict shape, just multiple of N (covering the content)
        target_width = math.ceil(bbox_width / n) * n
        target_height = math.ceil(bbox_height / n) * n
        target_depth = math.ceil(bbox_depth / n) * n
    else:
        # Tight crop
        target_width = bbox_width
        target_height = bbox_height
        target_depth = bbox_depth
    
    print(f"Final Target Shape: {target_width}x{target_height}x{target_depth}")
    
    # Create new data array
    new_data = np.zeros((target_width, target_height, target_depth), dtype=data.dtype)

    # Calculate offsets to center the bounding box
    # If target > bbox, we pad (offset > 0).
    # If target < bbox, we crop (offset < 0).
    
    offset_x = (target_width - bbox_width) // 2
    offset_y = (target_height - bbox_height) // 2
    offset_z = (target_depth - bbox_depth) // 2

    # Helper to calculate ranges
    def get_ranges(offset, bbox_start, bbox_len, target_len):
        if offset >= 0:
            # Padding: box fits inside target
            # Dst: [offset, offset + bbox_len]
            # Src: [bbox_start, bbox_start + bbox_len]
            return (offset, offset + bbox_len), (bbox_start, bbox_start + bbox_len)
        else:
            # Cropping: box is larger than target
            # We take the center part of the box that fits in target
            # Dst: [0, target_len]
            # Src: [bbox_start + (-offset), bbox_start + (-offset) + target_len]
            return (0, target_len), (bbox_start - offset, bbox_start - offset + target_len)

    (d_x_s, d_x_e), (s_x_s, s_x_e) = get_ranges(offset_x, min_x, bbox_width, target_width)
    (d_y_s, d_y_e), (s_y_s, s_y_e) = get_ranges(offset_y, min_y, bbox_height, target_height)
    (d_z_s, d_z_e), (s_z_s, s_z_e) = get_ranges(offset_z, min_z, bbox_depth, target_depth)

    # Ensure indices are within bounds (double check)
    # Src indices should be within [0, data_shape]
    # Dst indices within [0, target_shape]
    # With strict math above they should be, but let's clamp if necessary or rely on slicing.
    # Actually, slicing handles out-of-bounds gracefully by truncating, 
    # but we want to be precise.
    
    new_data[d_x_s:d_x_e, d_y_s:d_y_e, d_z_s:d_z_e] = \
        data[s_x_s:s_x_e, s_y_s:s_y_e, s_z_s:s_z_e]

    # Update affine
    # New origin W' = A * (new_center_index_in_old_space) + T?
    # Actually, let's look at the shift.
    # The pixel at new_data[0,0,0] corresponds to what pixel in old_data?
    # If offset_x >= 0 (padding), new[0] is padding. new[offset] comes from old[min_x].
    #   So new[0] corresponds to old[min_x - offset].
    # If offset_x < 0 (cropping), new[0] comes from old[min_x - offset].
    #   So new[0] corresponds to old[min_x - offset].
    # Formula is consistent: Source Index = Dest Index + min_x - offset.
    # So Dest Index 0 -> Source Index (min_x - offset).
    
    new_affine = img.affine.copy()
    origin_shift_indices = np.array([min_x - offset_x, min_y - offset_y, min_z - offset_z])
    
    translated_shift = new_affine[:3, :3].dot(origin_shift_indices)
    new_affine[:3, 3] += translated_shift

    # create new nifti image
    new_img = nib.Nifti1Image(new_data, new_affine, img.header)
    
    # Save
    nib.save(new_img, output_path)
