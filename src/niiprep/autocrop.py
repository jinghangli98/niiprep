import nibabel as nib
import numpy as np
import math


def apply_crop_bounds(input_path: str, output_path: str, bounds: dict):
    """Apply pre-computed autocrop bounds to an image without recomputing the bounding box."""
    img = nib.load(input_path)
    data = np.array(img.dataobj, dtype=np.float32)

    ox, oy, oz = bounds["origin"]
    tw, th, td = bounds["shape"]

    new_data = np.zeros((tw, th, td), dtype=data.dtype)

    src_x_s = max(0, ox);          src_x_e = min(data.shape[0], ox + tw)
    dst_x_s = max(0, -ox);         dst_x_e = dst_x_s + (src_x_e - src_x_s)

    src_y_s = max(0, oy);          src_y_e = min(data.shape[1], oy + th)
    dst_y_s = max(0, -oy);         dst_y_e = dst_y_s + (src_y_e - src_y_s)

    src_z_s = max(0, oz);          src_z_e = min(data.shape[2], oz + td)
    dst_z_s = max(0, -oz);         dst_z_e = dst_z_s + (src_z_e - src_z_s)

    new_data[dst_x_s:dst_x_e, dst_y_s:dst_y_e, dst_z_s:dst_z_e] = \
        data[src_x_s:src_x_e, src_y_s:src_y_e, src_z_s:src_z_e]

    new_affine = img.affine.copy()
    new_affine[:3, 3] += new_affine[:3, :3].dot(np.array([ox, oy, oz], dtype=float))

    nib.save(nib.Nifti1Image(new_data, new_affine, img.header), output_path)


def _otsu_threshold(data):
    """Compute Otsu's optimal threshold using only numpy."""
    hist, bin_edges = np.histogram(data.ravel(), bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = float(hist.sum())
    total_mean = float(np.dot(hist.astype(float), bin_centers))

    best_thresh, best_var = bin_centers[0], 0.0
    w_bg, s_bg = 0.0, 0.0

    for i in range(len(hist)):
        w_bg += hist[i]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        s_bg += hist[i] * bin_centers[i]
        m_bg = s_bg / w_bg
        m_fg = (total_mean - s_bg) / w_fg
        var = w_bg * w_fg * (m_bg - m_fg) ** 2
        if var > best_var:
            best_var = var
            best_thresh = bin_centers[i]

    return best_thresh


def _largest_run(profile):
    """Return (start, end) inclusive of the longest contiguous True run in a 1D boolean array.

    Used to ignore disconnected blobs (e.g. wrap-around MRI artifacts) that are separated
    from the main anatomy by a gap of empty slices.
    """
    best_start = best_end = best_len = 0
    cur_start = None
    for i, v in enumerate(profile):
        if v and cur_start is None:
            cur_start = i
        elif not v and cur_start is not None:
            length = i - cur_start
            if length > best_len:
                best_len, best_start, best_end = length, cur_start, i - 1
            cur_start = None
    if cur_start is not None:
        length = len(profile) - cur_start
        if length > best_len:
            best_len, best_start, best_end = length, cur_start, len(profile) - 1
    return (best_start, best_end) if best_len > 0 else (None, None)


def autocrop(input_path: str, output_path: str, n: int = None, target_shape: tuple = None):
    """
    Automatically crops the NIfTI image to the bounding box of detected meaningful content.

    Uses Otsu's intensity threshold to distinguish tissue from background. This is robust
    to non-zero background noise common in T1w and other MRI acquisitions.

    If target_shape is provided:
        Dimensions are set to target_shape.
        If n is also provided, dimensions are adjusted to be the largest multiple of n
        that is <= the target_shape dimension (floor behavior).

    If target_shape is NOT provided:
        If n is provided, pads so dimensions are the minimal multiple of n >= bbox.
        If n is NOT provided, crops tightly to the meaningful content.

    The content is centered in the output image.

    Args:
        input_path (str): Path to input NIfTI file.
        output_path (str): Path to output NIfTI file.
        n (int, optional): Integer multiple for output dimensions.
        target_shape (tuple, optional): Explicit target shape (x, y, z).
    """
    img = nib.load(input_path)
    data = img.get_fdata()
    data = data / np.max(data) * 255

    # For 4D images, collapse time axis for mask computation
    data_3d = np.mean(data, axis=3) if data.ndim == 4 else data

    # Otsu's method separates background noise from tissue reliably
    threshold = _otsu_threshold(data_3d)
    print(f"Intensity threshold (Otsu): {threshold:.2f}")

    mask = data_3d >= threshold

    # Bounding box: for each axis, find the largest contiguous block of active slices.
    # This ignores disconnected wrap-around artifacts (e.g. neck signal at the top of z
    # separated from the brain by ~20 empty slices) because they form a smaller run.
    profile_x = np.any(mask, axis=(1, 2))
    profile_y = np.any(mask, axis=(0, 2))
    profile_z = np.any(mask, axis=(0, 1))

    min_x, max_x = _largest_run(profile_x)
    min_y, max_y = _largest_run(profile_y)
    min_z, max_z = _largest_run(profile_z)

    if min_x is None or min_y is None or min_z is None:
        print("Warning: No meaningful content found. Output will be an empty image.")
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

    bbox_width  = max_x - min_x + 1
    bbox_height = max_y - min_y + 1
    bbox_depth  = max_z - min_z + 1

    print(f"Content BBox: {bbox_width}x{bbox_height}x{bbox_depth}")

    # Calculate target dimensions
    if target_shape is not None:
        t_w, t_h, t_d = target_shape
        if n is not None:
            t_w = (t_w // n) * n
            t_h = (t_h // n) * n
            t_d = (t_d // n) * n
            print(f"Adjusted target shape to multiples of {n}: {t_w}x{t_h}x{t_d}")
        target_width, target_height, target_depth = t_w, t_h, t_d
    elif n is not None:
        target_width  = math.ceil(bbox_width  / n) * n
        target_height = math.ceil(bbox_height / n) * n
        target_depth  = math.ceil(bbox_depth  / n) * n
    else:
        target_width, target_height, target_depth = bbox_width, bbox_height, bbox_depth

    print(f"Final Target Shape: {target_width}x{target_height}x{target_depth}")

    new_data = np.zeros((target_width, target_height, target_depth), dtype=data.dtype)

    offset_x = (target_width  - bbox_width)  // 2
    offset_y = (target_height - bbox_height) // 2
    offset_z = (target_depth  - bbox_depth)  // 2

    def get_ranges(offset, bbox_start, bbox_len, target_len):
        if offset >= 0:
            return (offset, offset + bbox_len), (bbox_start, bbox_start + bbox_len)
        else:
            return (0, target_len), (bbox_start - offset, bbox_start - offset + target_len)

    (d_x_s, d_x_e), (s_x_s, s_x_e) = get_ranges(offset_x, min_x, bbox_width,  target_width)
    (d_y_s, d_y_e), (s_y_s, s_y_e) = get_ranges(offset_y, min_y, bbox_height, target_height)
    (d_z_s, d_z_e), (s_z_s, s_z_e) = get_ranges(offset_z, min_z, bbox_depth,  target_depth)

    new_data[d_x_s:d_x_e, d_y_s:d_y_e, d_z_s:d_z_e] = \
        data[s_x_s:s_x_e, s_y_s:s_y_e, s_z_s:s_z_e]

    # Update affine: new[0,0,0] corresponds to old[min - offset] in each axis
    new_affine = img.affine.copy()
    origin_shift_indices = np.array([min_x - offset_x, min_y - offset_y, min_z - offset_z])
    new_affine[:3, 3] += new_affine[:3, :3].dot(origin_shift_indices)

    new_img = nib.Nifti1Image(new_data, new_affine, img.header)
    nib.save(new_img, output_path)

    return {
        "origin": [int(min_x - offset_x), int(min_y - offset_y), int(min_z - offset_z)],
        "shape": [int(target_width), int(target_height), int(target_depth)],
    }
