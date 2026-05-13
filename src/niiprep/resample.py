import torchio as tio
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom, gaussian_filter


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


def resample_seg(
    input_path: str,
    output_path: str,
    target_spacing: tuple = (1.0, 1.0, 1.0),
    sigma: float = 0.7,
    order: int = 1,
) -> None:
    """
    Resample a discrete-label segmentation NIfTI with anti-aliased boundaries.

    Each label is one-hot encoded, smoothed with a Gaussian and resampled with
    linear/cubic interpolation, then the label with the highest probability is
    chosen per output voxel (argmax). This preserves the original label set
    (no fractional labels) while removing the stair-step artifacts produced by
    pure nearest-neighbor resampling.

    Args:
        input_path: Path to input segmentation NIfTI.
        output_path: Path to save resampled segmentation.
        target_spacing: Target voxel spacing in mm (x, y, z).
        sigma: Gaussian smoothing in source voxels applied to each one-hot
            channel before interpolation. 0 disables smoothing (pure linear
            argmax). Larger values give smoother boundaries.
        order: Spline order for the interpolation step (1=linear, 3=cubic).
    """
    img = nib.load(input_path)
    data = np.asarray(img.get_fdata()).astype(np.int32)
    src_spacing = np.array(img.header.get_zooms()[:3], dtype=float)
    tgt_spacing = np.array(target_spacing, dtype=float)
    zoom_factors = src_spacing / tgt_spacing

    labels = np.unique(data)
    out_shape = tuple(int(round(s * z)) for s, z in zip(data.shape, zoom_factors))

    best_prob = np.full(out_shape, -np.inf, dtype=np.float32)
    best_label = np.zeros(out_shape, dtype=np.int32)

    for lbl in labels:
        mask = (data == lbl).astype(np.float32)
        if sigma > 0:
            mask = gaussian_filter(mask, sigma=sigma)
        prob = zoom(mask, zoom_factors, order=order, mode="nearest")
        if prob.shape != out_shape:
            # rounding can disagree by 1 voxel; crop/pad to match
            prob = _fit_shape(prob, out_shape)
        if lbl == 0:
            # never let background overwrite a real label at numerical ties
            better = prob > best_prob + 1e-6
        else:
            better = prob > best_prob
        np.putmask(best_label, better, lbl)
        np.putmask(best_prob, better, prob)

    new_affine = img.affine.copy()
    for i in range(3):
        col = new_affine[:3, i]
        n = np.linalg.norm(col)
        if n > 0:
            new_affine[:3, i] = col / n * tgt_spacing[i]

    out = nib.Nifti1Image(best_label.astype(np.int32), new_affine, img.header)
    out.header.set_zooms(tuple(tgt_spacing) + tuple(img.header.get_zooms()[3:]))
    out.set_data_dtype(np.int32)
    nib.save(out, output_path)


def _fit_shape(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Crop or zero-pad arr so its shape matches target_shape."""
    out = np.zeros(target_shape, dtype=arr.dtype)
    slices_src, slices_dst = [], []
    for s_src, s_dst in zip(arr.shape, target_shape):
        n = min(s_src, s_dst)
        slices_src.append(slice(0, n))
        slices_dst.append(slice(0, n))
    out[tuple(slices_dst)] = arr[tuple(slices_src)]
    return out
