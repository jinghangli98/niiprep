import numpy as np
import nibabel as nib
from pathlib import Path

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpnd
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def otsu_mask(vol: np.ndarray, verbose: bool = True) -> np.ndarray:
    """
    Fast 3D mask using Otsu threshold + largest connected component.

    Thresholds the volume, then keeps only the largest connected component
    (the imaging object), discarding air/background noise.

    Parameters
    ----------
    vol     : 3D numpy array (X, Y, Z)
    verbose : print progress

    Returns
    -------
    mask : binary 3D numpy array, same shape as vol
    """
    from skimage.filters import threshold_otsu
    from scipy.ndimage import label

    threshold = threshold_otsu(vol)
    if verbose:
        print(f"Otsu threshold: {threshold:.4f}")

    binary = vol > threshold
    labeled, n_components = label(binary)
    if verbose:
        print(f"Connected components found: {n_components}")

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # ignore background label
    largest = component_sizes.argmax()
    mask = labeled == largest

    if verbose:
        print(f"Done. Foreground voxels: {mask.sum()} / {mask.size} "
              f"({100*mask.mean():.1f}%)")
    return mask


def ray_fill_cpu(vol: np.ndarray, threshold: float, fill_axis: int = 0) -> np.ndarray:
    vol_view = np.moveaxis(vol, fill_axis, 0)
    filled_view = np.zeros(vol_view.shape, dtype=bool)
    for z in range(vol_view.shape[2]):
        for y in range(vol_view.shape[1]):
            ray = vol_view[:, y, z]
            above = np.where(ray > threshold)[0]
            if len(above) >= 2:
                filled_view[above[0]:above[-1]+1, y, z] = True
    return np.moveaxis(filled_view, 0, fill_axis)


def ray_fill_gpu(vol_gpu, threshold: float, fill_axis: int = 0):
    """Ray fill on GPU using CuPy."""
    vol_view = cp.moveaxis(vol_gpu, fill_axis, 0)
    above = vol_view > threshold
    # cumulative sum from left and right to find first/last hit per ray
    cum_fwd = cp.cumsum(above, axis=0)
    cum_bwd = cp.cumsum(above[::-1, :, :], axis=0)[::-1, :, :]
    filled_view = (cum_fwd > 0) & (cum_bwd > 0)
    return cp.moveaxis(filled_view, 0, fill_axis)


def rotation_mask(
    vol: np.ndarray,
    n_angles: int = 36,
    threshold: float = None,
    threshold_scale: float = 1.0,
    closing_radius: int = 0,
    rot_axes: str = 'z',
    fill_holes: bool = True,
    blur_sigma: float = 0.0,
    edge_shrink: bool = False,
    edge_blur_sigma: float = 1.0,
    grad_threshold: float = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Build an object mask by rotating the volume around Z, ray-filling along X,
    back-rotating, and OR-accumulating across all angles.

    Uses GPU (CuPy) if available, otherwise falls back to CPU (SciPy).

    Parameters
    ----------
    vol             : 3D numpy array (X, Y, Z)
    n_angles        : number of rotation angles over 0-360 degrees
    threshold       : intensity threshold for air/tissue boundary.
                      If None, uses Otsu's method as a default.
    threshold_scale : scale factor applied to the threshold (default 1.0).
                      Values < 1.0 lower the threshold, useful for 7T data
                      where signal dropout in the neck region causes the
                      global Otsu threshold to be too conservative.
                      E.g. 0.5 uses half the Otsu threshold.
    closing_radius  : if > 0, apply binary morphological closing with a ball
                      structuring element of this radius after accumulating
                      the rotation mask. Helps bridge Z-axis signal dropout
                      gaps (e.g. bottom of neck at 7T). Default 0 (off).
    verbose         : print progress

    Returns
    -------
    mask : binary 3D numpy array, same shape as vol
    """
    from skimage.filters import threshold_otsu

    if threshold is None:
        threshold = threshold_otsu(vol)
        if verbose:
            print(f"Otsu threshold: {threshold:.4f}")

    threshold *= threshold_scale
    if threshold_scale != 1.0 and verbose:
        print(f"Scaled threshold ({threshold_scale}x): {threshold:.4f}")

    use_gpu = CUPY_AVAILABLE
    if verbose:
        print(f"Using {'GPU (CuPy)' if use_gpu else 'CPU (SciPy)'} for rotation")

    angles = np.linspace(0, 360, n_angles, endpoint=False)
    
    valid_axes = []
    for c in rot_axes.lower():
        if c in 'xyz' and c not in valid_axes:
            valid_axes.append(c)
    
    if not valid_axes:
        valid_axes = ['z']

    if use_gpu:
        vol_gpu = cp.asarray(vol.astype(np.float32))
        final_mask_gpu = cp.ones(vol.shape, dtype=bool)

        for ax in valid_axes:
            axis_mask_gpu = cp.zeros(vol.shape, dtype=bool)
            if ax == 'z':
                plane_axes = (0, 1); fill_axis = 0
            elif ax == 'y':
                plane_axes = (0, 2); fill_axis = 0
            elif ax == 'x':
                plane_axes = (1, 2); fill_axis = 1

            for i, angle in enumerate(angles):
                if verbose:
                    print(f"  Axis {ax.upper()} Angle {angle:6.1f}° ({i+1}/{n_angles})    ", end="\r")

                rotated = cpnd.rotate(vol_gpu, angle, axes=plane_axes, reshape=False, order=1)
                filled = ray_fill_gpu(rotated, threshold, fill_axis=fill_axis)
                back = cpnd.rotate(filled.astype(cp.float32), -angle, axes=plane_axes, reshape=False, order=1)
                axis_mask_gpu |= back > 0.5
            
            final_mask_gpu &= axis_mask_gpu

        mask = cp.asnumpy(final_mask_gpu)
    else:
        from scipy.ndimage import rotate

        final_mask = np.ones(vol.shape, dtype=bool)

        for ax in valid_axes:
            axis_mask = np.zeros(vol.shape, dtype=bool)
            if ax == 'z':
                plane_axes = (0, 1); fill_axis = 0
            elif ax == 'y':
                plane_axes = (0, 2); fill_axis = 0
            elif ax == 'x':
                plane_axes = (1, 2); fill_axis = 1

            for i, angle in enumerate(angles):
                if verbose:
                    print(f"  Axis {ax.upper()} Angle {angle:6.1f}° ({i+1}/{n_angles})    ", end="\r")

                rotated = rotate(vol, angle, axes=plane_axes, reshape=False, order=1)
                filled = ray_fill_cpu(rotated, threshold, fill_axis=fill_axis)
                back = rotate(filled.astype(np.float32), -angle, axes=plane_axes, reshape=False, order=1)
                axis_mask |= back > 0.5
            
            final_mask &= axis_mask
            
        mask = final_mask

    if closing_radius > 0:
        if verbose:
            print(f"Applying morphological closing (radius={closing_radius})...")
        from scipy.ndimage import binary_closing
        from skimage.morphology import ball
        struct = ball(closing_radius)
        mask = binary_closing(mask, structure=struct)

    if edge_shrink:
        if verbose:
            print("Applying morphological edge shrink to conform to skin boundary...")
        from skimage.filters import threshold_otsu
        
        # We need vol_gpu here if use_gpu is True, but if we didn't use GPU earlier, it won't exist.
        # But we create vol_gpu at line 137 if use_gpu.
        if use_gpu:
            vol_sm = cpnd.gaussian_filter(vol_gpu, sigma=edge_blur_sigma)
            grad = cpnd.gaussian_gradient_magnitude(vol_sm, sigma=1.0)
            grad_numpy = cp.asnumpy(grad[cp.asarray(mask)])
        else:
            from scipy.ndimage import gaussian_filter, gaussian_gradient_magnitude
            vol_sm = gaussian_filter(vol.astype(np.float32), sigma=edge_blur_sigma)
            grad = gaussian_gradient_magnitude(vol_sm, sigma=1.0)
            grad_numpy = grad[mask]
            
        if grad_threshold is None:
            if grad_numpy.size > 0:
                gt = threshold_otsu(grad_numpy) * 0.5
            else:
                gt = 0.0
            if verbose:
                print(f"  Auto gradient threshold: {gt:.4f}")
        else:
            gt = grad_threshold
            if verbose:
                print(f"  Using gradient threshold: {gt:.4f}")

        if use_gpu:
            for_loop_mask = cp.asarray(mask)
            for it in range(100): # max 100 iterations
                eroded = cpnd.binary_erosion(for_loop_mask)
                border = for_loop_mask ^ eroded
                remove = border & (grad < gt) & (vol_sm < threshold)
                if not cp.any(remove):
                    break
                for_loop_mask ^= remove
            mask = cp.asnumpy(for_loop_mask)
        else:
            from scipy.ndimage import binary_erosion
            for_loop_mask = mask.copy()
            for it in range(100):
                eroded = binary_erosion(for_loop_mask)
                border = for_loop_mask ^ eroded
                remove = border & (grad < gt) & (vol_sm < threshold)
                if not np.any(remove):
                    break
                for_loop_mask ^= remove
            mask = for_loop_mask

    if fill_holes:
        if verbose:
            print("Cleaning up final mask (largest component & fill holes)...")
        from scipy.ndimage import label, binary_fill_holes
        labeled, n_components = label(mask)
        if n_components > 0:
            component_sizes = np.bincount(labeled.ravel())
            component_sizes[0] = 0  # ignore background label
            largest = component_sizes.argmax()
            mask = labeled == largest
        mask = binary_fill_holes(mask)

    if blur_sigma > 0.0:
        if verbose:
            print(f"Smoothing mask (Gaussian blur sigma={blur_sigma})...")
        from scipy.ndimage import gaussian_filter
        mask = gaussian_filter(mask.astype(np.float32), sigma=blur_sigma) > 0.5

    if verbose:
        print(f"\nDone. Foreground voxels: {mask.sum()} / {mask.size} "
              f"({100*mask.mean():.1f}%)")
    return mask


def main(
    input_path: str,
    output_path: str = None,
    use_rotation: bool = True,
    rot_axes: str = 'z',
    n_angles: int = 36,
    threshold: float = None,
    threshold_scale: float = 1.0,
    closing_radius: int = 0,
    fill_holes: bool = True,
    blur_sigma: float = 0.0,
    edge_shrink: bool = False,
    edge_blur_sigma: float = 1.0,
    grad_threshold: float = None,
):
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / (input_path.name.replace(".nii", "_mask.nii"))

    print(f"Loading: {input_path}")
    nii = nib.load(input_path)
    vol = nii.get_fdata()
    print(f"Volume shape: {vol.shape}, dtype: {vol.dtype}")

    if use_rotation:
        print(f"Running rotation mask (rot_axes={rot_axes}, n_angles={n_angles})...")
        mask = rotation_mask(
            vol,
            n_angles=n_angles,
            threshold=threshold,
            threshold_scale=threshold_scale,
            closing_radius=closing_radius,
            rot_axes=rot_axes,
            fill_holes=fill_holes,
            blur_sigma=blur_sigma,
            edge_shrink=edge_shrink,
            edge_blur_sigma=edge_blur_sigma,
            grad_threshold=grad_threshold,
        )
    else:
        print("Running fast Otsu mask...")
        mask = otsu_mask(vol)

    out_nii = nib.Nifti1Image(mask.astype(np.uint8), nii.affine, nii.header)
    nib.save(out_nii, output_path)
    print(f"Saved mask: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MRI object mask")
    parser.add_argument("input", help="Input NIfTI file (.nii or .nii.gz)")
    parser.add_argument("--output", default=None, help="Output mask path (default: <input>_mask.nii)")
    parser.add_argument("--no-rotation", action="store_true", help="Use fast Otsu-only mask instead of rotation-based masking")
    parser.add_argument("--n_angles", type=int, default=36, help="Number of rotation angles (default: 36)")
    parser.add_argument("--threshold", type=float, default=None, help="Intensity threshold (default: Otsu)")
    parser.add_argument("--threshold_scale", type=float, default=1.0,
                        help="Scale factor applied to the threshold (default: 1.0). "
                             "Use values < 1.0 (e.g. 0.5) for 7T data with low signal "
                             "in the neck region to include those voxels.")
    parser.add_argument("--axes", type=str, default="z", help="Axes to rotate around (e.g. z, or xyz for all 3)")
    parser.add_argument("--closing_radius", type=int, default=0,
                        help="Radius of morphological closing applied after rotation mask (default: 0 = off). "
                             "Helps bridge Z-axis signal dropout gaps at the bottom of the neck.")
    parser.add_argument("--no-fill-holes", action="store_true", help="Disable filling holes and extracting largest connected component in final mask")
    parser.add_argument("--blur_sigma", type=float, default=0.0, help="Gaussian blur sigma for smoothing the mask (default: 0 = off)")
    parser.add_argument('--edge-shrink', action='store_true', help='Shrink the mask inwards until hitting a strong gradient spike (true skin boundary).')
    parser.add_argument('--edge-blur-sigma', type=float, default=1.0, help='Blur applied before gradient calculation for shrinking. Default: 1.0.')
    parser.add_argument('--grad-threshold', type=float, default=None, help='Gradient threshold for shrink stop condition. Default is auto Otsu.')
    args = parser.parse_args()
    main(args.input, args.output, not args.no_rotation, args.axes, args.n_angles,
         args.threshold, args.threshold_scale, args.closing_radius, not args.no_fill_holes, args.blur_sigma,
         args.edge_shrink, args.edge_blur_sigma, args.grad_threshold)
