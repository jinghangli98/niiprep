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


def ray_fill_cpu(vol: np.ndarray, threshold: float) -> np.ndarray:
    filled = np.zeros(vol.shape, dtype=bool)
    for z in range(vol.shape[2]):
        for y in range(vol.shape[1]):
            ray = vol[:, y, z]
            above = np.where(ray > threshold)[0]
            if len(above) >= 2:
                filled[above[0]:above[-1]+1, y, z] = True
    return filled


def ray_fill_gpu(vol_gpu, threshold: float):
    """Ray fill on GPU using CuPy."""
    above = vol_gpu > threshold
    # cumulative sum from left and right to find first/last hit per ray
    cum_fwd = cp.cumsum(above, axis=0)
    cum_bwd = cp.cumsum(above[::-1, :, :], axis=0)[::-1, :, :]
    filled = (cum_fwd > 0) & (cum_bwd > 0)
    return filled


def rotation_mask(
    vol: np.ndarray,
    n_angles: int = 36,
    threshold: float = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Build an object mask by rotating the volume around Z, ray-filling along X,
    back-rotating, and OR-accumulating across all angles.

    Uses GPU (CuPy) if available, otherwise falls back to CPU (SciPy).

    Parameters
    ----------
    vol       : 3D numpy array (X, Y, Z)
    n_angles  : number of rotation angles over 0-360 degrees
    threshold : intensity threshold for air/tissue boundary.
                If None, uses Otsu's method as a default.
    verbose   : print progress

    Returns
    -------
    mask : binary 3D numpy array, same shape as vol
    """
    from skimage.filters import threshold_otsu

    if threshold is None:
        threshold = threshold_otsu(vol)
        if verbose:
            print(f"Otsu threshold: {threshold:.4f}")

    use_gpu = CUPY_AVAILABLE
    if verbose:
        print(f"Using {'GPU (CuPy)' if use_gpu else 'CPU (SciPy)'} for rotation")

    angles = np.linspace(0, 360, n_angles, endpoint=False)

    if use_gpu:
        vol_gpu = cp.asarray(vol.astype(np.float32))
        mask_gpu = cp.zeros(vol.shape, dtype=bool)

        for i, angle in enumerate(angles):
            if verbose:
                print(f"  Angle {angle:6.1f}° ({i+1}/{n_angles})", end="\r")

            rotated = cpnd.rotate(vol_gpu, angle, axes=(0, 1), reshape=False, order=1)
            filled = ray_fill_gpu(rotated, threshold)
            back = cpnd.rotate(filled.astype(cp.float32), -angle, axes=(0, 1), reshape=False, order=1)
            mask_gpu |= back > 0.5

        mask = cp.asnumpy(mask_gpu)
    else:
        from scipy.ndimage import rotate

        mask = np.zeros(vol.shape, dtype=bool)

        for i, angle in enumerate(angles):
            if verbose:
                print(f"  Angle {angle:6.1f}° ({i+1}/{n_angles})", end="\r")

            rotated = rotate(vol, angle, axes=(0, 1), reshape=False, order=1)
            filled = ray_fill_cpu(rotated, threshold)
            back = rotate(filled.astype(np.float32), -angle, axes=(0, 1), reshape=False, order=1)
            mask |= back > 0.5

    if verbose:
        print(f"\nDone. Foreground voxels: {mask.sum()} / {mask.size} "
              f"({100*mask.mean():.1f}%)")
    return mask


def main(
    input_path: str,
    output_path: str = None,
    use_rotation: bool = True,
    n_angles: int = 36,
    threshold: float = None,
):
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / (input_path.name.replace(".nii", "_mask.nii"))

    print(f"Loading: {input_path}")
    nii = nib.load(input_path)
    vol = nii.get_fdata()
    print(f"Volume shape: {vol.shape}, dtype: {vol.dtype}")

    if use_rotation:
        print(f"Running rotation mask (n_angles={n_angles})...")
        mask = rotation_mask(vol, n_angles=n_angles, threshold=threshold)
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
    args = parser.parse_args()
    main(args.input, args.output, not args.no_rotation, args.n_angles, args.threshold)
