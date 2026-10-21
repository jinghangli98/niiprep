import math
import os
import json
import numpy as np
import nibabel as nib
from numpy.lib.stride_tricks import sliding_window_view
from .autocrop import _otsu_threshold


def unpatchify(patches: np.ndarray, imsize: tuple) -> np.ndarray:
    """Merge a (n0, n1, n2, p0, p1, p2) grid of patches into a 3D volume.

    Port of ``patchify.unpatchify`` (3D case), vendored because every
    patchify release conflicts with numpy>=2 (0.2.3 pins numpy<2 and
    0.2.2 pins scikit-image<0.18, which no longer builds). The patch
    visiting order is kept identical so overlaps resolve the same way.
    Unlike the original, the index always advances, so a step larger
    than the patch size no longer loops forever.
    """
    if patches.ndim != 6:
        raise ValueError(f"Expected a 6D patch grid, got shape {patches.shape}")

    i_h, i_w, i_c = imsize
    image = np.zeros(imsize, dtype=patches.dtype)
    n_h, n_w, n_c, p_h, p_w, p_c = patches.shape

    steps = []
    for i_s, n, p in ((i_h, n_h, p_h), (i_w, n_w, p_w), (i_c, n_c, p_c)):
        s = 0 if n <= 1 else (i_s - p) / (n - 1)
        if int(s) != s:
            raise ValueError(
                f"Cannot reconstruct {n} x {p}px patches into size {i_s}: "
                f"non-integer step {s}")
        steps.append(int(s))
    s_h, s_w, s_c = steps

    i, j, k = 0, 0, 0
    while True:
        i_o, j_o, k_o = i * s_h, j * s_w, k * s_c
        image[i_o:i_o + p_h, j_o:j_o + p_w, k_o:k_o + p_c] = patches[i, j, k]

        if k < n_c - 1:
            k = min(max(k + 1, (k_o + p_c) // s_c), n_c - 1)
        elif j < n_w - 1:
            j = min(max(j + 1, (j_o + p_w) // s_w), n_w - 1)
            k = 0
        elif i < n_h - 1:
            i = min(max(i + 1, (i_o + p_h) // s_h), n_h - 1)
            j = 0
            k = 0
        else:
            break

    return image


def _compat_dim(dim: int, patch_size: int, step: int) -> int:
    """Return the smallest dim' >= dim where (dim' - patch_size) % step == 0."""
    if dim <= patch_size:
        return patch_size
    k = math.ceil((dim - patch_size) / step)
    return patch_size + k * step


def patchify_nii(
    input_path: str,
    output_dir: str,
    patch_size: tuple,
    step=None,
    pad: bool = True,
    norm: bool = False,
    skip: bool = False,
    min_nonzero_frac: float = 0.01,
    min_intensity_range: float = 0.0,
    foreground_threshold: float = None,
    forced_skip_indices: list = None,
    crop_bounds: dict = None,
):
    """Split a 3D NIfTI image into patches saved as individual .nii.gz files.

    Parameters
    ----------
    input_path : path to input NIfTI file
    output_dir : directory where patches and metadata are saved
    patch_size : (px, py, pz) patch dimensions in voxels
    step : stride between patches. int (same on all axes) or sequence of 3 ints
           (sx, sy, sz). Default: min(patch_size) on all axes.
    pad : zero-pad to compatible dimensions if needed (default True)
    norm : normalize image intensities to [0, 1] before patchifying (default False)
    skip : skip patches that do not pass the foreground filter (default False, all patches saved)
    min_nonzero_frac : (skip=True) min fraction of foreground voxels to keep a patch (default 0.01)
    min_intensity_range : (skip=True) min max-min intensity to keep a patch (default 0.0, disabled)
    foreground_threshold : (skip=True) intensity above which a voxel counts as foreground;
                           default None auto-computes Otsu's threshold on the full image

    Returns
    -------
    grid_shape : (n0, n1, n2) number of patches along each axis
    """
    img = nib.load(input_path)
    data = np.array(img.dataobj, dtype=np.float32)

    if data.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {data.shape}")

    if norm:
        v_min, v_max = float(data.min()), float(data.max())
        if v_max > v_min:
            data = (data - v_min) / (v_max - v_min)
        else:
            data = np.zeros_like(data)
        print(f"Normalized image intensities from [{v_min:.4g}, {v_max:.4g}] to [0, 1]")

    if skip and foreground_threshold is None:
        foreground_threshold = float(_otsu_threshold(data))
        print(f"Auto foreground threshold (Otsu): {foreground_threshold:.4g}")

    if step is None:
        step = (min(patch_size),) * 3
    elif isinstance(step, (int, np.integer)):
        step = (int(step),) * 3
    else:
        step = tuple(int(s) for s in step)
        if len(step) != 3:
            raise ValueError(f"step must be int or sequence of 3 ints, got {len(step)} values")

    original_shape = list(data.shape)
    padded_shape = [_compat_dim(d, ps, s) for d, ps, s in zip(data.shape, patch_size, step)]

    if padded_shape != original_shape:
        if not pad:
            raise ValueError(
                f"Image shape {original_shape} is incompatible with patch_size={patch_size} "
                f"and step={step}. Compatible shape would be {padded_shape}. "
                f"Use pad=True or resize the image first."
            )
        pad_widths = [(0, p - o) for o, p in zip(original_shape, padded_shape)]
        data = np.pad(data, pad_widths, mode="constant", constant_values=0)
        print(f"Padded image from {original_shape} to {padded_shape} for compatibility")

    windows = sliding_window_view(data, tuple(patch_size))
    patches = windows[::step[0], ::step[1], ::step[2]]
    # patches.shape: (n0, n1, n2, p0, p1, p2)
    grid_shape = patches.shape[:3]
    affine = img.affine

    os.makedirs(output_dir, exist_ok=True)

    saved, skipped = 0, 0
    skipped_indices = []
    forced_skip_set = {tuple(idx) for idx in (forced_skip_indices or [])}

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                patch_data = patches[i, j, k]

                if (i, j, k) in forced_skip_set:
                    skipped_indices.append([i, j, k])
                    skipped += 1
                    continue

                if skip:
                    nonzero_frac = float(np.sum(patch_data > foreground_threshold)) / patch_data.size
                    intensity_range = float(patch_data.max() - patch_data.min())
                    if nonzero_frac < min_nonzero_frac or intensity_range <= min_intensity_range:
                        skipped_indices.append([i, j, k])
                        skipped += 1
                        continue

                offset = np.array([i * step[0], j * step[1], k * step[2], 1.0])
                patch_affine = affine.copy()
                patch_affine[:3, 3] = (affine @ offset)[:3]

                patch_img = nib.Nifti1Image(patch_data, patch_affine, img.header)
                fname = f"patch_{i:04d}_{j:04d}_{k:04d}.nii.gz"
                nib.save(patch_img, os.path.join(output_dir, fname))
                saved += 1

    meta = {
        "original_shape": original_shape,
        "padded_shape": padded_shape,
        "patch_size": list(patch_size),
        "step": list(step),
        "grid_shape": list(grid_shape),
        "affine": affine.tolist(),
        "skipped_indices": skipped_indices,
        "crop_bounds": crop_bounds,
    }
    with open(os.path.join(output_dir, "patches_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total = grid_shape[0] * grid_shape[1] * grid_shape[2]
    print(f"Saved {saved}/{total} patches (grid {grid_shape[0]}x{grid_shape[1]}x{grid_shape[2]}, "
          f"{skipped} skipped) to {output_dir}")
    return grid_shape


def unpatchify_nii(input_dir: str, output_path: str):
    """Reconstruct a NIfTI image from patches created by patchify_nii.

    Parameters
    ----------
    input_dir : directory containing patches and patches_meta.json
    output_path : path for the reconstructed NIfTI file
    """
    meta_path = os.path.join(input_dir, "patches_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"patches_meta.json not found in {input_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    grid_shape = meta["grid_shape"]
    patch_size = meta["patch_size"]
    original_shape = tuple(meta["original_shape"])
    affine = np.array(meta["affine"])

    skipped = {tuple(idx) for idx in meta.get("skipped_indices", [])}

    patches = np.zeros(grid_shape + patch_size, dtype=np.float32)

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                if (i, j, k) in skipped:
                    continue  # leave as zeros
                fname = f"patch_{i:04d}_{j:04d}_{k:04d}.nii.gz"
                fpath = os.path.join(input_dir, fname)
                patches[i, j, k] = np.array(nib.load(fpath).dataobj, dtype=np.float32)

    padded_shape = tuple(meta.get("padded_shape", original_shape))
    reconstructed = unpatchify(patches, padded_shape)

    # Crop back to the original (pre-padding) shape
    s = original_shape
    reconstructed = reconstructed[:s[0], :s[1], :s[2]]

    out_img = nib.Nifti1Image(reconstructed.astype(np.float32), affine)
    nib.save(out_img, output_path)
    print(f"Reconstructed image saved to {output_path}")
