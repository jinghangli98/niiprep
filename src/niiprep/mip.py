import nibabel as nib
import numpy as np


def mip(input_path: str, output_path: str, axis: int = 2,
        slab: int = None) -> None:
    """
    Maximum Intensity Projection (MIP) of a NIfTI image.

    Useful for TOF MRA to render the vascular tree from the brightest
    voxels along the projection axis.

    Args:
        input_path: Path to input NIfTI file.
        axis: Axis to project along (0: sagittal, 1: coronal, 2: axial).
        slab: Sliding-slab thickness in voxels. If None, a single full
            projection is produced (the projection axis collapses to
            length 1). If set, a same-shape volume is produced where
            each voxel holds the max over a centered slab of that
            thickness along the projection axis.
        output_path: Path to save the MIP NIfTI file.
    """
    img = nib.load(input_path)
    data = img.get_fdata()

    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")

    if slab is None:
        mip_data = np.max(data, axis=axis, keepdims=True)
    else:
        if slab < 1:
            raise ValueError(f"slab must be >= 1, got {slab}")
        n = data.shape[axis]
        moved = np.moveaxis(data, axis, 0)
        out = np.empty_like(moved)
        half = slab // 2
        for k in range(n):
            lo = max(0, k - half)
            hi = min(n, k - half + slab)
            out[k] = np.max(moved[lo:hi], axis=0)
        mip_data = np.moveaxis(out, 0, axis)

    mip_img = nib.Nifti1Image(
        mip_data.astype(data.dtype), img.affine, img.header
    )
    nib.save(mip_img, output_path)


def _get_rotate_backend(use_gpu=True):
    """Return ``(xp, rotate, asnumpy, on_gpu)`` for the available backend.

    Falls back to NumPy/SciPy on the CPU when CuPy or a working CUDA
    device is unavailable, so the same code path runs everywhere.
    """
    if use_gpu:
        try:
            import cupy as cp
            from cupyx.scipy.ndimage import rotate as cp_rotate
            if cp.cuda.runtime.getDeviceCount() > 0:
                return cp, cp_rotate, cp.asnumpy, True
        except Exception:
            pass
    from scipy.ndimage import rotate as np_rotate
    return np, np_rotate, (lambda a: a), False


def _slab_project(vol, axis, slab):
    """Max-project ``vol`` along ``axis``, optionally over a centered slab.

    Works on both NumPy and CuPy arrays (uses the array's own ``max``).
    """
    n = vol.shape[axis]
    if slab is not None and slab < n:
        half = slab // 2
        lo = max(0, n // 2 - half)
        hi = min(n, lo + slab)
        sl = [slice(None)] * vol.ndim
        sl[axis] = slice(lo, hi)
        vol = vol[tuple(sl)]
    return vol.max(axis=axis)


def _pad_to(frame, height, width):
    """Center-pad a 2D frame to (height, width) with zeros."""
    h, w = frame.shape
    top = (height - h) // 2
    left = (width - w) // 2
    return np.pad(frame,
                  ((top, height - h - top), (left, width - w - left)))


def _rotating_mip_frames(data, axis, spin_axis, frames, slab, use_gpu=True):
    """Return uint8 2D frames of a MIP rotated 360 deg around spin_axis.

    Frames are projected over an optional centered slab and padded to a
    common canvas so the rotated content is never clipped. Uses CuPy on
    the GPU when available, otherwise SciPy on the CPU.
    """
    if axis == spin_axis:
        raise ValueError("projection --axis and --spin-axis must differ")

    xp, rotate, asnumpy, on_gpu = _get_rotate_backend(use_gpu)
    print(f"Rotating MIP backend: {'GPU (CuPy)' if on_gpu else 'CPU (SciPy)'}")

    # The rotation plane is spanned by the two axes that are not the
    # spin axis; the projection axis is one of them so the view changes
    # as the volume spins.
    plane = tuple(a for a in (0, 1, 2) if a != spin_axis)

    vmin, vmax = float(data.min()), float(data.max())
    angles = np.linspace(0.0, 360.0, frames, endpoint=True)

    # Move the volume to the device once; only small 2D frames are
    # copied back per angle, keeping host<->device transfer minimal.
    vol = xp.asarray(data) if on_gpu else data

    try:
        from tqdm import tqdm
        angle_iter = tqdm(angles, desc="Rotating MIP", unit="frame")
    except ImportError:
        angle_iter = angles

    raw = []
    for angle in angle_iter:
        # reshape=True grows the volume to fit the rotated content so
        # nothing is clipped to the original bounding box.
        # cval=0 keeps the zero-fill below the data range so the
        # rotated background maps to black after clipping.
        rot = rotate(vol, float(angle), axes=plane, reshape=True,
                     order=1, cval=0.0)
        proj = _slab_project(rot, axis, slab)
        if vmax > vmin:
            proj = (proj - vmin) / (vmax - vmin) * 255.0
        else:
            proj = xp.zeros_like(proj)
        # Clip before the uint8 cast: the rotation zero-fill projects
        # below vmin, giving small negatives that would otherwise wrap
        # to 255 and show as white bars at the rotated silhouette.
        proj = xp.clip(proj, 0.0, 255.0)
        raw.append(asnumpy(xp.rot90(proj)))

    max_h = max(f.shape[0] for f in raw)
    max_w = max(f.shape[1] for f in raw)
    return [_pad_to(f, max_h, max_w).astype(np.uint8) for f in raw]


def rotating_mip(input_path: str, output_path: str, axis: int = 1,
                 spin_axis: int = 2, frames: int = 36, fps: int = 10,
                 fmt: str = "gif", slab: int = None,
                 use_gpu: bool = True) -> None:
    """
    Render a rotating Maximum Intensity Projection cine.

    The volume is spun 360 deg around ``spin_axis`` and a MIP is
    projected along ``axis`` at each step, producing the classic
    spinning TOF MRA vascular tree.

    Args:
        input_path: Path to input NIfTI file.
        output_path: Path to save the .gif or .mp4 file.
        axis: Projection axis (0: sagittal, 1: coronal, 2: axial).
        spin_axis: Axis the volume rotates around (default 2, the
            cranio-caudal axis for TOF head data).
        frames: Number of frames over a full 360 deg rotation.
        fps: Playback frames per second.
        fmt: Output format, "gif" or "mp4".
        slab: Centered slab thickness in voxels to project through at
            each angle. If None, the full depth is projected.
        use_gpu: Use the CuPy GPU backend when a CUDA device is
            available; falls back to the CPU otherwise.
    """
    img = nib.load(input_path)
    data = img.get_fdata()

    frame_list = _rotating_mip_frames(data, axis, spin_axis, frames, slab,
                                      use_gpu=use_gpu)

    if fmt == "gif":
        import imageio
        imageio.mimsave(output_path, frame_list, fps=fps, loop=0)
    elif fmt == "mp4":
        import cv2
        height, width = frame_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path, fourcc, fps, (width, height), isColor=False
        )
        for frame in frame_list:
            out.write(frame)
        out.release()
    else:
        raise ValueError(f"fmt must be 'gif' or 'mp4', got {fmt}")
