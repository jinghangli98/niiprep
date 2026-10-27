import shutil
import subprocess

import nibabel as nib
import numpy as np
import cv2


def _write_ffmpeg(frames, output_path, fps, crf):
    """Encode grayscale uint8 frames with H.264 via the ffmpeg CLI.

    Returns True on success, False if ffmpeg is missing or fails to run.
    """
    exe = shutil.which("ffmpeg")
    if exe is None:
        return False
    height, width = frames[0].shape
    cmd = [
        exe, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-crf", str(crf), "-preset", "slow",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        output_path,
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for f in frames:
            proc.stdin.write(np.ascontiguousarray(f).tobytes())
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    if proc.wait() != 0:
        err = proc.stderr.read().decode(errors="replace").strip().splitlines()
        print(f"ffmpeg failed ({err[0] if err else 'unknown error'}); "
              "falling back to OpenCV.")
        return False
    return True


def _write_opencv(frames, output_path, fps):
    """Encode with OpenCV, preferring H.264 (avc1) over the grainier mp4v."""
    height, width = frames[0].shape
    for codec in ("avc1", "mp4v"):
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), fps,
                              (width, height), isColor=False)
        if out.isOpened():
            break
    else:
        raise RuntimeError("OpenCV could not open a video writer for " + output_path)
    if codec == "mp4v":
        print("H.264 unavailable; using mp4v, which is lower quality.")
    for f in frames:
        out.write(f)
    out.release()


def nii_to_mp4(
    input_path: str,
    output_path: str,
    dimension: int = 2,
    fps: int = 10,
    normalize: bool = True,
    norm_mode: str = "global",
    clip_percentile: float = 100.0,
    size: int = 1024,
    crf: int = 17,
    start: int = 0,
) -> None:
    """
    Convert a NIfTI file to MP4 video along a specified dimension.
    
    Args:
        input_path (str): Path to input NIfTI file
        output_path (str): Path to save MP4 file
        dimension (int): Dimension to slice along (0: sagittal, 1: coronal, 2: axial)
        fps (int): Frames per second for the output video
        normalize (bool): Whether to normalize intensities to 0-255
        norm_mode (str): "global" scales every slice with one volume-wide range
            (no flicker); "slice" scales each slice to its own min/max
        clip_percentile (float): Upper percentile used as the max in global mode
            (e.g. 99.5 to ignore bright outliers); 100 uses the true max
        size (int): Length in pixels of the longer frame edge. Frames are
            resampled with bicubic interpolation and voxel spacing is honored so
            anisotropic data keeps its true aspect ratio. 0 keeps native voxel grid.
        crf (int): H.264 quality when a working ffmpeg is on PATH (lower =
            better, 17 is visually lossless). Ignored by the OpenCV fallback.
        start (int): 0-based index of the first slice to include. Global
            normalization still uses the whole volume.
    """
    # Load NIfTI file
    img = nib.load(input_path)
    data = img.get_fdata()
    zooms = img.header.get_zooms()[:3]
    
    # Get slices based on dimension; (row_spacing, col_spacing) after rot90
    if dimension == 0:  # sagittal
        n_slices = data.shape[0]
        slices = [np.rot90(data[i, :, :]) for i in range(n_slices)]
        spacing = (zooms[2], zooms[1])
    elif dimension == 1:  # coronal
        n_slices = data.shape[1]
        slices = [np.rot90(data[:, i, :]) for i in range(n_slices)]
        spacing = (zooms[2], zooms[0])
    else:  # axial (dimension == 2)
        n_slices = data.shape[2]
        slices = [np.rot90(data[:, :, i]) for i in range(n_slices)]
        spacing = (zooms[1], zooms[0])
    
    if not 0 <= start < n_slices:
        raise ValueError(f"start must be in [0, {n_slices - 1}] for dimension "
                         f"{dimension}, got {start}")
    slices = slices[start:]

    # Normalize and convert to uint8
    def _scale(s, lo, hi):
        if hi <= lo:
            return np.zeros(s.shape, dtype=np.uint8)
        return (np.clip((s - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)

    if not normalize:
        slices = [np.clip(s, 0, 255).astype(np.uint8) for s in slices]
    elif norm_mode == "slice":
        slices = [_scale(s, s.min(), s.max()) for s in slices]
    elif norm_mode == "global":
        lo = float(data.min())
        hi = float(np.percentile(data, clip_percentile)) if clip_percentile < 100 else float(data.max())
        slices = [_scale(s, lo, hi) for s in slices]
    else:
        raise ValueError(f"norm_mode must be 'global' or 'slice', got {norm_mode!r}")
    
    # Upsample to the target size, keeping the physical aspect ratio
    if size and size > 0:
        rows, cols = slices[0].shape
        phys_h, phys_w = rows * spacing[0], cols * spacing[1]
        scale = size / max(phys_h, phys_w)
        # Even dimensions are required by yuv420p H.264
        height = max(2, int(round(phys_h * scale / 2)) * 2)
        width = max(2, int(round(phys_w * scale / 2)) * 2)
        if (height, width) != (rows, cols):
            interp = cv2.INTER_CUBIC if height >= rows else cv2.INTER_AREA
            slices = [cv2.resize(s, (width, height), interpolation=interp) for s in slices]
    else:
        # Native grid: pad to even dimensions for H.264
        rows, cols = slices[0].shape
        slices = [np.pad(s, ((0, rows % 2), (0, cols % 2))) for s in slices]

    if not _write_ffmpeg(slices, output_path, fps, crf):
        _write_opencv(slices, output_path, fps)
