import math
import os
import warnings

import numpy as np


def _to_gray_uint8(frame):
    """Convert an imageio frame (gray or RGB/RGBA, any dtype) to 2D uint8."""
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[..., :3]
        if arr.shape[-1] == 3:
            arr = arr.astype(np.float32)
            arr = (0.299 * arr[..., 0] + 0.587 * arr[..., 1]
                   + 0.114 * arr[..., 2])
        else:
            arr = arr[..., 0]
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    return arr


def _read_cine(path):
    """Return (list of 2D uint8 frames, fps or None) for a GIF/MP4 file."""
    import imageio
    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data()
        frames = [_to_gray_uint8(f) for f in reader]
    finally:
        reader.close()
    fps = meta.get('fps')
    if fps is None and meta.get('duration'):
        try:
            fps = 1000.0 / float(meta['duration'])
        except (TypeError, ValueError, ZeroDivisionError):
            fps = None
    return frames, fps


def tile_frames(tiles, cols=None):
    """Tile 2D uint8 frames into a single grid image.

    Smaller tiles are zero-padded (top-left aligned) to the largest tile
    size, and empty grid cells stay black.

    Args:
        tiles: Sequence of 2D uint8 arrays.
        cols: Number of columns. Defaults to ``ceil(sqrt(len(tiles)))``.

    Returns:
        2D uint8 array of shape ``(rows * H, cols * W)``.
    """
    n = len(tiles)
    if n == 0:
        raise ValueError("tile_frames needs at least one tile")
    if cols is None:
        cols = int(math.ceil(math.sqrt(n)))
    cols = max(1, int(cols))
    rows = int(math.ceil(n / cols))
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    out = np.zeros((rows * h, cols * w), dtype=np.uint8)
    for idx, t in enumerate(tiles):
        r, c = divmod(idx, cols)
        out[r * h:r * h + t.shape[0], c * w:c * w + t.shape[1]] = t
    return out


def _draw_label(frame, text):
    import cv2
    out = np.ascontiguousarray(frame)
    scale = max(0.3, min(1.0, frame.shape[1] / 400.0))
    thickness = 1 if scale < 0.7 else 2
    (_, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                 scale, thickness)
    cv2.putText(out, text, (3, th + 3), cv2.FONT_HERSHEY_SIMPLEX,
                scale, 255, thickness, cv2.LINE_AA)
    return out


def _write_cine(frames, output_path, fps):
    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.gif':
        import imageio
        imageio.mimsave(output_path, frames,
                        duration=1000.0 / float(fps), loop=0)
    elif ext == '.mp4':
        import cv2
        # mp4v needs even dimensions; odd sizes decode wrongly.
        height, width = frames[0].shape
        pad_h, pad_w = height % 2, width % 2
        if pad_h or pad_w:
            frames = [np.pad(f, ((0, pad_h), (0, pad_w))) for f in frames]
            height, width = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height),
                              isColor=False)
        for frame in frames:
            out.write(frame)
        out.release()
    else:
        raise ValueError(
            f"output must end with .gif or .mp4, got '{ext or output_path}'")


def montage(input_paths, output_path, cols=None, labels=None, fps=None):
    """Tile several MIP cines (GIF/MP4) into one synced montage cine.

    All inputs are truncated to the shortest frame count so every tile
    advances in lockstep. Tiles are laid out row-major in input order.

    Args:
        input_paths: Paths to input .gif / .mp4 files.
        output_path: Path to the output .gif or .mp4 file.
        cols: Grid columns. Defaults to ``ceil(sqrt(n_inputs))``.
        labels: Optional list of one text label per input, burned into
            the top-left corner of each tile.
        fps: Playback frame rate. Defaults to the first input's rate
            when it can be read, otherwise 10.
    """
    input_paths = list(input_paths)
    if not input_paths:
        raise ValueError("montage needs at least one input")
    if labels is not None and len(labels) != len(input_paths):
        raise ValueError(
            f"expected {len(input_paths)} labels, got {len(labels)}")
    ext = os.path.splitext(output_path)[1].lower()
    if ext not in ('.gif', '.mp4'):
        raise ValueError(
            f"output must end with .gif or .mp4, got '{ext or output_path}'")

    cines = []
    first_fps = None
    for p in input_paths:
        frames, in_fps = _read_cine(p)
        if not frames:
            raise ValueError(f"no frames found in {p}")
        if first_fps is None:
            first_fps = in_fps
        cines.append(frames)

    n_frames = min(len(c) for c in cines)
    cut = [os.path.basename(p) for p, c in zip(input_paths, cines)
           if len(c) > n_frames]
    if cut:
        warnings.warn(
            f"truncating to shortest cine ({n_frames} frames); cut: "
            + ", ".join(cut), UserWarning)

    if labels is not None:
        cines = [[_draw_label(f, lab) for f in c]
                 for c, lab in zip(cines, labels)]

    out_frames = [tile_frames([c[i] for c in cines], cols=cols)
                  for i in range(n_frames)]

    if fps is None:
        fps = first_fps if first_fps else 10
    _write_cine(out_frames, output_path, fps)
    print(f"Montage: {len(input_paths)} inputs, {n_frames} frames, "
          f"{out_frames[0].shape[1]}x{out_frames[0].shape[0]} -> {output_path}")
