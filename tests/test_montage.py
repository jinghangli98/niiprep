import numpy as np
import imageio
import pytest

from niiprep.montage import montage, tile_frames


def _write_gif(path, n_frames, shape, fps=10, value=None):
    """Write a synthetic grayscale GIF of ``n_frames`` frames of ``shape``."""
    frames = []
    for i in range(n_frames):
        v = (i * 255 // max(n_frames - 1, 1)) if value is None else value
        f = np.full(shape, v, dtype=np.uint8)
        # Pillow merges identical consecutive GIF frames; keep them distinct
        f[-1, -1] = i
        frames.append(f)
    imageio.mimsave(str(path), frames, duration=1000.0 / fps, loop=0)
    return str(path)


def _read_frames(path):
    return [np.asarray(f) for f in imageio.mimread(str(path))]


def _to_gray(frame):
    return frame if frame.ndim == 2 else frame[..., 0]


# --------------------------------------------------------------------- #
# tile_frames
# --------------------------------------------------------------------- #

def test_tile_frames_lays_out_grid_with_given_columns():
    a = np.full((4, 6), 10, dtype=np.uint8)
    b = np.full((4, 6), 20, dtype=np.uint8)
    c = np.full((4, 6), 30, dtype=np.uint8)
    out = tile_frames([a, b, c], cols=2)
    assert out.shape == (8, 12)
    assert (out[:4, :6] == 10).all()
    assert (out[:4, 6:] == 20).all()
    assert (out[4:, :6] == 30).all()
    # empty cell stays black
    assert (out[4:, 6:] == 0).all()


def test_tile_frames_pads_smaller_tiles_to_largest():
    big = np.full((6, 8), 50, dtype=np.uint8)
    small = np.full((3, 4), 90, dtype=np.uint8)
    out = tile_frames([big, small], cols=2)
    assert out.shape == (6, 16)
    assert (out[:, :8] == 50).all()
    assert (out[:3, 8:12] == 90).all()
    assert (out[3:, 8:] == 0).all()
    assert (out[:3, 12:] == 0).all()


def test_tile_frames_default_cols_is_ceil_sqrt():
    tiles = [np.zeros((2, 2), dtype=np.uint8) for _ in range(5)]
    out = tile_frames(tiles)
    # 5 tiles -> 3 columns, 2 rows
    assert out.shape == (4, 6)


# --------------------------------------------------------------------- #
# montage (file level)
# --------------------------------------------------------------------- #

def test_montage_gif_has_grid_shape_and_frame_count(tmp_path):
    ins = [_write_gif(tmp_path / f"in{i}.gif", 5, (8, 10)) for i in range(4)]
    out = tmp_path / "out.gif"
    montage(ins, str(out))
    frames = _read_frames(out)
    assert len(frames) == 5
    assert _to_gray(frames[0]).shape == (16, 20)


def test_montage_truncates_to_shortest_and_warns(tmp_path):
    short = _write_gif(tmp_path / "short.gif", 3, (8, 8))
    long_ = _write_gif(tmp_path / "long.gif", 7, (8, 8))
    out = tmp_path / "out.gif"
    with pytest.warns(UserWarning, match="long.gif"):
        montage([short, long_], str(out))
    assert len(_read_frames(out)) == 3


def test_montage_tiles_are_in_input_order(tmp_path):
    a = _write_gif(tmp_path / "a.gif", 2, (4, 4), value=40)
    b = _write_gif(tmp_path / "b.gif", 2, (4, 4), value=200)
    out = tmp_path / "out.gif"
    montage([a, b], str(out), cols=2)
    f = _to_gray(_read_frames(out)[0])
    assert abs(int(np.median(f[:, :4])) - 40) <= 2
    assert abs(int(np.median(f[:, 4:])) - 200) <= 2


def test_montage_writes_mp4_from_gif_inputs(tmp_path):
    ins = [_write_gif(tmp_path / f"in{i}.gif", 4, (16, 16)) for i in range(2)]
    out = tmp_path / "out.mp4"
    montage(ins, str(out), fps=5)
    reader = imageio.get_reader(str(out))
    n = sum(1 for _ in reader)
    reader.close()
    assert n == 4


def test_montage_accepts_mp4_inputs(tmp_path):
    gif = _write_gif(tmp_path / "in.gif", 4, (16, 16))
    mp4 = tmp_path / "in.mp4"
    montage([gif], str(mp4), fps=5)
    out = tmp_path / "out.gif"
    montage([str(mp4), gif], str(out), cols=2)
    frames = _read_frames(out)
    assert len(frames) == 4
    assert _to_gray(frames[0]).shape == (16, 32)


def test_montage_labels_change_pixels_in_tile_corner(tmp_path):
    a = _write_gif(tmp_path / "a.gif", 2, (40, 80), value=0)
    out_plain = tmp_path / "plain.gif"
    out_lab = tmp_path / "lab.gif"
    montage([a], str(out_plain))
    montage([a], str(out_lab), labels=["sub01"])
    plain = _to_gray(_read_frames(out_plain)[0])
    lab = _to_gray(_read_frames(out_lab)[0])
    assert (plain == 0).all()
    # label is drawn in the top-left region only
    assert lab[:20, :60].max() > 0
    assert (lab[20:, :] == 0).all()


def test_montage_rejects_label_count_mismatch(tmp_path):
    a = _write_gif(tmp_path / "a.gif", 2, (4, 4))
    with pytest.raises(ValueError, match="labels"):
        montage([a], str(tmp_path / "out.gif"), labels=["x", "y"])


def test_montage_rejects_unknown_output_extension(tmp_path):
    a = _write_gif(tmp_path / "a.gif", 2, (4, 4))
    with pytest.raises(ValueError, match="gif"):
        montage([a], str(tmp_path / "out.avi"))


def test_montage_mp4_pads_odd_dimensions_to_even(tmp_path):
    a = _write_gif(tmp_path / "a.gif", 3, (33, 47), value=200)
    out = tmp_path / "out.mp4"
    montage([a], str(out), fps=5)
    reader = imageio.get_reader(str(out))
    frames = [np.asarray(f) for f in reader]
    reader.close()
    assert len(frames) == 3
    assert frames[0].shape[:2] == (34, 48)
    # original content intact, padding row/col black
    assert abs(int(_to_gray(frames[0])[:33, :47].mean()) - 200) <= 3
    assert _to_gray(frames[0])[33, :].max() < 20
