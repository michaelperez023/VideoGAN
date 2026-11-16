import numpy as np
import pytest
import torch
from PIL import Image

from videogan.video_io import count_frames, read_frames, tile, to_uint8, write_video


def test_to_uint8_maps_the_value_range():
    video = torch.tensor([-1.0, 0.0, 1.0]).view(1, 1, 1, 3).expand(3, 2, 1, 3)
    frames = to_uint8(video)
    assert frames.shape == (2, 1, 3, 3)
    assert frames.dtype == np.uint8
    assert frames[0, 0, :, 0].tolist() == [0, 128, 255]


def test_tile_places_videos_in_a_grid():
    grid = tile(-torch.ones(5, 3, 2, 4, 6), ncols=3, pad=2)
    assert grid.shape == (3, 2, 2 * 4 + 2, 3 * 6 + 2 * 2)
    assert grid[:, :, :4, :6].eq(-1).all()  # first video
    assert grid[:, :, 4:6].eq(1).all()  # white separator between the rows
    assert grid[:, :, 6:, 16:].eq(1).all()  # the empty sixth cell stays white


@pytest.mark.parametrize("suffix", [".mp4", ".gif"])
def test_write_video(tmp_path, suffix):
    frames = np.stack([np.full((64, 64, 3), 8 * t, np.uint8) for t in range(10)])
    path = tmp_path / f"clip{suffix}"
    write_video(path, frames, fps=25)
    if suffix == ".mp4":
        assert count_frames(path) == 10
        decoded = read_frames(path, range(10))
        assert decoded.shape == frames.shape
        assert np.abs(decoded.astype(int) - frames).mean() < 3
    else:
        with Image.open(path) as gif:
            assert gif.n_frames == 10
            assert gif.size == (64, 64)


def test_write_video_rejects_other_formats(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        write_video(tmp_path / "clip.avi", np.zeros((2, 8, 8, 3), np.uint8))


def test_mp4_frames_are_padded_to_even_sizes(tmp_path):
    write_video(tmp_path / "odd.mp4", np.zeros((4, 33, 35, 3), np.uint8))
    assert read_frames(tmp_path / "odd.mp4", [0]).shape == (1, 34, 36, 3)


def test_read_frames_resizes_and_repeats_the_last_frame(tmp_path):
    frames = np.stack([np.full((48, 80, 3), 20 * t, np.uint8) for t in range(6)])
    write_video(tmp_path / "clip.mp4", frames)
    out = read_frames(tmp_path / "clip.mp4", [5, 0, 7, 9], short_side=24)
    assert out.shape == (4, 24, 40, 3)
    assert abs(out[0].mean() - 100) < 4
    assert out[1].mean() < 4
    np.testing.assert_array_equal(out[2], out[0])  # past the end: the last frame
    np.testing.assert_array_equal(out[3], out[0])
