import shutil

import pytest
import torch
from torch.utils.data import DataLoader

from videogan.data import (
    GolfStrips,
    UCF101Clips,
    collate_skip_none,
    read_class_index,
    read_split,
)


def _gray_levels(frames: torch.Tensor) -> torch.Tensor:
    """Mean gray level (0-255) of each frame of a (3, T, H, W) clip in [-1, 1]."""
    return (frames.mean(dim=(0, 2, 3)) + 1) * 127.5


def test_golf_strips_items(golf_root):
    dataset = GolfStrips(golf_root)
    assert len(dataset) == 5
    clip = dataset[0]
    assert clip.shape == (3, 32, 64, 64)
    assert clip.dtype == torch.float32
    assert clip.min() >= -1
    assert clip.max() <= 1
    # The first 32 of the 34 frames, in order; frame t was drawn with gray level 7 * t.
    torch.testing.assert_close(_gray_levels(clip), torch.arange(32) * 7.0, atol=3, rtol=0)


def test_short_golf_strips_are_left_out_of_batches(golf_root):
    dataset = GolfStrips(golf_root)
    assert dataset[4] is None
    assert collate_skip_none([dataset[0], dataset[4], dataset[1]]).shape == (2, 3, 32, 64, 64)
    assert collate_skip_none([None, None]) is None


def test_golf_strips_need_a_non_empty_index(tmp_path):
    (tmp_path / "golf.txt").write_text("\n")
    with pytest.raises(ValueError, match="does not list any files"):
        GolfStrips(tmp_path)


def test_read_class_index_and_split(ucf_root):
    _, splits = ucf_root
    assert read_class_index(splits) == ["ClassA", "ClassB"]
    assert read_split(splits, train=True)[0] == "ClassA/v_ClassA_g01_c01.avi"
    assert read_split(splits, train=False) == [
        "ClassA/v_ClassA_g02_c01.avi",
        "ClassB/v_ClassB_g02_c01.avi",
    ]


def test_class_index_is_ordered_by_number(tmp_path):
    (tmp_path / "classInd.txt").write_text("2 YoYo\n1 Kayaking\n")
    assert read_class_index(tmp_path) == ["Kayaking", "YoYo"]
    (tmp_path / "classInd.txt").write_text("1 Kayaking\n3 YoYo\n")
    with pytest.raises(ValueError, match="class indices"):
        read_class_index(tmp_path)


def test_ucf101_training_items(ucf_root):
    data, splits = ucf_root
    dataset = UCF101Clips(data, splits, train=True)
    assert len(dataset) == 4
    assert dataset.classes == ["ClassA", "ClassB"]
    video, label = dataset[0]
    assert video.shape == (3, 32, 64, 64)
    assert video.min() >= -1
    assert video.max() <= 1
    assert [dataset[i][1] for i in range(4)] == [0, 0, 1, 1]
    assert label == 0


def test_ucf101_test_items_stack_evenly_spaced_clips(ucf_root):
    data, splits = ucf_root
    clips, label = UCF101Clips(data, splits, train=False, clips_per_video=3)[1]
    assert clips.shape == (3, 3, 32, 64, 64)
    assert label == 1
    # The video has 38 frames, so the clips start at frames 0, 3 and 6 (gray 0, 15 and 30).
    first_frames = torch.stack([_gray_levels(clip)[0] for clip in clips])
    torch.testing.assert_close(first_frames, torch.tensor([0.0, 15.0, 30.0]), atol=4, rtol=0)


def test_ucf101_single_test_clip_comes_from_the_middle(ucf_root):
    data, splits = ucf_root
    clips, _ = UCF101Clips(data, splits, train=False, clips_per_video=1)[1]
    assert clips.shape == (1, 3, 32, 64, 64)
    # 38 frames: the middle clip starts at frame 3 (gray 15).
    torch.testing.assert_close(_gray_levels(clips[0])[0], torch.tensor(15.0), atol=4, rtol=0)


def test_ucf101_short_videos_repeat_their_last_frame(ucf_root):
    data, splits = ucf_root
    clips, label = UCF101Clips(data, splits, train=False, clips_per_video=2)[0]
    assert clips.shape == (2, 3, 32, 64, 64)
    assert label == 0
    # The video has 20 frames: frames 20-31 of the clip repeat frame 19.
    last = clips[0, :, 19:20]
    torch.testing.assert_close(clips[0, :, 19:], last.expand(-1, 13, -1, -1))


def test_ucf101_videos_can_sit_in_a_flat_folder(ucf_root, tmp_path):
    data, splits = ucf_root
    flat = tmp_path / "flat"
    flat.mkdir()
    for video in data.glob("*/*.avi"):
        shutil.copy(video, flat / video.name)
    assert len(UCF101Clips(flat, splits, train=True)) == 4


def test_ucf101_reports_missing_videos(ucf_root):
    data, splits = ucf_root
    (data / "ClassA" / "v_ClassA_g01_c02.avi").unlink()
    with pytest.raises(FileNotFoundError, match="1 of the 4 videos"):
        UCF101Clips(data, splits, train=True)


def test_ucf101_batches_mix_videos_with_and_without_audio(ucf_root):
    """The 2022 loader failed on batches that mixed clips with and without sound."""
    data, splits = ucf_root
    loader = DataLoader(UCF101Clips(data, splits, train=True), batch_size=4)
    videos, labels = next(iter(loader))
    assert videos.shape == (4, 3, 32, 64, 64)
    assert labels.tolist() == [0, 0, 1, 1]
