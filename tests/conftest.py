"""Fixtures that write tiny synthetic datasets to disk."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest
import torch
from PIL import Image

from videogan.models import Discriminator, Generator


def make_strip(path: Path, num_frames: int, frame_size: int = 32) -> None:
    """Write a golf-style JPEG strip whose frame t is a flat gray of level 7 * t (t < 37)."""
    frames = [np.full((frame_size, frame_size, 3), 7 * t, np.uint8) for t in range(num_frames)]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(frames)).save(path, quality=95)


def make_video(path: Path, num_frames: int, audio_samples: int = 0) -> None:
    """Write an 80 x 60 MPEG-4 AVI whose frame t is a flat gray of level 5 * t (t < 52).

    With ``audio_samples``, the file also gets a stereo PCM track of that many samples.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        # All streams have to exist before the first packet is written.
        video = container.add_stream("mpeg4", rate=25)
        video.width, video.height, video.pix_fmt = 80, 60, "yuv420p"
        audio = container.add_stream("pcm_s16le", rate=8000) if audio_samples else None
        if audio is not None:
            audio.layout = "stereo"
            frame = av.AudioFrame.from_ndarray(
                np.zeros((1, 2 * audio_samples), np.int16), format="s16", layout="stereo"
            )
            frame.sample_rate, frame.pts = 8000, 0
            container.mux(audio.encode(frame))
            container.mux(audio.encode())
        for t in range(num_frames):
            image = np.full((60, 80, 3), 5 * t, np.uint8)
            container.mux(video.encode(av.VideoFrame.from_ndarray(image, format="rgb24")))
        container.mux(video.encode())


@pytest.fixture
def golf_root(tmp_path: Path) -> Path:
    """Four 34-frame strips and one 20-frame strip, listed in golf.txt as in the real dataset."""
    root = tmp_path / "golf"
    names = [f"videos/{i}/{i}.mp4/0000.jpg" for i in range(5)]
    for i, name in enumerate(names):
        make_strip(root / "frames-stable-many" / name, num_frames=20 if i == 4 else 34)
    (root / "golf.txt").write_text("\n".join(names) + "\n")
    return root


@pytest.fixture
def ucf_root(tmp_path: Path) -> tuple[Path, Path]:
    """Two classes in the UCF-101 layout, with split files in the official format.

    Some videos have audio tracks of different lengths and some have none, and one test video
    is shorter than a 32-frame clip.
    """
    data, splits = tmp_path / "UCF-101", tmp_path / "splits"
    videos = {
        "ClassA/v_ClassA_g01_c01.avi": (40, 8000),
        "ClassA/v_ClassA_g01_c02.avi": (36, 0),
        "ClassB/v_ClassB_g01_c01.avi": (45, 0),
        "ClassB/v_ClassB_g01_c02.avi": (40, 5000),
        "ClassA/v_ClassA_g02_c01.avi": (20, 3000),
        "ClassB/v_ClassB_g02_c01.avi": (38, 0),
    }
    for rel, (num_frames, audio_samples) in videos.items():
        make_video(data / rel, num_frames, audio_samples)
    splits.mkdir()
    rels = list(videos)
    (splits / "classInd.txt").write_text("1 ClassA\r\n2 ClassB\r\n")
    train = [f"{rel} {1 if 'ClassA' in rel else 2}" for rel in rels[:4]]
    (splits / "trainlist01.txt").write_text("\r\n".join(train) + "\r\n")
    (splits / "testlist01.txt").write_text("\r\n".join(rels[4:]) + "\r\n")
    return data, splits


@pytest.fixture
def gan_checkpoint(tmp_path: Path) -> Path:
    """A checkpoint with untrained generator and discriminator weights."""
    torch.manual_seed(0)
    path = tmp_path / "gan.pt"
    state = {"generator": Generator().state_dict(), "discriminator": Discriminator().state_dict()}
    torch.save(state, path)
    return path
