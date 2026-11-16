"""Datasets: golf scenes for training the GAN, and UCF-101 clips for fine-tuning.

:class:`GolfStrips` reads the golf data the same way as train_golf.py in yhjo09/videogan-pytorch
(https://github.com/yhjo09/videogan-pytorch), except that it resizes frames with Pillow instead
of scikit-image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .models import FRAME_SIZE, NUM_FRAMES
from .video_io import count_frames, read_frames


class GolfStrips(Dataset):
    """Clips from the VideoGAN golf dataset.

    Each JPEG stores one clip as square frames stacked vertically (128 x 128 frames in the released
    data), and ``index`` lists the JPEGs relative to ``frames_dir``. An item is the first
    ``num_frames`` frames, resized to ``size`` x ``size`` and scaled to [-1, 1], as a (3, T, H, W)
    tensor. Strips with fewer frames are returned as ``None``; :func:`collate_skip_none` leaves
    them out of the batch.
    """

    def __init__(
        self,
        root: str | Path,
        index: str = "golf.txt",
        frames_dir: str = "frames-stable-many",
        num_frames: int = NUM_FRAMES,
        size: int = FRAME_SIZE,
    ) -> None:
        self.frames_dir = Path(root) / frames_dir
        index_path = Path(root) / index
        with index_path.open() as fh:
            self.files = [line.strip() for line in fh if line.strip()]
        if not self.files:
            raise ValueError(f"{index_path} does not list any files")
        self.num_frames = num_frames
        self.size = size

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor | None:
        with Image.open(self.frames_dir / self.files[idx]) as img:
            strip = img.convert("RGB")
        width, height = strip.size
        if height // width < self.num_frames:
            return None
        frames = [
            strip.crop((0, t * width, width, (t + 1) * width)).resize(
                (self.size, self.size), Image.Resampling.BILINEAR
            )
            for t in range(self.num_frames)
        ]
        video = torch.from_numpy(np.stack(frames))  # (T, H, W, 3) uint8
        return video.permute(3, 0, 1, 2).float().div(127.5).sub(1.0)


def collate_skip_none(batch: list[torch.Tensor | None]) -> torch.Tensor | None:
    """Stack the clips of a batch, leaving out the ``None`` items of strips that were too short."""
    clips = [clip for clip in batch if clip is not None]
    return torch.stack(clips) if clips else None


def read_class_index(split_dir: str | Path) -> list[str]:
    """Read ``classInd.txt`` (one "<index> <name>" per line, from 1) as names ordered by index."""
    path = Path(split_dir) / "classInd.txt"
    entries = sorted(
        (int(index), name)
        for index, name in (line.split() for line in path.read_text().splitlines() if line.strip())
    )
    if [index for index, _ in entries] != list(range(1, len(entries) + 1)):
        raise ValueError(f"{path}: class indices must run from 1 to the number of classes")
    return [name for _, name in entries]


def read_split(split_dir: str | Path, train: bool, fold: int = 1) -> list[str]:
    """Read the video paths ("Class/v_Class_gXX_cYY.avi") of a train or test list."""
    path = Path(split_dir) / f"{'train' if train else 'test'}list{fold:02d}.txt"
    return [line.split()[0] for line in path.read_text().splitlines() if line.strip()]


class UCF101Clips(Dataset):
    """UCF-101 videos as fixed-length clips, read through an official-format split.

    ``split_dir`` holds ``classInd.txt`` and the ``trainlistNN.txt``/``testlistNN.txt`` files;
    a folder that lists only some classes (such as ``splits/kayaking-yoyo``) gives a smaller
    task. Labels are 0-based positions in ``classInd.txt``. Videos are looked up as
    ``root/Class/file.avi`` (the layout of the official archive) or ``root/file.avi``.

    Training items are one random clip per video with a random crop and horizontal flip, as
    ``(video, label)`` with ``video`` of shape (3, T, H, W). Test items stack
    ``clips_per_video`` evenly spaced, center-cropped clips (from the first frame to the last;
    the middle clip if there is only one) into a (K, 3, T, H, W) tensor, so predictions can be
    averaged per video. Frames are resized so that their shorter side is ``size``. Only the
    video stream is decoded, and videos shorter than ``num_frames`` repeat their last frame.
    """

    def __init__(
        self,
        root: str | Path,
        split_dir: str | Path,
        train: bool,
        fold: int = 1,
        clips_per_video: int = 5,
        num_frames: int = NUM_FRAMES,
        size: int = FRAME_SIZE,
    ) -> None:
        self.classes = read_class_index(split_dir)
        class_to_idx = {name: i for i, name in enumerate(self.classes)}
        root = Path(root)
        self.samples: list[tuple[Path, int]] = []
        missing = []
        for rel in read_split(split_dir, train, fold):
            class_name = rel.split("/")[0]
            if class_name not in class_to_idx:
                raise ValueError(f"{rel}: class {class_name!r} is not listed in classInd.txt")
            path = root / rel
            if not path.exists():
                path = root / Path(rel).name
            if path.exists():
                self.samples.append((path, class_to_idx[class_name]))
            else:
                missing.append(rel)
        if missing:
            total = len(missing) + len(self.samples)
            raise FileNotFoundError(
                f"{len(missing)} of the {total} videos in the split are not under {root} "
                f"(for example {missing[0]})"
            )
        self.train = train
        self.clips_per_video = clips_per_video
        self.num_frames = num_frames
        self.size = size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        last_start = max(count_frames(path) - self.num_frames, 0)
        if self.train:
            starts = [int(torch.randint(last_start + 1, ()))]
        elif self.clips_per_video == 1:
            starts = [last_start // 2]
        else:
            starts = torch.linspace(0, last_start, self.clips_per_video).round().long().tolist()
        indices = [start + t for start in starts for t in range(self.num_frames)]
        frames = torch.from_numpy(read_frames(path, indices, short_side=self.size))
        clips = frames.view(len(starts), self.num_frames, *frames.shape[1:])  # (K, T, H, W, 3)
        clips = self._crop(clips.permute(0, 4, 1, 2, 3).float().div(127.5).sub(1.0))
        return (clips[0] if self.train else clips), label

    def _crop(self, clips: torch.Tensor) -> torch.Tensor:
        height, width = clips.shape[-2:]
        if self.train:
            top = int(torch.randint(height - self.size + 1, ()))
            left = int(torch.randint(width - self.size + 1, ()))
        else:
            top, left = (height - self.size) // 2, (width - self.size) // 2
        clips = clips[..., top : top + self.size, left : left + self.size]
        if self.train and torch.rand(()) < 0.5:
            clips = clips.flip(-1)
        return clips
