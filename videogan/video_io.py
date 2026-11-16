"""Video and image I/O: PyAV for decoding and MP4 files, Pillow for GIFs and still images."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image


def to_uint8(video: torch.Tensor) -> np.ndarray:
    """Convert a (3, T, H, W) tensor in [-1, 1] to (T, H, W, 3) uint8 frames."""
    frames = video.detach().float().cpu().clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
    return np.ascontiguousarray(frames.permute(1, 2, 3, 0).numpy())


def tile(videos: torch.Tensor, ncols: int, pad: int = 2) -> torch.Tensor:
    """Arrange (N, C, T, H, W) videos in a grid with ``ncols`` columns, separated by white lines.

    Returns a (C, T, H', W') tensor. Works for still images too, with T = 1.
    """
    n, c, t, h, w = videos.shape
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    grid = videos.new_ones(c, t, nrows * (h + pad) - pad, ncols * (w + pad) - pad)
    for i, video in enumerate(videos):
        top, left = (i // ncols) * (h + pad), (i % ncols) * (w + pad)
        grid[:, :, top : top + h, left : left + w] = video
    return grid


def write_video(path: str | Path, frames: np.ndarray, fps: int = 25) -> None:
    """Write (T, H, W, 3) uint8 frames as an H.264 ``.mp4`` or a looping ``.gif``."""
    path = Path(path)
    if path.suffix == ".mp4":
        _write_mp4(path, frames, fps)
    elif path.suffix == ".gif":
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(
            path, save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0
        )
    else:
        raise ValueError(f"unsupported video format {path.suffix!r}; use .mp4 or .gif")


def _write_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    # yuv420p, the pixel format every player supports, needs an even width and height.
    _, height, width, _ = frames.shape
    if height % 2 or width % 2:
        frames = np.pad(frames, ((0, 0), (0, height % 2), (0, width % 2), (0, 0)), mode="edge")
    codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream(codec, rate=fps)
        stream.height, stream.width = frames.shape[1:3]
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())  # flush the encoder


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Write an (H, W, 3) uint8 image; the format follows the file suffix."""
    Image.fromarray(image).save(path)


def count_frames(path: str | Path) -> int:
    """Return the number of frames in a video, from the container header when it has one."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.frames:
            return stream.frames
        return sum(1 for _ in container.decode(stream))


def read_frames(
    path: str | Path, indices: Sequence[int], short_side: int | None = None
) -> np.ndarray:
    """Decode the frames at ``indices`` and return them as (len(indices), H, W, 3) uint8.

    Indices may repeat and need not be sorted. Indices past the end of the video repeat its last
    frame. With ``short_side``, frames are resized (area interpolation) so that their shorter side
    has that length. Only the video stream is decoded.
    """
    wanted = set(indices)
    last_wanted = max(wanted)
    decoded: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        size = {}
        if short_side is not None:
            width, height = stream.codec_context.width, stream.codec_context.height
            scale = short_side / min(width, height)
            size = {
                "width": max(short_side, round(width * scale)),
                "height": max(short_side, round(height * scale)),
                "interpolation": "AREA",
            }
        last_frame = None
        for i, frame in enumerate(container.decode(stream)):
            last_frame = frame
            if i in wanted:
                decoded[i] = frame.to_ndarray(format="rgb24", **size)
            if i >= last_wanted:
                break
        if last_frame is None:
            raise ValueError(f"no video frames could be decoded from {path}")
        if len(decoded) < len(wanted):
            filler = last_frame.to_ndarray(format="rgb24", **size)
            return np.stack([decoded.get(i, filler) for i in indices])
    return np.stack([decoded[i] for i in indices])
