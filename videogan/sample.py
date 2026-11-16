"""Generate videos with a trained VideoGAN generator.

Example::

    python -m videogan.sample --checkpoint runs/golf/checkpoints/last.pt --out-dir samples
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from .models import Generator
from .utils import gan_state_dicts, load_checkpoint, resolve_device
from .video_io import tile, to_uint8, write_image, write_video


@torch.no_grad()
def write_sample_grid(generator: Generator, z: torch.Tensor, prefix: Path, fps: int = 25) -> None:
    """Write ``<prefix>.mp4``, a grid of the videos for latent codes ``z``, and
    ``<prefix>_bg.jpg``, the grid of their background streams.

    The generator runs in eval mode (BatchNorm uses its running statistics) and is put back in
    the mode it was in.
    """
    was_training = generator.training
    generator.eval()
    out = generator(z)
    generator.train(was_training)
    ncols = math.ceil(math.sqrt(len(z)))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_video(prefix.with_name(prefix.name + ".mp4"), to_uint8(tile(out.video, ncols)), fps)
    write_image(prefix.with_name(prefix.name + "_bg.jpg"), to_uint8(tile(out.background, ncols))[0])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate videos with a trained VideoGAN generator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="GAN training checkpoint")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("samples"), help="folder to write the videos to"
    )
    parser.add_argument("--num-videos", type=int, default=16, help="videos to generate")
    parser.add_argument(
        "--format", choices=["gif", "mp4"], default="gif", help="file type of the videos"
    )
    parser.add_argument("--fps", type=int, default=25, help="frames per second")
    parser.add_argument("--seed", type=int, default=0, help="seed for the latent codes")
    parser.add_argument("--device", default="auto", help='"auto", "cuda", "cpu", ...')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    generator_state, _ = gan_state_dicts(load_checkpoint(args.checkpoint, map_location=device))
    z_dim = generator_state["conv1.weight"].shape[0]
    generator = Generator(z_dim).to(device).eval()
    generator.load_state_dict(generator_state)

    z = torch.randn(args.num_videos, z_dim, generator=torch.Generator().manual_seed(args.seed))
    with torch.no_grad():
        out = generator(z.to(device))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, (video, background) in enumerate(zip(out.video, out.background, strict=True)):
        write_video(args.out_dir / f"video_{i:03d}.{args.format}", to_uint8(video), args.fps)
        write_image(args.out_dir / f"background_{i:03d}.png", to_uint8(background)[0])
    ncols = math.ceil(math.sqrt(args.num_videos))
    write_video(args.out_dir / f"grid.{args.format}", to_uint8(tile(out.video, ncols)), args.fps)
    print(f"Wrote {args.num_videos} videos to {args.out_dir}")


if __name__ == "__main__":
    main()
