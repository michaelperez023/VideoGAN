"""Train VideoGAN on the golf-scene dataset.

The update rule follows train_golf.py from yhjo09/videogan-pytorch
(https://github.com/yhjo09/videogan-pytorch): one discriminator step and one generator step per
batch, binary cross-entropy losses, Adam with lr 2e-4 and beta1 0.5, and an L1 penalty with
weight 0.1 that keeps the foreground mask sparse.

Example::

    python -m videogan.train --data-root data/golf --out-dir runs/golf
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from .data import GolfStrips, collate_skip_none
from .models import Discriminator, Generator
from .sample import write_sample_grid
from .utils import (
    CsvLog,
    autocast,
    gan_state_dicts,
    grad_scaler,
    load_checkpoint,
    plain_args,
    resolve_device,
    save_checkpoint,
    seed_everything,
)

log = logging.getLogger("videogan.train")

LOG_FIELDS = ("epoch", "step", "epoch_progress", "loss_d", "loss_g", "d_real", "d_fake")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VideoGAN on the golf-scene dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root", type=Path, required=True, help="folder with golf.txt and frames-stable-many/"
    )
    parser.add_argument("--index", default="golf.txt", help="list of JPEG strips in --data-root")
    parser.add_argument(
        "--frames-dir", default="frames-stable-many", help="folder the listed paths start from"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/golf"),
        help="folder for checkpoints, samples and train_log.csv",
    )
    parser.add_argument("--epochs", type=int, default=10, help="total number of epochs")
    parser.add_argument("--batch-size", type=int, default=20, help="clips per batch")
    parser.add_argument("--lr", type=float, default=2e-4, help="Adam learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1")
    parser.add_argument("--z-dim", type=int, default=100, help="size of the latent code")
    parser.add_argument("--mask-l1", type=float, default=0.1, help="weight of the mask penalty")
    parser.add_argument("--num-workers", type=int, default=4, help="data loading processes")
    parser.add_argument("--log-every", type=int, default=100, help="steps per log line")
    parser.add_argument(
        "--sample-every",
        type=int,
        default=2000,
        help="steps between sample grids, which are also written after every epoch (0: only then)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=2000,
        help="steps between checkpoints, which are also saved after every epoch (0: only then)",
    )
    parser.add_argument("--num-samples", type=int, default=16, help="videos per sample grid")
    parser.add_argument("--max-steps", type=int, default=None, help="stop after this many steps")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint to continue from")
    parser.add_argument("--device", default="auto", help='"auto", "cuda", "cuda:1", "cpu", ...')
    parser.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for the weights, the noise and the clip order"
    )
    return parser.parse_args(argv)


class EpochSampler(Sampler[int]):
    """A shuffled order of the dataset that is the same for a given seed and epoch.

    ``start`` skips the first items, so a run resumed in the middle of an epoch continues with
    the clips it had not seen yet.
    """

    def __init__(self, size: int, seed: int, epoch: int, start: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed * 10_000 + epoch)
        self.order = torch.randperm(size, generator=generator)[start:].tolist()

    def __iter__(self) -> Iterator[int]:
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)


def _update(loss: torch.Tensor, optimizer: torch.optim.Optimizer, scaler: Any) -> None:
    optimizer.zero_grad(set_to_none=True)
    if scaler is None:
        loss.backward()
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def train_step(
    generator: Generator,
    discriminator: Discriminator,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    real: torch.Tensor,
    mask_l1: float = 0.1,
    scaler_g: Any = None,
    scaler_d: Any = None,
    amp: bool = False,
) -> dict[str, torch.Tensor]:
    """Run one discriminator update and one generator update on a batch of real clips.

    Returns the losses and the discriminator's mean probabilities as detached scalar tensors,
    so the caller decides when to synchronize with the device.
    """
    n, device = real.shape[0], real.device
    ones = torch.ones(n, 1, device=device)
    zeros = torch.zeros(n, 1, device=device)

    # Discriminator: real clips should score 1 and generated clips 0.
    with torch.no_grad(), autocast(device, amp):
        fake = generator(torch.randn(n, generator.z_dim, device=device)).video
    with autocast(device, amp):
        logit_real = discriminator(real)
        logit_fake = discriminator(fake)
        loss_d = 0.5 * (
            F.binary_cross_entropy_with_logits(logit_real, ones)
            + F.binary_cross_entropy_with_logits(logit_fake, zeros)
        )
    _update(loss_d, opt_d, scaler_d)

    # Generator: make the discriminator score new samples as real, and keep the mask sparse.
    with autocast(device, amp):
        out = generator(torch.randn(n, generator.z_dim, device=device))
        logit_gen = discriminator(out.video)
        loss_g = F.binary_cross_entropy_with_logits(logit_gen, ones)
        loss_g = loss_g + mask_l1 * out.mask.float().abs().mean()
    _update(loss_g, opt_g, scaler_g)

    return {
        "loss_d": loss_d.detach().float(),
        "loss_g": loss_g.detach().float(),
        "d_real": torch.sigmoid(logit_real.detach().float()).mean(),
        "d_fake": torch.sigmoid(logit_fake.detach().float()).mean(),
    }


@dataclass
class Position:
    """Where training stands: the epoch, the batches done in it, and the total steps."""

    epoch: int = 0
    batch: int = 0
    step: int = 0


def restore(
    checkpoint: dict[str, Any],
    generator: Generator,
    discriminator: Discriminator,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    scaler_g: Any,
    scaler_d: Any,
    batches_per_epoch: int,
) -> Position:
    """Load a checkpoint into the models and optimizers and return where to continue.

    Also accepts the end-of-epoch checkpoints of the 2022 run, which training continues from
    the start of the next epoch.
    """
    generator_state, discriminator_state = gan_state_dicts(checkpoint)
    generator.load_state_dict(generator_state)
    discriminator.load_state_dict(discriminator_state)
    if "model_state_dict" in checkpoint:  # 2022 format
        opt_g_state, opt_d_state = checkpoint["optimizer_state_dict"]
        opt_g.load_state_dict(opt_g_state)
        opt_d.load_state_dict(opt_d_state)
        epoch = checkpoint["epoch"] + 1
        return Position(epoch=epoch, batch=0, step=epoch * batches_per_epoch)
    opt_g.load_state_dict(checkpoint["opt_g"])
    opt_d.load_state_dict(checkpoint["opt_d"])
    # A scaler saved while mixed precision was off has an empty state.
    if checkpoint["scaler_g"]:
        scaler_g.load_state_dict(checkpoint["scaler_g"])
    if checkpoint["scaler_d"]:
        scaler_d.load_state_dict(checkpoint["scaler_d"])
    return Position(checkpoint["epoch"], checkpoint["batch"], checkpoint["step"])


def check_clip_order(checkpoint: dict[str, Any], args: argparse.Namespace, num_clips: int) -> None:
    """Exit if a checkpoint from the middle of an epoch is resumed with a different clip order.

    The rest of the epoch is the tail of a shuffled order that these settings determine; with
    other values, some clips would be used twice in that epoch and others not at all.
    """
    saved_args = checkpoint.get("args", {})
    settings = {
        "--batch-size": (saved_args.get("batch_size"), args.batch_size),
        "--seed": (saved_args.get("seed"), args.seed),
        "--index": (saved_args.get("index"), args.index),
        "the number of clips": (checkpoint.get("num_clips"), num_clips),
    }
    changed = [
        f"{name} was {saved}, now {current}"
        for name, (saved, current) in settings.items()
        if saved is not None and saved != current
    ]
    if changed:
        raise SystemExit(
            "the checkpoint stopped in the middle of an epoch, and finishing that epoch in the "
            f"same clip order needs the settings it was saved with ({'; '.join(changed)})"
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    device = resolve_device(args.device)
    amp = args.amp and device.type == "cuda"
    if args.amp and not amp:
        log.warning("--amp needs a CUDA device; training in full precision")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    seed_everything(args.seed)

    dataset = GolfStrips(args.data_root, args.index, args.frames_dir)
    batches_per_epoch = len(dataset) // args.batch_size
    if batches_per_epoch == 0:
        raise SystemExit(f"--batch-size {args.batch_size} is larger than the dataset")

    generator = Generator(args.z_dim).to(device)
    discriminator = Discriminator().to(device)
    betas = (args.beta1, 0.999)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=betas)
    scaler_g, scaler_d = grad_scaler(amp), grad_scaler(amp)
    # Fixed latent codes, so samples from different steps show the same scenes.
    fixed_z = torch.randn(
        args.num_samples, args.z_dim, generator=torch.Generator().manual_seed(args.seed)
    )

    pos = Position()
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, map_location=device)
        pos = restore(
            checkpoint,
            generator,
            discriminator,
            opt_g,
            opt_d,
            scaler_g,
            scaler_d,
            batches_per_epoch,
        )
        if pos.batch:
            check_clip_order(checkpoint, args, len(dataset))
        fixed_z = checkpoint.get("fixed_z", fixed_z).cpu()
        log.info("Resumed from %s at epoch %d, step %d", args.resume, pos.epoch, pos.step)
    fixed_z = fixed_z.to(device)

    def save(name: str) -> None:
        save_checkpoint(
            args.out_dir / "checkpoints" / name,
            {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "scaler_g": scaler_g.state_dict(),
                "scaler_d": scaler_d.state_dict(),
                "epoch": pos.epoch,
                "batch": pos.batch,
                "step": pos.step,
                "num_clips": len(dataset),
                "fixed_z": fixed_z.cpu(),
                "args": plain_args(args),
            },
        )

    train_log = CsvLog(args.out_dir / "train_log.csv", LOG_FIELDS)
    if args.resume is not None:
        # Steps after the checkpoint are trained again, so rows already logged for them would
        # appear twice.
        dropped = train_log.drop_rows(lambda row: int(row["step"]) > pos.step)
        if dropped:
            log.info("Dropped %d log rows written after step %d", dropped, pos.step)
    samples = args.out_dir / "samples"
    totals: dict[str, torch.Tensor] = {}
    window = 0
    log.info(
        "Training on %d clips (%d batches per epoch) on %s", len(dataset), batches_per_epoch, device
    )

    while pos.epoch < args.epochs:
        sampler = EpochSampler(len(dataset), args.seed, pos.epoch, pos.batch * args.batch_size)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_skip_none,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
        for real in loader:
            pos.batch += 1
            if real is None:  # every strip in this batch was too short
                continue
            pos.step += 1
            metrics = train_step(
                generator,
                discriminator,
                opt_g,
                opt_d,
                real.to(device, non_blocking=True),
                mask_l1=args.mask_l1,
                scaler_g=scaler_g,
                scaler_d=scaler_d,
                amp=amp,
            )
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0) + value
            window += 1

            if args.log_every > 0 and pos.step % args.log_every == 0:
                means = {key: (value / window).item() for key, value in totals.items()}
                progress = pos.epoch + pos.batch / batches_per_epoch
                train_log.write(
                    {"epoch": pos.epoch, "step": pos.step, "epoch_progress": f"{progress:.4f}"}
                    | {key: f"{value:.4f}" for key, value in means.items()}
                )
                log.info(
                    "epoch %d  batch %d/%d  step %d  D %.4f  G %.4f  D(real) %.3f  D(fake) %.3f",
                    pos.epoch,
                    pos.batch,
                    batches_per_epoch,
                    pos.step,
                    means["loss_d"],
                    means["loss_g"],
                    means["d_real"],
                    means["d_fake"],
                )
                totals.clear()
                window = 0
            if args.sample_every > 0 and pos.step % args.sample_every == 0:
                write_sample_grid(generator, fixed_z, samples / f"step{pos.step:07d}")
            if args.checkpoint_every > 0 and pos.step % args.checkpoint_every == 0:
                save("last.pt")
            if args.max_steps is not None and pos.step >= args.max_steps:
                save("last.pt")
                log.info("Stopped after --max-steps %d", args.max_steps)
                return

        write_sample_grid(generator, fixed_z, samples / f"epoch{pos.epoch:02d}")
        pos.epoch, pos.batch = pos.epoch + 1, 0
        save(f"epoch{pos.epoch - 1:02d}.pt")
        save("last.pt")
        log.info("Finished epoch %d", pos.epoch - 1)

    log.info("Training complete: %d epochs, %d steps", args.epochs, pos.step)


if __name__ == "__main__":
    main()
