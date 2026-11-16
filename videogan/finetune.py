"""Fine-tune the VideoGAN discriminator for action recognition on UCF-101.

The real/fake output layer is replaced by a K-way classifier with dropout before it, as in the
paper, and the whole network is trained with cross-entropy. After every epoch the model is
evaluated on the full test list in eval mode: clip accuracy scores each clip on its own, and
video accuracy averages the class probabilities of a video's clips.

Examples::

    # Kayaking vs. YoYo
    python -m videogan.finetune --data-root data/UCF-101 --split-dir splits/kayaking-yoyo \\
        --gan-checkpoint runs/golf/checkpoints/last.pt --out-dir runs/kayaking-yoyo

    # All 101 classes, official split 1
    python -m videogan.finetune --data-root data/UCF-101 --split-dir splits/ucf101 \\
        --gan-checkpoint runs/golf/checkpoints/last.pt --out-dir runs/ucf101
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import UCF101Clips, read_class_index
from .models import Discriminator
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

log = logging.getLogger("videogan.finetune")

METRIC_FIELDS = ("epoch", "train_loss", "train_acc", "test_loss", "test_clip_acc", "test_video_acc")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the VideoGAN discriminator on UCF-101.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, required=True, help="UCF-101 video folder")
    parser.add_argument(
        "--split-dir",
        type=Path,
        required=True,
        help="folder with classInd.txt, trainlistNN.txt and testlistNN.txt",
    )
    parser.add_argument("--fold", type=int, default=1, help="which train/test list to use")
    parser.add_argument(
        "--num-classes", type=int, default=None, help="defaults to the classes in classInd.txt"
    )
    parser.add_argument(
        "--gan-checkpoint",
        type=Path,
        default=None,
        help="GAN checkpoint to start from; without one the network starts from random weights",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/finetune"),
        help="folder for metrics.csv and classifier.pt",
    )
    parser.add_argument("--epochs", type=int, default=20, help="passes over the training videos")
    parser.add_argument("--batch-size", type=int, default=16, help="videos per batch")
    parser.add_argument("--lr", type=float, default=2e-4, help="Adam learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout before the new head")
    parser.add_argument("--test-clips", type=int, default=5, help="clips per test video")
    parser.add_argument("--num-workers", type=int, default=4, help="data loading processes")
    parser.add_argument("--device", default="auto", help='"auto", "cuda", "cuda:1", "cpu", ...')
    parser.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    return parser.parse_args(argv)


def build_classifier(
    num_classes: int, gan_checkpoint: str | Path | None = None, dropout: float = 0.5
) -> Discriminator:
    """Return the discriminator with GAN weights (if given) and a new ``num_classes``-way head."""
    model = Discriminator()
    if gan_checkpoint is not None:
        _, discriminator_state = gan_state_dicts(load_checkpoint(gan_checkpoint))
        model.load_state_dict(discriminator_state)
    model.replace_head(num_classes, dropout)
    return model


def make_optimizer(model: torch.nn.Module, lr: float, beta1: float) -> torch.optim.Optimizer:
    """Adam over all parameters. Call it after the head is replaced, so the head is trained too."""
    return torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))


def train_one_epoch(
    model: Discriminator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Any = None,
    amp: bool = False,
) -> dict[str, float]:
    """Train for one pass over ``loader``; returns the mean loss and the accuracy."""
    model.train()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    seen = 0
    for videos, labels in loader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(device, amp):
            logits = model(videos)
            loss = F.cross_entropy(logits.float(), labels)
        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        loss_sum += loss.detach() * len(labels)
        correct += (logits.argmax(1) == labels).sum()
        seen += len(labels)
    return {"loss": loss_sum.item() / seen, "acc": correct.item() / seen}


@torch.no_grad()
def evaluate(
    model: Discriminator, loader: DataLoader, device: torch.device, amp: bool = False
) -> dict[str, float]:
    """Score every item of ``loader`` in eval mode.

    Items are (clips, label) with clips of shape (K, 3, T, H, W), or (3, T, H, W) for a single
    clip. Returns the mean clip loss, the clip accuracy, and the video accuracy, for which a
    video's class probabilities are averaged over its clips.
    """
    model.eval()
    loss_sum = clip_correct = video_correct = 0.0
    clips_seen = videos_seen = 0
    for clips, labels in loader:
        if clips.dim() == 5:
            clips = clips.unsqueeze(1)
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        n, k = clips.shape[:2]
        with autocast(device, amp):
            logits = model(clips.flatten(0, 1)).float().view(n, k, -1)
        loss_sum += F.cross_entropy(
            logits.flatten(0, 1), labels.repeat_interleave(k), reduction="sum"
        ).item()
        clip_correct += (logits.argmax(2) == labels[:, None]).sum().item()
        video_correct += (logits.softmax(2).mean(1).argmax(1) == labels).sum().item()
        clips_seen += n * k
        videos_seen += n
    return {
        "loss": loss_sum / clips_seen,
        "clip_acc": clip_correct / clips_seen,
        "video_acc": video_correct / videos_seen,
    }


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

    class_names = read_class_index(args.split_dir)
    num_classes = args.num_classes or len(class_names)
    if num_classes != len(class_names):
        raise SystemExit(
            f"--num-classes {num_classes} does not match the {len(class_names)} classes "
            f"in {args.split_dir / 'classInd.txt'}"
        )
    train_set = UCF101Clips(args.data_root, args.split_dir, train=True, fold=args.fold)
    test_set = UCF101Clips(
        args.data_root, args.split_dir, train=False, fold=args.fold, clips_per_video=args.test_clips
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)
    log.info(
        "%d classes, %d training videos, %d test videos (%d clips each)",
        num_classes,
        len(train_set),
        len(test_set),
        args.test_clips,
    )

    model = build_classifier(num_classes, args.gan_checkpoint, args.dropout).to(device)
    optimizer = make_optimizer(model, args.lr, args.beta1)
    scaler = grad_scaler(amp)
    metrics_log = CsvLog(args.out_dir / "metrics.csv", METRIC_FIELDS)

    for epoch in range(args.epochs):
        train = train_one_epoch(model, train_loader, optimizer, device, scaler, amp)
        test = evaluate(model, test_loader, device, amp)
        metrics_log.write(
            {
                "epoch": epoch,
                "train_loss": f"{train['loss']:.4f}",
                "train_acc": f"{train['acc']:.4f}",
                "test_loss": f"{test['loss']:.4f}",
                "test_clip_acc": f"{test['clip_acc']:.4f}",
                "test_video_acc": f"{test['video_acc']:.4f}",
            }
        )
        log.info(
            "epoch %d/%d  train loss %.4f acc %.3f  test loss %.4f  clip acc %.3f  video acc %.3f",
            epoch + 1,
            args.epochs,
            train["loss"],
            train["acc"],
            test["loss"],
            test["clip_acc"],
            test["video_acc"],
        )
        save_checkpoint(
            args.out_dir / "classifier.pt",
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "class_names": class_names,
                "test_metrics": test,
                "args": plain_args(args),
            },
        )


if __name__ == "__main__":
    main()
