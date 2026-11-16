"""Helpers shared by the command-line tools: devices, seeding, mixed precision, checkpoints."""

from __future__ import annotations

import contextlib
import csv
import os
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch


def resolve_device(name: str = "auto") -> torch.device:
    """Return the device to run on; ``"auto"`` picks CUDA when it is available, else the CPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    """Seed Python's and PyTorch's random number generators (PyTorch on every device)."""
    random.seed(seed)
    torch.manual_seed(seed)


def grad_scaler(enabled: bool) -> Any:
    """Return a CUDA loss scaler for mixed precision; it does nothing when ``enabled`` is False."""
    if hasattr(torch.amp, "GradScaler"):  # PyTorch 2.3 and later
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast(device: torch.device, enabled: bool) -> contextlib.AbstractContextManager:
    """Float16 autocast on ``device`` when ``enabled``, otherwise a no-op context."""
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Save ``state`` atomically, so an interrupted job never leaves a truncated checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint of tensors and plain Python values (no pickled objects)."""
    return torch.load(path, map_location=map_location, weights_only=True)


def gan_state_dicts(checkpoint: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the (generator, discriminator) weights stored in a GAN checkpoint.

    Accepts checkpoints written by ``videogan.train`` and the ``*_state_normal.ckpt`` files that
    the 2022 training script wrote.
    """
    if "generator" in checkpoint:
        return checkpoint["generator"], checkpoint["discriminator"]
    if "model_state_dict" in checkpoint:  # 2022 format: [generator, discriminator]
        generator, discriminator = checkpoint["model_state_dict"]
        return generator, discriminator
    raise ValueError("not a VideoGAN training checkpoint")


def plain_args(args: Any) -> dict[str, Any]:
    """Command-line arguments as a dict that ``torch.load(weights_only=True)`` can read back."""
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


class CsvLog:
    """Append rows to a CSV file, writing the header first when the file is new."""

    def __init__(self, path: Path, fields: Sequence[str]) -> None:
        self.path = path
        self.fields = list(fields)
        if not path.exists() or path.stat().st_size == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as fh:
                csv.writer(fh).writerow(self.fields)

    def write(self, row: dict[str, Any]) -> None:
        with self.path.open("a", newline="") as fh:
            csv.DictWriter(fh, self.fields).writerow(row)

    def drop_rows(self, condition: Callable[[dict[str, str]], bool]) -> int:
        """Remove the rows for which ``condition`` is true and return how many there were.

        ``condition`` gets each row as read back from the file, with string values.
        """
        with self.path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        kept = [row for row in rows if not condition(row)]
        if len(kept) < len(rows):
            tmp = self.path.with_name(self.path.name + ".tmp")
            with tmp.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, self.fields)
                writer.writeheader()
                writer.writerows(kept)
            os.replace(tmp, self.path)
        return len(rows) - len(kept)
