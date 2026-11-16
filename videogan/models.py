"""VideoGAN generator and discriminator (Vondrick, Pirsiavash & Torralba, NeurIPS 2016).

Adapted from yhjo09/videogan-pytorch (https://github.com/yhjo09/videogan-pytorch), a PyTorch port
of the original Torch7 code (https://github.com/cvondrick/videogan). The layers and their names
are unchanged, so the ``*_state_normal.ckpt`` checkpoints of the 2022 run load as they are.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

NUM_FRAMES = 32  # frames per video
FRAME_SIZE = 64  # height and width of a frame
Z_DIM = 100  # size of the latent code

_CONV_TYPES = (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)


def _init_conv(module: nn.Module) -> None:
    """Draw convolution weights from N(0, 0.01) and zero the biases, as in the paper.

    BatchNorm layers keep PyTorch's default initialization, which is also what the upstream code
    (and the 2022 run) used.
    """
    for m in module.modules():
        if isinstance(m, _CONV_TYPES):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.zeros_(m.bias)


class GeneratorOutput(NamedTuple):
    """Generated videos and the two streams they are composed from."""

    video: torch.Tensor  # (N, 3, T, H, W) in [-1, 1]
    foreground: torch.Tensor  # (N, 3, T, H, W) in [-1, 1]
    background: torch.Tensor  # (N, 3, 1, H, W) in [-1, 1], the same image for every frame
    mask: torch.Tensor  # (N, 1, T, H, W) in [0, 1]; 1 selects the foreground


class Generator(nn.Module):
    """Two-stream generator.

    A 3D transposed-convolution stream produces a moving foreground and a mask, a 2D stream
    produces a static background, and the video is ``mask * foreground + (1 - mask) * background``.
    """

    def __init__(self, z_dim: int = Z_DIM) -> None:
        super().__init__()
        self.z_dim = z_dim

        # Background, height x width: 1x1 -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64.
        self.conv1b = nn.ConvTranspose2d(z_dim, 512, 4, 1)
        self.bn1b = nn.BatchNorm2d(512)
        self.conv2b = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.bn2b = nn.BatchNorm2d(256)
        self.conv3b = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn3b = nn.BatchNorm2d(128)
        self.conv4b = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.bn4b = nn.BatchNorm2d(64)
        self.conv5b = nn.ConvTranspose2d(64, 3, 4, 2, 1)

        # Foreground, time x height x width: 1x1x1 -> 2x4x4 -> 4x8x8 -> ... -> 32x64x64.
        self.conv1 = nn.ConvTranspose3d(z_dim, 512, (2, 4, 4), 1)
        self.bn1 = nn.BatchNorm3d(512)
        self.conv2 = nn.ConvTranspose3d(512, 256, 4, 2, 1)
        self.bn2 = nn.BatchNorm3d(256)
        self.conv3 = nn.ConvTranspose3d(256, 128, 4, 2, 1)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = nn.ConvTranspose3d(128, 64, 4, 2, 1)
        self.bn4 = nn.BatchNorm3d(64)
        self.conv5 = nn.ConvTranspose3d(64, 3, 4, 2, 1)
        # The mask shares the foreground layers except the last one.
        self.conv5m = nn.ConvTranspose3d(64, 1, 4, 2, 1)

        _init_conv(self)

    def forward(self, z: torch.Tensor) -> GeneratorOutput:
        """Map latent codes (N, z_dim) to videos (N, 3, 32, 64, 64)."""
        n = z.shape[0]
        b = F.relu(self.bn1b(self.conv1b(z.view(n, self.z_dim, 1, 1))))
        b = F.relu(self.bn2b(self.conv2b(b)))
        b = F.relu(self.bn3b(self.conv3b(b)))
        b = F.relu(self.bn4b(self.conv4b(b)))
        b = torch.tanh(self.conv5b(b)).unsqueeze(2)

        f = F.relu(self.bn1(self.conv1(z.view(n, self.z_dim, 1, 1, 1))))
        f = F.relu(self.bn2(self.conv2(f)))
        f = F.relu(self.bn3(self.conv3(f)))
        f = F.relu(self.bn4(self.conv4(f)))
        m = torch.sigmoid(self.conv5m(f))
        f = torch.tanh(self.conv5(f))

        return GeneratorOutput(m * f + (1 - m) * b, f, b, m)


class Discriminator(nn.Module):
    """Five-layer 3D convolutional network that mirrors the generator's foreground stream.

    The last layer (``conv5``) is the head: a single real/fake logit while the GAN trains, or one
    logit per class after :meth:`replace_head`.
    """

    def __init__(self, num_outputs: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        # Time x height x width: 32x64x64 -> 16x32x32 -> 8x16x16 -> 4x8x8 -> 2x4x4 -> 1x1x1.
        self.conv1 = nn.Conv3d(3, 64, 4, 2, 1)
        self.conv2 = nn.Conv3d(64, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 256, 4, 2, 1)
        self.bn3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256, 512, 4, 2, 1)
        self.bn4 = nn.BatchNorm3d(512)
        self.dropout = nn.Dropout(dropout)  # has no parameters, so checkpoints are unaffected
        self.conv5 = nn.Conv3d(512, num_outputs, (2, 4, 4), 1)

        _init_conv(self)

    @property
    def num_outputs(self) -> int:
        return self.conv5.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map videos (N, 3, 32, 64, 64) in [-1, 1] to logits (N, num_outputs)."""
        if x.shape[1:] != (3, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE):
            raise ValueError(
                f"expected videos of shape (N, 3, {NUM_FRAMES}, {FRAME_SIZE}, {FRAME_SIZE}), "
                f"got {tuple(x.shape)}"
            )
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        return self.conv5(self.dropout(x)).flatten(1)

    def replace_head(self, num_outputs: int, dropout: float = 0.0) -> None:
        """Swap ``conv5`` for a newly initialized ``num_outputs``-way layer, with dropout before it.

        Create the optimizer after calling this, so that it also updates the new layer.
        """
        old = self.conv5
        self.conv5 = nn.Conv3d(old.in_channels, num_outputs, old.kernel_size, old.stride).to(
            device=old.weight.device, dtype=old.weight.dtype
        )
        _init_conv(self.conv5)
        self.dropout = nn.Dropout(dropout)
