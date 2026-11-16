"""VideoGAN in PyTorch: golf-scene video generation and UCF-101 fine-tuning of the discriminator."""

from .models import Discriminator, Generator, GeneratorOutput

__all__ = ["Discriminator", "Generator", "GeneratorOutput"]
