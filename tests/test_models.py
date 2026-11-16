import pytest
import torch

from videogan.models import FRAME_SIZE, NUM_FRAMES, Discriminator, Generator


def _state_keys(convs: list[str], norms: list[str]) -> set[str]:
    keys = {f"{name}.{p}" for name in convs for p in ("weight", "bias")}
    stats = ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")
    return keys | {f"{name}.{p}" for name in norms for p in stats}


def test_generator_output_shapes_and_ranges():
    torch.manual_seed(0)
    out = Generator()(torch.randn(2, 100))
    assert out.video.shape == (2, 3, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE)
    assert out.foreground.shape == out.video.shape
    assert out.background.shape == (2, 3, 1, FRAME_SIZE, FRAME_SIZE)
    assert out.mask.shape == (2, 1, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE)
    assert out.video.abs().max() <= 1
    assert out.mask.min() >= 0
    assert out.mask.max() <= 1


def test_discriminator_output_shapes():
    videos = torch.zeros(2, 3, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE)
    assert Discriminator()(videos).shape == (2, 1)
    assert Discriminator(num_outputs=5)(videos).shape == (2, 5)


def test_discriminator_rejects_other_video_sizes():
    with pytest.raises(ValueError, match="expected videos"):
        Discriminator()(torch.zeros(2, 3, 16, FRAME_SIZE, FRAME_SIZE))


def test_layer_names_match_the_2022_checkpoints():
    """The 2022 checkpoints load with strict=True only if every layer keeps its name."""
    background = ["conv1b", "conv2b", "conv3b", "conv4b", "conv5b"]
    foreground = ["conv1", "conv2", "conv3", "conv4", "conv5", "conv5m"]
    generator_keys = _state_keys(
        [*background, *foreground], ["bn1b", "bn2b", "bn3b", "bn4b", "bn1", "bn2", "bn3", "bn4"]
    )
    discriminator_keys = _state_keys(
        ["conv1", "conv2", "conv3", "conv4", "conv5"], ["bn2", "bn3", "bn4"]
    )
    assert set(Generator().state_dict()) == generator_keys
    assert set(Discriminator().state_dict()) == discriminator_keys


def test_parameter_counts():
    assert sum(p.numel() for p in Generator().parameters()) == 16_245_383
    assert sum(p.numel() for p in Discriminator().parameters()) == 11_041_473


def test_replace_head_keeps_the_body_and_starts_a_new_head():
    model = Discriminator()
    body = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith("conv5")}
    model.replace_head(4, dropout=0.5)

    assert model.num_outputs == 4
    assert model.conv5.weight.shape == (4, 512, 2, 4, 4)
    assert model.conv5.weight.std() < 0.02
    assert torch.count_nonzero(model.conv5.bias) == 0
    assert model.dropout.p == 0.5
    for key, value in body.items():
        torch.testing.assert_close(model.state_dict()[key], value)
    model.eval()
    assert model(torch.zeros(3, 3, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE)).shape == (3, 4)
