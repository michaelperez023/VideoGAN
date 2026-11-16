import pytest
import torch

from videogan.models import Discriminator, Generator
from videogan.sample import main
from videogan.video_io import count_frames, read_frames


@pytest.mark.parametrize("layout", ["current", "2022"])
def test_sample_writes_videos_backgrounds_and_a_grid(tmp_path, layout):
    torch.manual_seed(0)
    weights = Generator().state_dict(), Discriminator().state_dict()
    if layout == "current":
        state = {"generator": weights[0], "discriminator": weights[1]}
    else:
        state = {"epoch": 9, "model_state_dict": list(weights)}
    checkpoint = tmp_path / "gan.pt"
    torch.save(state, checkpoint)
    out = tmp_path / "samples"

    main(
        [
            "--checkpoint", str(checkpoint),
            "--out-dir", str(out),
            "--num-videos", "3",
            "--format", "mp4",
            "--device", "cpu",
        ]
    )  # fmt: skip

    names = sorted(path.name for path in out.iterdir())
    assert names == [
        "background_000.png",
        "background_001.png",
        "background_002.png",
        "grid.mp4",
        "video_000.mp4",
        "video_001.mp4",
        "video_002.mp4",
    ]
    assert count_frames(out / "grid.mp4") == 32
    assert read_frames(out / "grid.mp4", [0]).shape == (1, 2 * 64 + 2, 2 * 64 + 2, 3)
