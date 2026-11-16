import csv
import shutil

import pytest
import torch

from videogan.models import Discriminator, Generator
from videogan.train import EpochSampler, Position, main, restore, train_step
from videogan.utils import grad_scaler, load_checkpoint


def _adam(model: torch.nn.Module) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))


def _train(golf_root, out_dir, *extra: str) -> None:
    main(
        [
            "--data-root", str(golf_root),
            "--out-dir", str(out_dir),
            "--batch-size", "2",
            "--num-workers", "0",
            "--num-samples", "4",
            "--log-every", "1",
            "--sample-every", "1",
            "--checkpoint-every", "0",  # only at --max-steps and at the end of each epoch
            "--device", "cpu",
            *extra,
        ]
    )  # fmt: skip


def _logged_steps(out_dir) -> list[int]:
    with (out_dir / "train_log.csv").open() as fh:
        return [int(row["step"]) for row in csv.DictReader(fh)]


def test_train_step_updates_both_networks():
    torch.manual_seed(0)
    generator, discriminator = Generator(), Discriminator()
    g_before = generator.conv5.weight.detach().clone()
    d_before = discriminator.conv1.weight.detach().clone()
    real = torch.rand(2, 3, 32, 64, 64) * 2 - 1

    metrics = train_step(generator, discriminator, _adam(generator), _adam(discriminator), real)

    assert set(metrics) == {"loss_d", "loss_g", "d_real", "d_fake"}
    assert all(torch.isfinite(value) for value in metrics.values())
    assert 0 <= metrics["d_real"] <= 1
    assert not torch.equal(generator.conv5.weight, g_before)
    assert not torch.equal(discriminator.conv1.weight, d_before)


def test_epoch_sampler_is_reproducible_and_can_start_mid_epoch():
    order = list(EpochSampler(10, seed=0, epoch=1))
    assert sorted(order) == list(range(10))
    assert order == list(EpochSampler(10, seed=0, epoch=1))
    assert order != list(EpochSampler(10, seed=0, epoch=2))
    assert list(EpochSampler(10, seed=0, epoch=1, start=4)) == order[4:]


def test_smoke_run_writes_samples_log_and_checkpoint(golf_root, tmp_path):
    out = tmp_path / "run"
    _train(golf_root, out, "--max-steps", "2")

    for name in ("step0000001.mp4", "step0000002.mp4", "step0000002_bg.jpg"):
        assert (out / "samples" / name).exists()
    assert _logged_steps(out) == [1, 2]
    checkpoint = load_checkpoint(out / "checkpoints" / "last.pt")
    assert (checkpoint["epoch"], checkpoint["batch"], checkpoint["step"]) == (0, 2, 2)


def test_resume_continues_where_training_stopped(golf_root, tmp_path):
    out = tmp_path / "run"
    _train(golf_root, out, "--max-steps", "1")
    _train(golf_root, out, "--resume", str(out / "checkpoints" / "last.pt"), "--epochs", "2")

    # Five strips at batch size 2 give two batches per epoch.
    final = load_checkpoint(out / "checkpoints" / "last.pt")
    assert (final["epoch"], final["batch"], final["step"]) == (2, 0, 4)
    assert _logged_steps(out) == [1, 2, 3, 4]
    assert (out / "checkpoints" / "epoch00.pt").exists()
    assert (out / "checkpoints" / "epoch01.pt").exists()
    assert (out / "samples" / "epoch01.mp4").exists()


def test_resume_drops_log_rows_written_after_the_checkpoint(golf_root, tmp_path):
    """A stopped job trains the steps after its last checkpoint again when it is resumed."""
    out = tmp_path / "run"
    _train(golf_root, out, "--max-steps", "1")
    checkpoint = tmp_path / "step1.pt"
    shutil.copy(out / "checkpoints" / "last.pt", checkpoint)
    _train(golf_root, out, "--resume", str(checkpoint), "--max-steps", "3")
    assert _logged_steps(out) == [1, 2, 3]

    # Continue from the step 1 checkpoint, as if the job had stopped before saving another one.
    _train(golf_root, out, "--resume", str(checkpoint), "--epochs", "2")
    assert _logged_steps(out) == [1, 2, 3, 4]


def test_mid_epoch_resume_needs_the_same_clip_order(golf_root, tmp_path):
    out = tmp_path / "run"
    _train(golf_root, out, "--max-steps", "1")  # stops after the first of two batches
    resume = ("--resume", str(out / "checkpoints" / "last.pt"))

    with pytest.raises(SystemExit, match="--seed was 0, now 1"):
        _train(golf_root, out, *resume, "--seed", "1")
    with pytest.raises(SystemExit, match="--batch-size was 2, now 1"):
        _train(golf_root, out, *resume, "--batch-size", "1")
    index = golf_root / "golf.txt"
    index.write_text("".join(index.read_text().splitlines(keepends=True)[:4]))
    with pytest.raises(SystemExit, match="number of clips was 5, now 4"):
        _train(golf_root, out, *resume)
    assert _logged_steps(out) == [1]


def test_restore_loads_models_optimizers_and_position(golf_root, tmp_path):
    out = tmp_path / "run"
    _train(golf_root, out, "--max-steps", "1")
    checkpoint = load_checkpoint(out / "checkpoints" / "last.pt")

    generator, discriminator = Generator(), Discriminator()
    opt_g, opt_d = _adam(generator), _adam(discriminator)
    scalers = grad_scaler(False), grad_scaler(False)
    position = restore(checkpoint, generator, discriminator, opt_g, opt_d, *scalers, 2)

    assert position == Position(epoch=0, batch=1, step=1)
    for key, value in checkpoint["generator"].items():
        torch.testing.assert_close(generator.state_dict()[key], value)
    for key, value in checkpoint["discriminator"].items():
        torch.testing.assert_close(discriminator.state_dict()[key], value)
    assert opt_d.state_dict()["state"]  # Adam moments came back too


def test_restore_accepts_2022_checkpoints(tmp_path):
    torch.manual_seed(0)
    old_g, old_d = Generator(), Discriminator()
    path = tmp_path / "2022-04-24-11-34-57_9_30187_state_normal.ckpt"
    torch.save(
        {
            "epoch": 9,
            "model_state_dict": [old_g.state_dict(), old_d.state_dict()],
            "optimizer_state_dict": [_adam(old_g).state_dict(), _adam(old_d).state_dict()],
        },
        path,
    )
    checkpoint = load_checkpoint(path)  # weights_only=True works on these files

    generator, discriminator = Generator(), Discriminator()
    scalers = grad_scaler(False), grad_scaler(False)
    position = restore(
        checkpoint, generator, discriminator, _adam(generator), _adam(discriminator), *scalers, 100
    )

    assert position == Position(epoch=10, batch=0, step=1000)
    torch.testing.assert_close(generator.conv5m.weight, old_g.conv5m.weight)
    torch.testing.assert_close(discriminator.conv5.weight, old_d.conv5.weight)
