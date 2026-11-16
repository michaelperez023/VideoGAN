import csv

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from videogan.finetune import build_classifier, evaluate, main, make_optimizer, train_one_epoch
from videogan.utils import load_checkpoint

CPU = torch.device("cpu")


def _videos(n: int) -> torch.Tensor:
    return torch.rand(n, 3, 32, 64, 64) * 2 - 1


def test_new_head_is_trained(gan_checkpoint):
    """In 2022 the optimizer was created before the head was swapped, so the head never changed."""
    model = build_classifier(3, gan_checkpoint, dropout=0.5)
    optimizer = make_optimizer(model, lr=1e-3, beta1=0.5)
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) in optimized for p in model.conv5.parameters())

    gan_weights = torch.load(gan_checkpoint, weights_only=True)["discriminator"]
    torch.testing.assert_close(model.conv1.weight, gan_weights["conv1.weight"])

    head_before = model.conv5.weight.detach().clone()
    data = TensorDataset(_videos(4), torch.tensor([0, 1, 2, 0]))
    metrics = train_one_epoch(model, DataLoader(data, batch_size=2), optimizer, CPU)

    assert model.conv5.weight.grad is not None
    assert not torch.equal(model.conv5.weight, head_before)
    assert 0 <= metrics["acc"] <= 1


def test_evaluate_scores_every_test_video_in_eval_mode():
    model = build_classifier(2)
    calls = []
    model.register_forward_hook(
        lambda module, inputs, _: calls.append(
            (module.training, torch.is_grad_enabled(), len(inputs[0]))
        )
    )
    clips = _videos(14).view(7, 2, 3, 32, 64, 64)  # 7 videos with 2 clips each
    data = TensorDataset(clips, torch.tensor([0, 1, 0, 1, 0, 1, 0]))
    model.train()

    metrics = evaluate(model, DataLoader(data, batch_size=3), CPU)

    assert [n for _, _, n in calls] == [6, 6, 2]  # all 14 clips, including the short last batch
    assert not any(training or grad for training, grad, _ in calls)
    assert set(metrics) == {"loss", "clip_acc", "video_acc"}


def test_video_accuracy_averages_clip_probabilities():
    class Stub(nn.Module):
        """Clips filled with 1 score [0, 10] (sure of class 1), clips of 0 score [1, 0]."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            v = x.flatten(1)[:, 0]
            return torch.stack([1 - v, 10 * v], dim=1)

    clips = torch.zeros(1, 3, 3, 32, 64, 64)
    clips[0, 0] = 1  # one confident clip for class 1, two weak ones for class 0
    data = TensorDataset(clips, torch.tensor([1]))

    metrics = evaluate(Stub(), DataLoader(data), CPU)

    assert metrics["clip_acc"] == pytest.approx(1 / 3)
    assert metrics["video_acc"] == 1.0


def test_main_fine_tunes_and_evaluates(ucf_root, gan_checkpoint, tmp_path):
    data, splits = ucf_root
    out = tmp_path / "finetune"
    main(
        [
            "--data-root", str(data),
            "--split-dir", str(splits),
            "--gan-checkpoint", str(gan_checkpoint),
            "--out-dir", str(out),
            "--epochs", "2",
            "--batch-size", "2",
            "--test-clips", "2",
            "--num-workers", "0",
            "--device", "cpu",
        ]
    )  # fmt: skip

    with (out / "metrics.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert [row["epoch"] for row in rows] == ["0", "1"]
    assert all(0 <= float(row["test_video_acc"]) <= 1 for row in rows)
    saved = load_checkpoint(out / "classifier.pt")
    assert saved["class_names"] == ["ClassA", "ClassB"]
    assert saved["model"]["conv5.weight"].shape[0] == 2


def test_main_checks_the_number_of_classes(ucf_root, tmp_path):
    data, splits = ucf_root
    args = ["--data-root", str(data), "--split-dir", str(splits), "--num-classes", "101"]
    with pytest.raises(SystemExit, match="does not match"):
        main([*args, "--out-dir", str(tmp_path / "out"), "--device", "cpu"])
