"""Checks of the train/test lists in splits/."""

import re
from pathlib import Path

import pytest

from videogan.data import read_class_index, read_split

SPLITS = Path(__file__).resolve().parent.parent / "splits"
FOLDS = [("kayaking-yoyo", 1), ("ucf101", 1), ("ucf101", 2), ("ucf101", 3)]
# Class folder and group of paths such as "YoYo/v_YoYo_g23_c06.avi".
VIDEO_PATH = re.compile(r"(?P<cls>[^/]+)/v_[^_]+_g(?P<group>\d+)_c\d+\.avi")


def _groups(paths: list[str]) -> set[tuple[str, int]]:
    groups = set()
    for path in paths:
        match = VIDEO_PATH.fullmatch(path)
        assert match, path
        groups.add((match["cls"], int(match["group"])))
    return groups


@pytest.mark.parametrize(("split", "fold"), FOLDS)
def test_no_group_is_on_both_sides_of_a_split(split, fold):
    """Videos of one group share scenes and people, so a group must not be split."""
    train = _groups(read_split(SPLITS / split, train=True, fold=fold))
    test = _groups(read_split(SPLITS / split, train=False, fold=fold))
    assert train
    assert test
    assert train.isdisjoint(test)


@pytest.mark.parametrize(("split", "fold"), FOLDS)
def test_training_labels_match_the_class_index(split, fold):
    classes = read_class_index(SPLITS / split)
    lines = (SPLITS / split / f"trainlist{fold:02d}.txt").read_text().splitlines()
    for line in lines:
        path, label = line.split()
        assert classes[int(label) - 1] == path.split("/")[0], line
