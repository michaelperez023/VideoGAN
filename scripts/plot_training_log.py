"""Plot a VideoGAN training log (the CSV written by ``videogan.train``) as a PNG.

Three panels share the epoch axis: the discriminator loss, the generator loss, and the
discriminator's mean probability for real and for generated clips. Stretches of training that
are missing from the log are left blank and shaded.

Example::

    python scripts/plot_training_log.py logs/golf_training_2022.csv docs/figures/training_loss.png
"""

from __future__ import annotations

import argparse
import csv
import math
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "gap": "#f0efec",
        "series": ["#2a78d6", "#eb6834"],
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "gap": "#262624",
        "series": ["#3987e5", "#d95926"],
    },
}


def read_log(path: Path) -> dict[str, list[float]]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    keys = ("epoch_progress", "loss_d", "loss_g", "d_real", "d_fake")
    return {key: [float(row[key]) for row in rows] for key in keys}


def find_gaps(x: list[float]) -> list[tuple[float, float]]:
    """Return the (start, end) of every jump in ``x`` much larger than its usual spacing."""
    if len(x) < 3:
        return []
    steps = sorted(b - a for a, b in pairwise(x))
    typical = steps[len(steps) // 2]
    return [(a, b) for a, b in pairwise(x) if b - a > 3 * typical]


def with_breaks(x: list[float], y: list[float], gaps: list[tuple[float, float]]):
    """Insert NaNs at the gaps so lines are not drawn across missing data."""
    starts = {a for a, _ in gaps}
    xs, ys = [], []
    for xi, yi in zip(x, y, strict=True):
        xs.append(xi)
        ys.append(yi)
        if xi in starts:
            xs.append(xi)
            ys.append(float("nan"))
    return xs, ys


def plot(log: dict[str, list[float]], out: Path, theme: str, title: str) -> None:
    c = THEMES[theme]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
        }
    )
    x = log["epoch_progress"]
    gaps = find_gaps(x)
    fig, axes = plt.subplots(
        3, 1, figsize=(8, 7.2), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1]}
    )
    fig.patch.set_facecolor(c["surface"])
    panels = [
        ("Discriminator loss", [("loss_d", None)]),
        ("Generator loss (adversarial loss + mask penalty)", [("loss_g", None)]),
        (
            "Discriminator's mean probability of 'real'",
            [("d_real", "real clips"), ("d_fake", "generated clips")],
        ),
    ]
    for ax, (panel_title, series) in zip(axes, panels, strict=True):
        ax.set_facecolor(c["surface"])
        for a, b in gaps:
            ax.axvspan(a, b, color=c["gap"], lw=0, zorder=0)
        for (key, label), color in zip(series, c["series"], strict=False):
            xs, ys = with_breaks(x, log[key], gaps)
            ax.plot(xs, ys, color=color, lw=1.2, solid_capstyle="round", label=label, zorder=2)
            if label:  # direct label at the end of the line
                ax.annotate(
                    label,
                    (x[-1], log[key][-1]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    color=c["secondary"],
                    fontsize=9,
                )
        ax.set_title(panel_title, loc="left", color=c["primary"], fontsize=10.5, pad=6)
        ax.grid(axis="y", color=c["grid"], lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(c["axis"])
        ax.tick_params(colors=c["muted"], labelcolor=c["secondary"], length=3)
        if len(series) > 1:
            ax.set_ylim(0, 1)
            legend = ax.legend(
                loc="center left", bbox_to_anchor=(0.01, 0.5), frameon=False, fontsize=9
            )
            for text in legend.get_texts():
                text.set_color(c["secondary"])
    if gaps:
        a, b = gaps[0]
        axes[0].text(
            (a + b) / 2,
            0.95,
            "not logged",
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            color=c["muted"],
            fontsize=8.5,
        )
    last = axes[-1]
    last.set_xlabel("Epoch", color=c["secondary"])
    if x[-1] >= 1:  # whole epochs, so the ticks mark epoch boundaries
        epochs = math.ceil(x[-1])
        last.set_xlim(0, epochs)
        last.set_xticks(range(epochs + 1))
    else:  # a run shorter than one epoch
        last.set_xlim(0, x[-1])
    last.margins(x=0)
    if title:
        fig.suptitle(title, x=0.01, ha="left", color=c["primary"], fontsize=12, weight="bold")
    fig.tight_layout()
    fig.subplots_adjust(right=0.86)
    fig.savefig(out, dpi=150, facecolor=c["surface"])
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("log", type=Path, help="train_log.csv from a training run")
    parser.add_argument("out", type=Path, help="PNG file to write")
    parser.add_argument(
        "--theme", choices=sorted(THEMES), default="light", help="light or dark background"
    )
    parser.add_argument("--title", default="", help="title above the panels")
    args = parser.parse_args()
    plot(read_log(args.log), args.out, args.theme, args.title)


if __name__ == "__main__":
    main()
