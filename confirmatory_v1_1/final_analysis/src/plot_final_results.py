"""Generate publication-ready confirmatory figures from canonical CSV tables."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "final_analysis" / "results"
FIGURES = ROOT / "final_analysis" / "figures"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GRAY = "#555555"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def save_all(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}_600dpi.png", dpi=600, bbox_inches="tight")


def main() -> None:
    curves = read_csv(RESULTS / "budget_curve_summary.csv")
    deltas = read_csv(RESULTS / "primary_acquisition_deltas_20.csv")
    primary = read_csv(RESULTS / "primary_summary_2.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.3), constrained_layout=True)
    for column, dataset in enumerate(("GC10-DET", "NEU-DET")):
        subset = sorted((row for row in curves if row["dataset"] == dataset), key=lambda row: int(row["round"]))
        x = np.asarray([100.0 * float(row["budget_fraction"]) for row in subset])
        random = np.asarray([100.0 * float(row["random_mean_map50_95"]) for row in subset])
        dino = np.asarray([100.0 * float(row["dino_mean_map50_95"]) for row in subset])
        ax = axes[0, column]
        ax.plot(x, random, marker="o", color=BLUE, label="Random")
        ax.plot(x, dino, marker="s", color=VERMILLION, label="Frozen DINO")
        ax.set_title(dataset)
        ax.set_xlabel("Labeled acquisition pool (%)")
        ax.set_ylabel("Locked-final mAP50–95 (%)")
        ax.grid(axis="y", color="#D0D0D0", linewidth=0.6)
        ax.legend(frameon=False, loc="lower right")

        paired = sorted((row for row in deltas if row["dataset"] == dataset), key=lambda row: int(row["acquisition_index"]))
        y = np.asarray([100.0 * float(row["dino_minus_random"]) for row in paired])
        ax = axes[1, column]
        ax.axhline(0.0, color=GRAY, linewidth=0.9)
        ax.scatter(np.arange(10), y, color=np.where(y >= 0, VERMILLION, BLUE), s=34, zorder=3)
        summary = next(row for row in primary if row["dataset"] == dataset)
        mean = 100.0 * float(summary["mean_dino_minus_random"])
        low = 100.0 * float(summary["ci95_low"])
        high = 100.0 * float(summary["ci95_high"])
        ax.errorbar(10.7, mean, yerr=[[mean - low], [high - mean]], fmt="D", color="black", capsize=4, label="Mean and 95% CI")
        ax.set_xlim(-0.6, 11.4)
        ax.set_xticks([0, 2, 4, 6, 8, 10.7], ["a00", "a02", "a04", "a06", "a08", "Mean"])
        ax.set_xlabel("Acquisition seed")
        ax.set_ylabel("DINO − Random AULC (percentage points)")
        ax.grid(axis="y", color="#D0D0D0", linewidth=0.6)
        ax.legend(frameon=False, loc="best")
    fig.suptitle("Controlled translation from frozen DINO selection to downstream detection", fontsize=13)
    save_all(fig, "confirmatory_primary_results")
    plt.close(fig)

    classwise = read_csv(RESULTS / "classwise_final_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.4), constrained_layout=True)
    for ax, dataset in zip(axes, ("GC10-DET", "NEU-DET")):
        subset = [row for row in classwise if row["dataset"] == dataset]
        names = [row["class_name"] for row in subset]
        means = np.asarray([100.0 * float(row["mean_dino_minus_random"]) for row in subset])
        lows = np.asarray([100.0 * float(row["ci95_low"]) for row in subset])
        highs = np.asarray([100.0 * float(row["ci95_high"]) for row in subset])
        y = np.arange(len(names))
        ax.axvline(0.0, color=GRAY, linewidth=0.9)
        ax.errorbar(means, y, xerr=np.vstack((means - lows, highs - means)), fmt="o", color=VERMILLION, ecolor=GRAY, capsize=3)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_title(dataset)
        ax.set_xlabel("Final-budget DINO − Random AP50–95 (percentage points)")
        ax.grid(axis="x", color="#D0D0D0", linewidth=0.6)
    fig.suptitle("Class-wise final-budget effects (descriptive 95% acquisition bootstrap CI)", fontsize=13)
    save_all(fig, "confirmatory_classwise_results")
    plt.close(fig)


if __name__ == "__main__":
    main()

