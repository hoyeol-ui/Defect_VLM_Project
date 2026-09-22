"""Publication-style redesign using the audited, unchanged development CSVs.

This script only reads source tables and writes new figures/table in redesign_v2.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
TABLES = BASE / "02_tables"
STATS = BASE / "05_statistics"
METHODS = ("Random", "Entropy", "DINO_F518")
LABEL = {"Random": "Random", "Entropy": "Entropy", "DINO_F518": "DINOv2-FF"}
STYLE = {
    "Random": {"color": "#4A5564", "marker": "o", "linestyle": "-"},
    "Entropy": {"color": "#168378", "marker": "s", "linestyle": (0, (4, 2))},
    "DINO_F518": {"color": "#B45B25", "marker": "^", "linestyle": "-"},
}
DATASETS = (("GC10-DET", "(a) GC10-DET"), ("NEU-DET", "(b) NEU-DET"))

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 8.4, "axes.labelsize": 9,
    "axes.titlesize": 9.4, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.linewidth": .75, "pdf.fonttype": 42,
    "svg.fonttype": "none", "savefig.transparent": False,
})


def read(name: Path) -> list[dict[str, str]]:
    with name.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


NUM = read(TABLES / "02_FINAL_NUMERIC_RESULTS.csv")
NAULC = read(TABLES / "03_FINAL_NAULC_RESULTS.csv")
RAW_NAULC = read(TABLES / "RAW_SEED_NAULC.csv")
CLASS = read(TABLES / "05_CLASSWISE_RESULTS.csv")
PAIRED = read(STATS / "PAIRED_NAULC_SUMMARY.csv")


def save(fig: plt.Figure, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", pad_inches=.08, **kwargs)
    plt.close(fig)


def axes_style(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#7C8389")
    ax.tick_params(length=2.5, width=.65, colors="#30363B")
    ax.grid(axis=grid_axis, color="#E7E9EB", linewidth=.55, zorder=0)


def curve(dataset: str, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = sorted(
        (r for r in NUM if r["dataset"] == dataset and r["method"] == method),
        key=lambda r: int(r["round"][1:]),
    )
    assert len(sub) == 6 and all(r["partition"] == "development" for r in sub)
    return (
        np.array([float(r["labeled_pct"]) for r in sub]),
        np.array([float(r["map5095_mean_pct"]) for r in sub]),
        np.array([float(r["map5095_sd_pp"]) for r in sub]),
    )


def performance(shared_axis: bool) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(7.65, 3.55), sharey=shared_axis,
                            layout="constrained")
    handles = {}
    for ax, (dataset, title) in zip(axs, DATASETS):
        for method in METHODS:
            x, y, sd = curve(dataset, method)
            sty = STYLE[method]
            ax.fill_between(x, y-sd, y+sd, color=sty["color"], alpha=.085,
                            linewidth=0, zorder=1)
            line, = ax.plot(x, y, color=sty["color"], marker=sty["marker"],
                            linestyle=sty["linestyle"], linewidth=1.9,
                            markersize=3.9, markeredgewidth=.65, label=LABEL[method],
                            zorder=3)
            handles[method] = line
        full = next(r for r in NUM if r["dataset"] == dataset and r["round"] == "Full")
        full_y = float(full["map5095_mean_pct"])
        ax.axhline(full_y, color="#8D9397", linestyle=(0, (3, 2)),
                   linewidth=.95, zorder=2)
        ax.text(38.1, full_y+.33, f"Full-data: {full_y:.2f}", ha="right", va="bottom",
                fontsize=7.4, color="#62696F",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86, "pad": .8},
                zorder=5)
        ax.set_xlim(9, 39)
        ax.set_xticks([10, 15, 20, 25, 30, 35])
        ax.set_title(title, loc="left", fontweight="medium", pad=7)
        axes_style(ax)
    if shared_axis:
        axs[0].set_ylim(15, 40)
        axs[0].set_yticks([15, 20, 25, 30, 35, 40])
        stem = "MAIN_FIGURE_1_AL_CURVES_COMMON_15_40"
    else:
        axs[0].set_ylim(16, 31)
        axs[0].set_yticks([16, 19, 22, 25, 28, 31])
        axs[1].set_ylim(27, 41)
        axs[1].set_yticks([27, 30, 33, 36, 39])
        stem = "AL_CURVES_ALTERNATIVE_DATA_RANGE"
    axs[0].set_ylabel("mAP@0.50:0.95 (%)")
    if not shared_axis:
        axs[1].set_ylabel("mAP@0.50:0.95 (%)")
    fig.supxlabel("Labeled images (%)", fontsize=9)
    fig.legend([handles[m] for m in METHODS], [LABEL[m] for m in METHODS],
               loc="upper center", bbox_to_anchor=(.5, 1.09), ncol=3,
               frameon=False, handlelength=2.2, columnspacing=1.5)
    save(fig, stem)


def naulc_point_range() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(7.45, 3.5), layout="constrained")
    offsets = np.array([-.12, -.06, 0, .06, .12])
    for ax, (dataset, title) in zip(axs, DATASETS):
        for j, method in enumerate(METHODS):
            summary = next(r for r in NAULC if r["dataset"] == dataset and r["method"] == method)
            raw = sorted((r for r in RAW_NAULC if r["dataset"] == dataset and r["method"] == method),
                         key=lambda r: r["acquisition_seed"])
            assert len(raw) == 5 and int(summary["n_seeds"]) == 5
            sty = STYLE[method]
            ax.scatter(j + offsets, [float(r["naulc_pct"]) for r in raw], s=12,
                       marker=sty["marker"], facecolor="white", edgecolor=sty["color"],
                       linewidth=.7, zorder=3)
            ax.errorbar(j, float(summary["naulc_mean_pct"]),
                        yerr=float(summary["naulc_sd_pp"]), fmt=sty["marker"],
                        markersize=5.8, color=sty["color"], capsize=2.1,
                        capthick=1, elinewidth=1.15, zorder=5)
        ax.set_xlim(-.43, 2.43)
        ax.set_xticks(range(3), [LABEL[m] for m in METHODS])
        ax.set_title(title, loc="left", fontweight="medium", pad=7)
        ax.set_ylabel("nAULC (%)")
        axes_style(ax)
    axs[0].set_ylim(20.7, 24.7)
    axs[0].set_yticks([21, 22, 23, 24])
    axs[1].set_ylim(30, 33.8)
    axs[1].set_yticks([30, 31, 32, 33])
    save(fig, "NAULC_POINT_RANGE_ALTERNATIVE")


def naulc_table() -> None:
    lines = [
        "# 본문용 nAULC 표 초안",
        "",
        "**표 1.** GC10-DET와 NEU-DET의 development nAULC. 방법별 수치는 "
        "5개 acquisition seed의 평균 ± 표본 표준편차(%)이며, Δ는 같은 seed끼리 짝지은 "
        "DINOv2-FF − Random 평균 차이(%p)다. DINOv2-FF는 F518 입력 특징을 사용한다. "
        "사전 정의된 1차 분석은 아직 봉인된 test에 대해 수행해야 한다.",
        "",
        "| Dataset | Random | DINOv2-FF | Entropy | Δ DINOv2-FF − Random |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset, _ in DATASETS:
        cells = []
        for method in METHODS:
            r = next(r for r in NAULC if r["dataset"] == dataset and r["method"] == method)
            cells.append(f"{float(r['naulc_mean_pct']):.2f} ± {float(r['naulc_sd_pp']):.2f}")
        delta = next(r for r in PAIRED if r["dataset"] == dataset and
                     r["contrast"] == "DINO_F518 - Random")
        # Display order: Random, DINO, Entropy; no values are recomputed here.
        lines.append(f"| {dataset} | {cells[0]} | {cells[2]} | {cells[1]} | "
                     f"{float(delta['mean_delta_pp']):+.2f} |")
    lines += ["", "수치 출처: `02_tables/03_FINAL_NAULC_RESULTS.csv`와 "
              "`05_statistics/PAIRED_NAULC_SUMMARY.csv`. 표시는 소수 둘째 자리로 반올림했다.", ""]
    (OUT / "NAULC_MAIN_TABLE.md").write_text("\n".join(lines), encoding="utf-8")


def class_delta() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 4.8), layout="constrained")
    for ax, (dataset, title) in zip(axs, DATASETS):
        sub = sorted((r for r in CLASS if r["dataset"] == dataset and
                      r["method"] == "DINO_F518"),
                     key=lambda r: float(r["delta_vs_random_mean_pp"]), reverse=True)
        assert len(sub) == (10 if dataset == "GC10-DET" else 6)
        vals = np.array([float(r["delta_vs_random_mean_pp"]) for r in sub])
        names = [r["class_name"].replace("_", " ") for r in sub]
        ys = np.arange(len(sub))
        colors = np.where(vals >= 0, STYLE["DINO_F518"]["color"], "#6A7783")
        ax.barh(ys, vals, color=colors, height=.55, zorder=3)
        ax.axvline(0, color="#30363B", linewidth=.9, zorder=4)
        for y, val in zip(ys, vals):
            ax.text(val + (.27 if val >= 0 else -.27), y, f"{val:+.2f}",
                    ha="left" if val >= 0 else "right", va="center", fontsize=7.1,
                    color="#2F363A")
        ax.set_yticks(ys, names)
        ax.invert_yaxis()
        ax.set_xlim(-8.2, 16.3)
        ax.set_xticks([-8, -4, 0, 4, 8, 12, 16])
        ax.set_title(title, loc="left", fontweight="medium", pad=7)
        ax.set_xlabel("DINOv2-FF − Random R5 class AP (%p)")
        axes_style(ax, grid_axis="y")
        ax.tick_params(axis="y", length=0)
    save(fig, "MAIN_FIGURE_2_R5_CLASS_DELTA")


def absolute_class(dataset: str, short: str) -> None:
    sub = [r for r in CLASS if r["dataset"] == dataset]
    names = list(dict.fromkeys(r["class_name"] for r in sub))
    fig, ax = plt.subplots(figsize=(6.8, 4.7 if short == "GC10" else 3.7),
                           layout="constrained")
    offsets = {"Random": -.20, "Entropy": 0, "DINO_F518": .20}
    for method in METHODS:
        rr = [next(r for r in sub if r["class_name"] == name and r["method"] == method)
              for name in names]
        sty = STYLE[method]
        ax.errorbar([float(r["r5_ap_mean_pct"]) for r in rr],
                    np.arange(len(rr)) + offsets[method],
                    xerr=[float(r["r5_ap_sd_pp"]) for r in rr], fmt=sty["marker"],
                    color=sty["color"], markersize=3.6, capsize=1.9,
                    elinewidth=1, linewidth=0, label=LABEL[method], zorder=3)
    max_endpoint = max(float(r["r5_ap_mean_pct"]) + float(r["r5_ap_sd_pp"]) for r in sub)
    xmax = math.ceil((max_endpoint + 1.5) / 5) * 5
    ax.set_xlim(0, xmax)
    ax.set_xticks(np.arange(0, xmax + 1, 10))
    ax.set_yticks(range(len(names)), [n.replace("_", " ") for n in names])
    ax.invert_yaxis()
    ax.set_title(f"R5 class AP - {dataset}", loc="left", fontweight="medium", pad=7)
    ax.set_xlabel("Class AP@0.50:0.95 (%)")
    axes_style(ax, grid_axis="y")
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, ncol=3, loc="upper right",
              bbox_to_anchor=(1.0, 1.105), borderaxespad=0,
              handletextpad=.3, columnspacing=.9)
    save(fig, f"APPENDIX_{short}_R5_ABSOLUTE_CLASS_AP")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    performance(shared_axis=True)
    performance(shared_axis=False)
    naulc_point_range()
    naulc_table()
    class_delta()
    absolute_class("GC10-DET", "GC10")
    absolute_class("NEU-DET", "NEU")
    print("Wrote 6 redesigned figures x PNG/PDF/SVG and nAULC table")


if __name__ == "__main__":
    main()
