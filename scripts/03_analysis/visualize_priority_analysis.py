"""
===============================================================================
[File] visualize_priority_analysis.py

[Purpose]
Create Notion-ready visualizations for VLM-based Active Learning priority score
analysis.

This script visualizes:
1) Groundedness reason distribution
2) Dataset-wise reason distribution
3) Strict vs SoftPenalty Top-K selection behavior
4) Priority score scatter plot
5) Top/Bottom priority examples
6) Strict vs SoftPenalty score shift
7) A single dashboard image for Notion

Input:
    outputs/pseudo_boxes_*/priority_scores_pseudo.csv
    outputs/pseudo_boxes_*/priority_scores_summary.json

Output:
    outputs/pseudo_boxes_*/priority_visualizations_YYYYMMDD_HHMMSS/
        01_reason_distribution.png
        02_dataset_reason_stacked.png
        03_strict_vs_soft_topk_reason.png
        04_priority_scatter.png
        05_top_bottom_priority.png
        06_strict_vs_soft_shift.png
        notion_summary_dashboard.png
        notion_interpretation.md
===============================================================================
"""

import json
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# =============================================================================
# [0] Project paths
# =============================================================================
PROJECT_ROOT = Path("/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project")
OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs"


# =============================================================================
# [1] Matplotlib settings
# =============================================================================
def set_korean_font():
    """
    Set Korean font for macOS/Windows/Linux if available.
    """
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    candidates = [
        "AppleGothic",
        "NanumGothic",
        "Malgun Gothic",
        "Noto Sans CJK KR",
        "DejaVu Sans"
    ]

    for font in candidates:
        if font in available_fonts:
            plt.rcParams["font.family"] = font
            break

    plt.rcParams["axes.unicode_minus"] = False


REASON_LABELS = {
    "no_pseudo_box": "No pseudo box",
    "partial_match": "Partial match",
    "location_mismatch": "Location mismatch",
    "matched": "Matched",
}

REASON_COLORS = {
    "no_pseudo_box": "#A0A0A0",
    "partial_match": "#5B8FF9",
    "location_mismatch": "#E76F51",
    "matched": "#2A9D8F",
}

REASON_ORDER = [
    "location_mismatch",
    "partial_match",
    "no_pseudo_box",
    "matched",
]


# =============================================================================
# [2] File loading
# =============================================================================
def find_latest_file(pattern: str) -> Path:
    files = list(OUTPUT_BASE_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return max(files, key=lambda p: p.stat().st_mtime)


def load_data():
    csv_file = find_latest_file("pseudo_boxes_*/priority_scores_pseudo.csv")
    summary_file = csv_file.parent / "priority_scores_summary.json"

    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    df = pd.read_csv(csv_file)

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    return csv_file, summary_file, df, summary


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "consistency_score",
        "groundedness_norm",
        "groundedness_effective_strict",
        "groundedness_effective_soft",
        "missing_box_penalty",
        "uncertainty_consistency",
        "uncertainty_groundedness_strict",
        "uncertainty_groundedness_soft",
        "score_combined_strict",
        "score_combined_soft_missing",
        "score_combined_soft_penalty",
        "best_box_score",
        "best_box_quality",
        "best_box_area_ratio",
        "best_box_aspect_ratio",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["groundedness_reason"] = df["groundedness_reason"].fillna("unknown")
    df["dataset_type"] = df["dataset_type"].fillna("unknown")

    return df


# =============================================================================
# [3] Utility plotting functions
# =============================================================================
def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {path}")


def add_bar_labels(ax, bars, fmt="{:.0f}", fontsize=9):
    for bar in bars:
        width = bar.get_width()
        height = bar.get_height()
        if width > height:
            ax.text(
                width + 0.2,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(width),
                va="center",
                fontsize=fontsize,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                fmt.format(height),
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )


# =============================================================================
# [4] Individual charts
# =============================================================================
def plot_reason_distribution(df: pd.DataFrame, save_dir: Path):
    counts = df["groundedness_reason"].value_counts()
    ordered = [r for r in REASON_ORDER if r in counts.index]
    values = [counts[r] for r in ordered]
    labels = [REASON_LABELS.get(r, r) for r in ordered]
    colors = [REASON_COLORS.get(r, "#999999") for r in ordered]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.title("Groundedness reason distribution", fontsize=14, fontweight="bold")
    plt.ylabel("Number of samples")
    plt.xlabel("Groundedness reason")
    plt.grid(axis="y", alpha=0.25)

    total = len(df)
    for bar, value in zip(bars, values):
        pct = value / total * 100
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    save_fig(save_dir / "01_reason_distribution.png")


def plot_dataset_reason_stacked(df: pd.DataFrame, save_dir: Path):
    pivot = (
        df.groupby(["dataset_type", "groundedness_reason"])
        .size()
        .unstack(fill_value=0)
    )

    for reason in REASON_ORDER:
        if reason not in pivot.columns:
            pivot[reason] = 0

    pivot = pivot[REASON_ORDER]

    ax = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5),
        color=[REASON_COLORS.get(r, "#999999") for r in REASON_ORDER],
        edgecolor="white",
    )

    plt.title("Dataset-wise groundedness reason distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Dataset")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(
        [REASON_LABELS.get(r, r) for r in REASON_ORDER],
        title="Reason",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    for container in ax.containers:
        ax.bar_label(container, label_type="center", fontsize=8)

    save_fig(save_dir / "02_dataset_reason_stacked.png")


def plot_strict_vs_soft_topk_reason(df: pd.DataFrame, save_dir: Path, top_k: int = 15):
    strict_top = df.sort_values("score_combined_strict", ascending=False).head(top_k)
    soft_top = df.sort_values("score_combined_soft_penalty", ascending=False).head(top_k)

    rows = []
    for strategy_name, subset in [
        ("Strict score", strict_top),
        ("SoftPenalty score", soft_top),
    ]:
        counter = Counter(subset["groundedness_reason"])
        for reason in REASON_ORDER:
            rows.append({
                "strategy": strategy_name,
                "reason": reason,
                "count": counter.get(reason, 0)
            })

    plot_df = pd.DataFrame(rows)
    pivot = plot_df.pivot(index="strategy", columns="reason", values="count").fillna(0)
    pivot = pivot[REASON_ORDER]

    ax = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5),
        color=[REASON_COLORS.get(r, "#999999") for r in REASON_ORDER],
        edgecolor="white",
    )

    plt.title(f"Top-{top_k} candidate composition: Strict vs SoftPenalty", fontsize=14, fontweight="bold")
    plt.xlabel("Acquisition score")
    plt.ylabel(f"Number of samples in Top-{top_k}")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(
        [REASON_LABELS.get(r, r) for r in REASON_ORDER],
        title="Reason",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    for container in ax.containers:
        ax.bar_label(container, label_type="center", fontsize=8)

    save_fig(save_dir / "03_strict_vs_soft_topk_reason.png")


def plot_priority_scatter(df: pd.DataFrame, save_dir: Path):
    plt.figure(figsize=(8, 6))

    for reason in REASON_ORDER:
        subset = df[df["groundedness_reason"] == reason]
        if len(subset) == 0:
            continue

        plt.scatter(
            subset["uncertainty_consistency"],
            subset["uncertainty_groundedness_soft"],
            s=70,
            alpha=0.75,
            label=REASON_LABELS.get(reason, reason),
            color=REASON_COLORS.get(reason, "#999999"),
            edgecolor="white",
            linewidth=0.5,
        )

    top5 = df.sort_values("score_combined_soft_penalty", ascending=False).head(5)

    for _, row in top5.iterrows():
        label = Path(row["image_name"]).stem[:16]
        plt.annotate(
            label,
            (row["uncertainty_consistency"], row["uncertainty_groundedness_soft"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    plt.title("Priority space: consistency uncertainty vs groundedness uncertainty", fontsize=14, fontweight="bold")
    plt.xlabel("Semantic uncertainty = 1 - consistency")
    plt.ylabel("Groundedness uncertainty (soft)")
    plt.grid(alpha=0.25)
    plt.legend(title="Reason", bbox_to_anchor=(1.02, 1), loc="upper left")

    save_fig(save_dir / "04_priority_scatter.png")


def plot_top_bottom_priority(df: pd.DataFrame, save_dir: Path, n: int = 10):
    top = df.sort_values("score_combined_soft_penalty", ascending=False).head(n)
    bottom = df.sort_values("score_combined_soft_penalty", ascending=True).head(n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, subset, title in [
        (axes[0], top.iloc[::-1], f"Top-{n} high-priority samples"),
        (axes[1], bottom.iloc[::-1], f"Bottom-{n} low-priority samples"),
    ]:
        labels = [
            f"{Path(name).stem[:18]}\n{reason}"
            for name, reason in zip(subset["image_name"], subset["groundedness_reason"])
        ]

        colors = [REASON_COLORS.get(r, "#999999") for r in subset["groundedness_reason"]]

        bars = ax.barh(
            labels,
            subset["score_combined_soft_penalty"],
            color=colors,
            edgecolor="white",
        )

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("SoftPenalty priority score")
        ax.grid(axis="x", alpha=0.25)

        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                va="center",
                fontsize=8,
            )

    plt.suptitle("Priority ranking examples", fontsize=15, fontweight="bold")
    save_fig(save_dir / "05_top_bottom_priority.png")


def plot_strict_vs_soft_shift(df: pd.DataFrame, save_dir: Path):
    plt.figure(figsize=(7, 7))

    for reason in REASON_ORDER:
        subset = df[df["groundedness_reason"] == reason]
        if len(subset) == 0:
            continue

        plt.scatter(
            subset["score_combined_strict"],
            subset["score_combined_soft_penalty"],
            s=65,
            alpha=0.75,
            label=REASON_LABELS.get(reason, reason),
            color=REASON_COLORS.get(reason, "#999999"),
            edgecolor="white",
            linewidth=0.5,
        )

    max_score = max(
        df["score_combined_strict"].max(),
        df["score_combined_soft_penalty"].max()
    )

    plt.plot([0, max_score], [0, max_score], linestyle="--", color="black", alpha=0.4)

    plt.title("Score shift after missing-evidence correction", fontsize=14, fontweight="bold")
    plt.xlabel("Strict combined score")
    plt.ylabel("SoftPenalty combined score")
    plt.grid(alpha=0.25)
    plt.legend(title="Reason", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.text(
        0.05,
        max_score * 0.92,
        "Below diagonal:\npriority reduced after\nmissing evidence correction",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    save_fig(save_dir / "06_strict_vs_soft_shift.png")


# =============================================================================
# [5] Dashboard
# =============================================================================
def plot_dashboard(df: pd.DataFrame, save_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "VLM-based active learning priority score analysis",
        fontsize=18,
        fontweight="bold",
        y=0.98
    )

    # Panel 1: reason distribution
    ax = axes[0, 0]
    counts = df["groundedness_reason"].value_counts()
    ordered = [r for r in REASON_ORDER if r in counts.index]
    values = [counts[r] for r in ordered]
    labels = [REASON_LABELS.get(r, r) for r in ordered]
    colors = [REASON_COLORS.get(r, "#999999") for r in ordered]

    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Groundedness reason counts", fontweight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value),
                ha="center", va="bottom", fontsize=9)

    # Panel 2: dataset reason stacked
    ax = axes[0, 1]
    pivot = (
        df.groupby(["dataset_type", "groundedness_reason"])
        .size()
        .unstack(fill_value=0)
    )
    for reason in REASON_ORDER:
        if reason not in pivot.columns:
            pivot[reason] = 0
    pivot = pivot[REASON_ORDER]

    bottom = np.zeros(len(pivot.index))
    x = np.arange(len(pivot.index))
    for reason in REASON_ORDER:
        vals = pivot[reason].values
        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=REASON_LABELS.get(reason, reason),
            color=REASON_COLORS.get(reason, "#999999"),
            edgecolor="white",
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_title("Reason by dataset", fontweight="bold")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)

    # Panel 3: strict vs soft Top-15
    ax = axes[0, 2]
    top_k = 15
    strict_top = df.sort_values("score_combined_strict", ascending=False).head(top_k)
    soft_top = df.sort_values("score_combined_soft_penalty", ascending=False).head(top_k)

    strategy_data = []
    for subset in [strict_top, soft_top]:
        counter = Counter(subset["groundedness_reason"])
        strategy_data.append([counter.get(reason, 0) for reason in REASON_ORDER])

    strategy_data = np.array(strategy_data)
    strategies = ["Strict", "SoftPenalty"]
    bottom = np.zeros(len(strategies))

    for idx, reason in enumerate(REASON_ORDER):
        ax.bar(
            strategies,
            strategy_data[:, idx],
            bottom=bottom,
            color=REASON_COLORS.get(reason, "#999999"),
            edgecolor="white",
            label=REASON_LABELS.get(reason, reason)
        )
        bottom += strategy_data[:, idx]

    ax.set_title(f"Top-{top_k} composition", fontweight="bold")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)

    # Panel 4: scatter
    ax = axes[1, 0]
    for reason in REASON_ORDER:
        subset = df[df["groundedness_reason"] == reason]
        if len(subset) == 0:
            continue
        ax.scatter(
            subset["uncertainty_consistency"],
            subset["uncertainty_groundedness_soft"],
            s=45,
            alpha=0.75,
            color=REASON_COLORS.get(reason, "#999999"),
            label=REASON_LABELS.get(reason, reason),
            edgecolor="white",
            linewidth=0.4,
        )

    ax.set_title("Priority space", fontweight="bold")
    ax.set_xlabel("1 - consistency")
    ax.set_ylabel("Groundedness uncertainty")
    ax.grid(alpha=0.25)

    # Panel 5: top 10 priority
    ax = axes[1, 1]
    top10 = df.sort_values("score_combined_soft_penalty", ascending=False).head(10).iloc[::-1]
    labels = [Path(x).stem[:14] for x in top10["image_name"]]
    colors = [REASON_COLORS.get(r, "#999999") for r in top10["groundedness_reason"]]
    ax.barh(labels, top10["score_combined_soft_penalty"], color=colors)
    ax.set_title("Top-10 high-priority samples", fontweight="bold")
    ax.set_xlabel("SoftPenalty score")
    ax.grid(axis="x", alpha=0.25)

    # Panel 6: strict vs soft shift
    ax = axes[1, 2]
    for reason in REASON_ORDER:
        subset = df[df["groundedness_reason"] == reason]
        if len(subset) == 0:
            continue
        ax.scatter(
            subset["score_combined_strict"],
            subset["score_combined_soft_penalty"],
            s=45,
            alpha=0.75,
            color=REASON_COLORS.get(reason, "#999999"),
            label=REASON_LABELS.get(reason, reason),
            edgecolor="white",
            linewidth=0.4,
        )

    max_score = max(df["score_combined_strict"].max(), df["score_combined_soft_penalty"].max())
    ax.plot([0, max_score], [0, max_score], linestyle="--", color="black", alpha=0.35)
    ax.set_title("Strict → SoftPenalty shift", fontweight="bold")
    ax.set_xlabel("Strict score")
    ax.set_ylabel("SoftPenalty score")
    ax.grid(alpha=0.25)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Reason",
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(save_dir / "notion_summary_dashboard.png", dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {save_dir / 'notion_summary_dashboard.png'}")


# =============================================================================
# [6] Markdown summary for Notion
# =============================================================================
def write_notion_interpretation(df: pd.DataFrame, summary: dict, save_dir: Path):
    reason_counts = Counter(df["groundedness_reason"])
    num_items = len(df)

    dataset_rows = []
    for ds, sub in df.groupby("dataset_type"):
        dataset_rows.append(
            f"- {ds}: {len(sub)} samples, "
            f"avg SoftPenalty score = {sub['score_combined_soft_penalty'].mean():.3f}, "
            f"no_pseudo_box = {(sub['groundedness_reason'] == 'no_pseudo_box').sum()}"
        )

    top10 = df.sort_values("score_combined_soft_penalty", ascending=False).head(10)
    top_reason_counts = Counter(top10["groundedness_reason"])

    bottom10 = df.sort_values("score_combined_soft_penalty", ascending=True).head(10)
    bottom_reason_counts = Counter(bottom10["groundedness_reason"])

    md = f"""# VLM Active Learning Priority Score 분석 요약

## 1. 실험 목적

본 실험의 목적은 GT bounding box를 사용하지 않고, VLM 설명의 일관성과 OVD 기반 pseudo box의 위치 근거성을 결합하여 Active Learning 샘플 우선순위를 계산하는 것이다.

기존 GT 기반 groundedness는 Active Learning의 unlabeled pool 조건에 맞지 않기 때문에, OWL-ViT 기반 pseudo box를 사용하여 label-free groundedness를 계산하였다.

---

## 2. 전체 결과 요약

- 전체 샘플 수: {num_items}
- no_pseudo_box: {reason_counts.get('no_pseudo_box', 0)}
- partial_match: {reason_counts.get('partial_match', 0)}
- location_mismatch: {reason_counts.get('location_mismatch', 0)}
- matched: {reason_counts.get('matched', 0)}

Dataset별 요약:

{chr(10).join(dataset_rows)}

---

## 3. 핵심 관찰

초기 strict score에서는 pseudo box가 없는 샘플을 groundedness 0으로 처리했기 때문에, no_pseudo_box 샘플이 상위 priority를 과도하게 차지하는 문제가 있었다.

이를 보완하기 위해 no_pseudo_box를 완전한 불일치가 아니라 missing evidence로 해석하고, groundedness_effective_soft = 0.5와 missing_box_penalty = 0.2를 적용하였다.

그 결과 Top-10 후보는 다음과 같이 구성되었다.

- no_pseudo_box: {top_reason_counts.get('no_pseudo_box', 0)}
- partial_match: {top_reason_counts.get('partial_match', 0)}
- location_mismatch: {top_reason_counts.get('location_mismatch', 0)}
- matched: {top_reason_counts.get('matched', 0)}

반대로 Bottom-10 후보는 다음과 같이 구성되었다.

- no_pseudo_box: {bottom_reason_counts.get('no_pseudo_box', 0)}
- partial_match: {bottom_reason_counts.get('partial_match', 0)}
- location_mismatch: {bottom_reason_counts.get('location_mismatch', 0)}
- matched: {bottom_reason_counts.get('matched', 0)}

---

## 4. 해석

상위 priority 샘플은 VLM 설명이 불안정하거나, pseudo box와 위치 근거가 불일치하거나, OVD가 결함 후보 영역을 찾지 못한 샘플들로 구성되었다.

하위 priority 샘플은 상대적으로 VLM 설명이 안정적이고, pseudo box와 위치/크기 근거가 잘 맞는 샘플로 구성되었다.

따라서 현재 score는 “설명 불안정성”과 “위치 근거성 부족”을 함께 반영하는 acquisition signal로 동작하고 있다고 볼 수 있다.

---

## 5. 다음 단계

다음 단계에서는 이 priority score를 기반으로 Active Learning 파일럿 실험을 수행한다.

비교 전략은 다음과 같이 설정한다.

1. Random
2. Consistency-only
3. Combined-strict
4. Combined-soft-penalty

핵심 비교는 Combined-strict와 Combined-soft-penalty이다. 이를 통해 missing pseudo evidence를 완전한 실패로 처리하는 방식과, 중립적 missing evidence로 처리하는 방식이 downstream YOLO detector 성능에 어떤 차이를 만드는지 확인한다.
"""

    save_path = save_dir / "notion_interpretation.md"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[SAVE] {save_path}")


# =============================================================================
# [7] Main
# =============================================================================
def main():
    set_korean_font()

    csv_file, summary_file, df, summary = load_data()
    df = prepare_dataframe(df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = csv_file.parent / f"priority_visualizations_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[*] Input priority CSV:")
    print(csv_file)
    print("[*] Input summary JSON:")
    print(summary_file)
    print("[*] Save directory:")
    print(save_dir)
    print("=" * 80)

    plot_reason_distribution(df, save_dir)
    plot_dataset_reason_stacked(df, save_dir)
    plot_strict_vs_soft_topk_reason(df, save_dir, top_k=15)
    plot_priority_scatter(df, save_dir)
    plot_top_bottom_priority(df, save_dir, n=10)
    plot_strict_vs_soft_shift(df, save_dir)
    plot_dashboard(df, save_dir)
    write_notion_interpretation(df, summary, save_dir)

    print("\n[완료] Notion용 priority analysis 시각화 생성 완료")
    print(save_dir)


if __name__ == "__main__":
    main()