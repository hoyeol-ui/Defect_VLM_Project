"""
===============================================================================
[File] inspect_acquisition_candidates.py

[Purpose]
Inspect which samples are selected by each acquisition strategy before running
real YOLO Active Learning.

This is a lightweight audit step.

Input:
    outputs/pseudo_boxes_*/priority_scores_pseudo.csv
    outputs/pseudo_boxes_*/pseudo_groundedness_results.json

Output:
    outputs/pseudo_boxes_*/acquisition_candidate_audit_YYYYMMDD_HHMMSS/
        selected_candidates_all_strategies.csv
        selected_candidates_<strategy>.csv
        01_strategy_reason_composition.png
        02_strategy_dataset_composition.png
        03_priority_space_selected.png
        montage_<strategy>.png
        notion_acquisition_audit.md

Strategies:
    1. Random
    2. ConsistencyOnly
    3. CombinedStrict
    4. CombinedSoftPenalty
    5. SoftPenaltyHybrid50
    6. LowPrioritySoft

Why this file is needed:
    The score-level dashboard showed that SoftPenalty improved candidate
    distribution. Before expensive YOLO training, we check whether the selected
    images are qualitatively and distributionally reasonable.
===============================================================================
"""

import json
import random
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont


# =============================================================================
# [0] Paths and settings
# =============================================================================
PROJECT_ROOT = Path("/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project")
OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs"

TOP_K = 15
RANDOM_SEED = 42

REASON_ORDER = [
    "location_mismatch",
    "partial_match",
    "no_pseudo_box",
    "matched",
]

REASON_LABELS = {
    "location_mismatch": "Location mismatch",
    "partial_match": "Partial match",
    "no_pseudo_box": "No pseudo box",
    "matched": "Matched",
}

REASON_COLORS = {
    "location_mismatch": "#E76F51",
    "partial_match": "#5B8FF9",
    "no_pseudo_box": "#A0A0A0",
    "matched": "#2A9D8F",
}

DATASET_COLORS = {
    "NEU-DET": "#5B8FF9",
    "GC10-DET": "#F6BD16",
}


# =============================================================================
# [1] Font setup
# =============================================================================
def set_korean_font():
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


def get_pil_font(size=16):
    candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]

    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass

    return ImageFont.load_default()


# =============================================================================
# [2] File loading
# =============================================================================
def find_latest_file(pattern: str) -> Path:
    files = list(OUTPUT_BASE_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return max(files, key=lambda p: p.stat().st_mtime)


def load_inputs():
    priority_csv = find_latest_file("pseudo_boxes_*/priority_scores_pseudo.csv")
    output_dir = priority_csv.parent

    grounded_json = output_dir / "pseudo_groundedness_results.json"

    if not grounded_json.exists():
        raise FileNotFoundError(f"pseudo_groundedness_results.json not found: {grounded_json}")

    df = pd.read_csv(priority_csv)

    with open(grounded_json, "r", encoding="utf-8") as f:
        grounded_data = json.load(f)

    grounded_index = {
        item["image_name"]: item
        for item in grounded_data
        if "image_name" in item
    }

    return priority_csv, grounded_json, df, grounded_index


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
        "score_consistency_only",
        "score_combined_strict",
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
# [3] Strategy selection
# =============================================================================
def select_random(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    return df.sample(n=min(top_k, len(df)), random_state=RANDOM_SEED)


def select_consistency_only(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    return df.sort_values("score_consistency_only", ascending=False).head(top_k)


def select_combined_strict(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    return df.sort_values("score_combined_strict", ascending=False).head(top_k)


def select_combined_soft_penalty(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    return df.sort_values("score_combined_soft_penalty", ascending=False).head(top_k)


def select_soft_penalty_hybrid50(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """
    50% hard samples from SoftPenalty score + 50% random samples.

    This is safer for small active learning pools because it prevents one score
    from over-dominating the selected batch.
    """
    n_hard = top_k // 2
    n_random = top_k - n_hard

    hard = df.sort_values("score_combined_soft_penalty", ascending=False).head(n_hard)
    remaining = df.drop(hard.index)

    if len(remaining) > 0 and n_random > 0:
        rand = remaining.sample(n=min(n_random, len(remaining)), random_state=RANDOM_SEED)
        selected = pd.concat([hard, rand])
    else:
        selected = hard

    return selected


def select_low_priority_soft(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """
    Sanity check group.
    These are expected to be easier / more grounded samples.
    """
    return df.sort_values("score_combined_soft_penalty", ascending=True).head(top_k)


def build_strategy_selections(df: pd.DataFrame, top_k: int) -> dict:
    return {
        "Random": select_random(df, top_k),
        "ConsistencyOnly": select_consistency_only(df, top_k),
        "CombinedStrict": select_combined_strict(df, top_k),
        "CombinedSoftPenalty": select_combined_soft_penalty(df, top_k),
        "SoftPenaltyHybrid50": select_soft_penalty_hybrid50(df, top_k),
        "LowPrioritySoft": select_low_priority_soft(df, top_k),
    }


# =============================================================================
# [4] Visualization utilities
# =============================================================================
def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {path}")


def plot_reason_composition(strategy_dict: dict, save_dir: Path):
    rows = []

    for strategy, subset in strategy_dict.items():
        counter = Counter(subset["groundedness_reason"])

        for reason in REASON_ORDER:
            rows.append({
                "strategy": strategy,
                "reason": reason,
                "count": counter.get(reason, 0)
            })

    plot_df = pd.DataFrame(rows)
    pivot = plot_df.pivot(index="strategy", columns="reason", values="count").fillna(0)
    pivot = pivot[REASON_ORDER]

    ax = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(11, 6),
        color=[REASON_COLORS.get(r, "#999999") for r in REASON_ORDER],
        edgecolor="white",
    )

    plt.title(f"Top-{TOP_K} selected candidate composition by strategy", fontsize=15, fontweight="bold")
    plt.xlabel("Acquisition strategy")
    plt.ylabel(f"Number of samples in Top-{TOP_K}")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(
        [REASON_LABELS.get(r, r) for r in REASON_ORDER],
        title="Groundedness reason",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    for container in ax.containers:
        ax.bar_label(container, label_type="center", fontsize=8)

    save_fig(save_dir / "01_strategy_reason_composition.png")


def plot_dataset_composition(strategy_dict: dict, save_dir: Path):
    rows = []

    for strategy, subset in strategy_dict.items():
        counter = Counter(subset["dataset_type"])

        for dataset in sorted(counter.keys()):
            rows.append({
                "strategy": strategy,
                "dataset_type": dataset,
                "count": counter[dataset]
            })

    plot_df = pd.DataFrame(rows)
    pivot = plot_df.pivot(index="strategy", columns="dataset_type", values="count").fillna(0)

    colors = [
        DATASET_COLORS.get(col, "#999999")
        for col in pivot.columns
    ]

    ax = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 5),
        color=colors,
        edgecolor="white",
    )

    plt.title(f"Top-{TOP_K} dataset composition by strategy", fontsize=15, fontweight="bold")
    plt.xlabel("Acquisition strategy")
    plt.ylabel(f"Number of samples in Top-{TOP_K}")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left")

    for container in ax.containers:
        ax.bar_label(container, label_type="center", fontsize=8)

    save_fig(save_dir / "02_strategy_dataset_composition.png")


def plot_priority_space_selected(df: pd.DataFrame, strategy_dict: dict, save_dir: Path):
    """
    Highlight CombinedSoftPenalty and Hybrid selected samples on priority space.
    """
    plt.figure(figsize=(9, 7))

    # background
    plt.scatter(
        df["uncertainty_consistency"],
        df["uncertainty_groundedness_soft"],
        s=45,
        alpha=0.25,
        color="#999999",
        label="All samples",
    )

    selected_soft = strategy_dict["CombinedSoftPenalty"]
    selected_hybrid = strategy_dict["SoftPenaltyHybrid50"]

    plt.scatter(
        selected_soft["uncertainty_consistency"],
        selected_soft["uncertainty_groundedness_soft"],
        s=90,
        alpha=0.85,
        marker="o",
        label="CombinedSoftPenalty Top-K",
        edgecolor="black",
        linewidth=0.8,
    )

    plt.scatter(
        selected_hybrid["uncertainty_consistency"],
        selected_hybrid["uncertainty_groundedness_soft"],
        s=100,
        alpha=0.85,
        marker="x",
        label="SoftPenaltyHybrid50 selected",
        linewidth=1.6,
    )

    plt.title("Selected samples in priority space", fontsize=15, fontweight="bold")
    plt.xlabel("Semantic uncertainty = 1 - consistency")
    plt.ylabel("Groundedness uncertainty (soft)")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")

    save_fig(save_dir / "03_priority_space_selected.png")


# =============================================================================
# [5] Image montage
# =============================================================================
def get_matched_box(grounded_index: dict, image_name: str):
    item = grounded_index.get(image_name)
    if not item:
        return None

    box = item.get("pseudo_matched_box")
    if isinstance(box, dict):
        return box

    return None


def draw_text_box(draw, xy, text, font, fill=(255, 255, 255), bg=(0, 0, 0)):
    x, y = xy

    try:
        bbox = draw.textbbox((x, y), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        text_w = len(text) * 8
        text_h = 14

    draw.rectangle([x, y, x + text_w + 6, y + text_h + 6], fill=bg)
    draw.text((x + 3, y + 3), text, fill=fill, font=font)


def create_strategy_montage(
    strategy_name: str,
    subset: pd.DataFrame,
    grounded_index: dict,
    save_dir: Path,
    thumb_w: int = 280,
    thumb_h: int = 220,
    cols: int = 5
):
    rows = int(np.ceil(len(subset) / cols))
    canvas_w = cols * thumb_w
    canvas_h = rows * thumb_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 245))

    title_font = get_pil_font(size=14)
    small_font = get_pil_font(size=11)

    for i, (_, row) in enumerate(subset.iterrows()):
        col = i % cols
        r = i // cols

        x0 = col * thumb_w
        y0 = r * thumb_h

        image_path = Path(str(row["image_path"]))

        if not image_path.exists():
            tile = Image.new("RGB", (thumb_w, thumb_h), color=(220, 220, 220))
            d = ImageDraw.Draw(tile)
            d.text((10, 10), f"Image not found\n{row['image_name']}", fill=(0, 0, 0), font=small_font)
            canvas.paste(tile, (x0, y0))
            continue

        img = Image.open(image_path).convert("RGB")
        original_w, original_h = img.size

        # Resize keeping aspect ratio
        img.thumbnail((thumb_w, thumb_h - 60))
        tile = Image.new("RGB", (thumb_w, thumb_h), color=(255, 255, 255))

        paste_x = (thumb_w - img.width) // 2
        paste_y = 5
        tile.paste(img, (paste_x, paste_y))

        draw = ImageDraw.Draw(tile)

        # Draw pseudo matched box if available
        matched_box = get_matched_box(grounded_index, row["image_name"])
        if matched_box and matched_box.get("xyxy"):
            xyxy = matched_box["xyxy"]
            scale_x = img.width / original_w
            scale_y = img.height / original_h

            bx1 = paste_x + int(float(xyxy[0]) * scale_x)
            by1 = paste_y + int(float(xyxy[1]) * scale_y)
            bx2 = paste_x + int(float(xyxy[2]) * scale_x)
            by2 = paste_y + int(float(xyxy[3]) * scale_y)

            draw.rectangle([bx1, by1, bx2, by2], outline=(255, 0, 0), width=2)

        # Metadata text
        reason = str(row["groundedness_reason"])
        score = float(row["score_combined_soft_penalty"])
        consistency = float(row["consistency_score"])
        g_soft = float(row["groundedness_effective_soft"])

        line1 = f"{i + 1}. {Path(row['image_name']).stem[:22]}"
        line2 = f"{row['dataset_type']} | {reason}"
        line3 = f"S={score:.3f}, C={consistency:.3f}, G={g_soft:.2f}"

        text_y = thumb_h - 52

        reason_color = REASON_COLORS.get(reason, "#777777")
        # hex to rgb
        reason_rgb = tuple(int(reason_color.lstrip("#")[j:j+2], 16) for j in (0, 2, 4))

        draw.rectangle([0, thumb_h - 58, thumb_w, thumb_h], fill=(250, 250, 250))
        draw.rectangle([0, thumb_h - 58, 8, thumb_h], fill=reason_rgb)

        draw.text((12, text_y), line1, fill=(0, 0, 0), font=title_font)
        draw.text((12, text_y + 18), line2, fill=(40, 40, 40), font=small_font)
        draw.text((12, text_y + 34), line3, fill=(40, 40, 40), font=small_font)

        canvas.paste(tile, (x0, y0))

    save_path = save_dir / f"montage_{strategy_name}.png"
    canvas.save(save_path)
    print(f"[SAVE] {save_path}")


# =============================================================================
# [6] Save CSV and Markdown
# =============================================================================
def save_strategy_csvs(strategy_dict: dict, save_dir: Path):
    all_rows = []

    for strategy, subset in strategy_dict.items():
        out = subset.copy()
        out.insert(0, "strategy", strategy)
        out.insert(1, "rank_in_strategy", range(1, len(out) + 1))

        out_file = save_dir / f"selected_candidates_{strategy}.csv"
        out.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {out_file}")

        all_rows.append(out)

    all_df = pd.concat(all_rows, ignore_index=True)
    all_file = save_dir / "selected_candidates_all_strategies.csv"
    all_df.to_csv(all_file, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {all_file}")


def summarize_strategy(strategy: str, subset: pd.DataFrame) -> str:
    reason_counter = Counter(subset["groundedness_reason"])
    dataset_counter = Counter(subset["dataset_type"])

    avg_score = subset["score_combined_soft_penalty"].mean()
    avg_consistency = subset["consistency_score"].mean()
    avg_g = subset["groundedness_effective_soft"].mean()

    lines = [
        f"### {strategy}",
        "",
        f"- selected samples: {len(subset)}",
        f"- avg SoftPenalty score: {avg_score:.3f}",
        f"- avg consistency: {avg_consistency:.3f}",
        f"- avg groundedness_effective_soft: {avg_g:.3f}",
        f"- dataset distribution: {dict(dataset_counter)}",
        f"- reason distribution: {dict(reason_counter)}",
        "",
    ]

    return "\n".join(lines)


def write_notion_markdown(strategy_dict: dict, save_dir: Path):
    soft = strategy_dict["CombinedSoftPenalty"]
    hybrid = strategy_dict["SoftPenaltyHybrid50"]
    strict = strategy_dict["CombinedStrict"]

    soft_reason = Counter(soft["groundedness_reason"])
    strict_reason = Counter(strict["groundedness_reason"])
    hybrid_reason = Counter(hybrid["groundedness_reason"])

    md = f"""# Acquisition Candidate Audit

## 1. 목적

이 단계는 실제 YOLO Active Learning 학습 전에, 각 acquisition strategy가 어떤 샘플을 선택하는지 점검하기 위한 사전 검증이다.

이 검증을 통해 score가 특정 reason 또는 특정 dataset에 과도하게 편향되어 있는지 확인한다.

---

## 2. 비교한 전략

1. Random
2. ConsistencyOnly
3. CombinedStrict
4. CombinedSoftPenalty
5. SoftPenaltyHybrid50
6. LowPrioritySoft

---

## 3. 핵심 비교

### CombinedStrict

CombinedStrict는 pseudo box가 없는 샘플을 groundedness 실패로 강하게 처리한다.

- reason distribution: {dict(strict_reason)}

### CombinedSoftPenalty

CombinedSoftPenalty는 no_pseudo_box를 완전한 실패가 아니라 missing evidence로 처리한다.

- reason distribution: {dict(soft_reason)}

### SoftPenaltyHybrid50

SoftPenaltyHybrid50은 SoftPenalty score 상위 샘플 50%와 random 샘플 50%를 섞는 전략이다.

- reason distribution: {dict(hybrid_reason)}

---

## 4. 해석

CombinedSoftPenalty는 strict 방식보다 no_pseudo_box 편향을 완화하면서도, location_mismatch와 partial_match를 함께 우선 후보로 포함한다.

다만 전체 pool에서 no_pseudo_box 비율이 높기 때문에, 실제 Active Learning 학습에서는 SoftPenalty score만 100% 사용하는 것보다 SoftPenaltyHybrid50 전략이 더 안정적일 가능성이 있다.

---

## 5. 다음 실행 판단 기준

다음 조건이 만족되면 YOLO Active Learning 파일럿으로 진행한다.

- CombinedSoftPenalty 또는 Hybrid 후보가 한 reason에만 과도하게 몰리지 않는다.
- NEU-DET와 GC10-DET 중 한쪽 dataset에만 과도하게 치우치지 않는다.
- Top candidate montage에서 실제로 난이도가 높거나 근거 불일치가 있는 샘플이 보인다.

---

## 6. Strategy summaries

"""

    for strategy, subset in strategy_dict.items():
        md += summarize_strategy(strategy, subset)

    save_path = save_dir / "notion_acquisition_audit.md"

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[SAVE] {save_path}")


# =============================================================================
# [7] Main
# =============================================================================
def main():
    set_korean_font()

    priority_csv, grounded_json, df, grounded_index = load_inputs()
    df = prepare_dataframe(df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = priority_csv.parent / f"acquisition_candidate_audit_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[*] Priority CSV:")
    print(priority_csv)
    print("[*] Pseudo groundedness JSON:")
    print(grounded_json)
    print("[*] Save directory:")
    print(save_dir)
    print("=" * 80)

    strategy_dict = build_strategy_selections(df, TOP_K)

    save_strategy_csvs(strategy_dict, save_dir)
    plot_reason_composition(strategy_dict, save_dir)
    plot_dataset_composition(strategy_dict, save_dir)
    plot_priority_space_selected(df, strategy_dict, save_dir)

    for strategy, subset in strategy_dict.items():
        create_strategy_montage(
            strategy_name=strategy,
            subset=subset,
            grounded_index=grounded_index,
            save_dir=save_dir,
            thumb_w=280,
            thumb_h=220,
            cols=5
        )

    write_notion_markdown(strategy_dict, save_dir)

    print("\n[완료] Acquisition candidate audit 생성 완료")
    print(save_dir)


if __name__ == "__main__":
    main()