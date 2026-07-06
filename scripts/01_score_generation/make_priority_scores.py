"""
References
----------
1. Grounding DINO
   Official repo: https://github.com/IDEA-Research/GroundingDINO
   Used concept: text-conditioned open-vocabulary pseudo bounding box generation.

2. OWL-ViT
   Hugging Face docs: https://huggingface.co/docs/transformers/model_doc/owlvit
   Used concept: open-vocabulary object detection with text queries.

3. YOLO-World
   Official repo: https://github.com/AILab-CVC/YOLO-World
   Used concept: real-time open-vocabulary object detection candidate.

4. CALD
   Official repo: https://github.com/we1pingyu/CALD
   Used concept: consistency as active learning acquisition signal for object detection.

5. Ultralytics YOLO
   Official repo: https://github.com/ultralytics/ultralytics
   Used concept: downstream detector training and mAP-based active learning evaluation.
"""

import json
import csv
from pathlib import Path
from collections import Counter, defaultdict


PROJECT_ROOT = Path("/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project")
OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs"


def find_latest_file(pattern: str) -> Path:
    files = list(OUTPUT_BASE_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return max(files, key=lambda p: p.stat().st_mtime)


def safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def clamp(x, low=0.0, high=1.0):
    return max(low, min(high, x))


def compute_priority_score(
    consistency: float,
    groundedness: float,
    alpha: float = 1.0,
    beta: float = 1.0
) -> float:
    """
    Priority = alpha * (1 - Consistency) + beta * (1 - Groundedness)
    """
    c = clamp(consistency)
    g = clamp(groundedness)

    return round(alpha * (1.0 - c) + beta * (1.0 - g), 6)


def get_effective_groundedness(item: dict, missing_value: float = 0.5) -> float:
    """
    Convert pseudo groundedness result into acquisition-ready groundedness.

    If groundedness_norm is None, it usually means pseudo evidence is missing
    or not comparable. This should not be treated as a complete mismatch.

    Therefore:
        groundedness_norm is None -> missing_value

    Recommended:
        missing_value = 0.5
    """
    g_norm = safe_float(item.get("groundedness_norm"), default=None)

    if g_norm is None:
        return missing_value

    return clamp(g_norm)


def make_row(item: dict) -> dict:
    consistency = safe_float(item.get("consistency_score"), default=0.0)

    # 기존 strict 방식: 참고용으로만 유지
    g_strict = 0.0 if item.get("groundedness_norm") is None else clamp(float(item.get("groundedness_norm")))

    # 수정 방식: no_pseudo_box는 0이 아니라 0.5로 중립 처리
    g_soft = get_effective_groundedness(item, missing_value=0.5)

    is_missing_box = item.get("groundedness_reason") == "no_pseudo_box"
    missing_box_penalty = 0.2 if is_missing_box else 0.0

    u_consistency = round(1.0 - consistency, 6)
    u_groundedness_strict = round(1.0 - g_strict, 6)
    u_groundedness_soft = round(1.0 - g_soft, 6)

    score_consistency_only = round(u_consistency, 6)

    score_groundedness_strict = round(u_groundedness_strict, 6)

    score_combined_strict = round(
        u_consistency + u_groundedness_strict,
        6
    )

    score_combined_soft_missing = round(
        u_consistency + u_groundedness_soft,
        6
    )

    score_combined_soft_penalty = round(
        u_consistency + u_groundedness_soft + missing_box_penalty,
        6
    )

    return {
        "image_name": item.get("image_name"),
        "dataset_type": item.get("dataset_type"),
        "image_path": item.get("image_path"),

        "consistency_score": consistency,
        "groundedness_norm": item.get("groundedness_norm"),

        # 기존 방식
        "groundedness_effective_strict": g_strict,

        # 새 방식
        "groundedness_effective_soft": g_soft,
        "missing_box_penalty": missing_box_penalty,

        "groundedness_gt_oracle": item.get("groundedness_gt_oracle"),
        "groundedness_reason": item.get("groundedness_reason"),

        "pseudo_box_found": item.get("pseudo_box_found"),
        "pseudo_box_count": item.get("pseudo_box_count"),

        "best_box_score": item.get("best_box_score"),
        "best_box_quality": item.get("best_box_quality"),
        "best_box_area_ratio": item.get("best_box_area_ratio"),
        "best_box_aspect_ratio": item.get("best_box_aspect_ratio"),
        "best_box_spatial_zone": item.get("best_box_spatial_zone"),
        "best_box_scale_category": item.get("best_box_scale_category"),
        "best_box_prompt": item.get("best_box_prompt"),

        "uncertainty_consistency": u_consistency,
        "uncertainty_groundedness_strict": u_groundedness_strict,
        "uncertainty_groundedness_soft": u_groundedness_soft,

        "score_consistency_only": score_consistency_only,
        "score_groundedness_strict": score_groundedness_strict,

        # 기존 문제 있는 점수: 비교용
        "score_combined_strict": score_combined_strict,

        # 새 후보 점수 1
        "score_combined_soft_missing": score_combined_soft_missing,

        # 새 후보 점수 2: 추천
        "score_combined_soft_penalty": score_combined_soft_penalty,
    }


def summarize(rows):
    reason_counter = Counter(r["groundedness_reason"] for r in rows)
    dataset_counter = Counter(r["dataset_type"] for r in rows)

    # 새 추천 점수 기준
    score_key = "score_combined_soft_penalty"

    top_combined = sorted(
        rows,
        key=lambda x: x[score_key],
        reverse=True
    )[:10]

    bottom_combined = sorted(
        rows,
        key=lambda x: x[score_key]
    )[:10]

    dataset_summary = {}

    for ds in dataset_counter:
        ds_rows = [r for r in rows if r["dataset_type"] == ds]

        dataset_summary[ds] = {
            "count": len(ds_rows),
            "avg_score_combined_soft_penalty": round(
                sum(r[score_key] for r in ds_rows) / len(ds_rows),
                6
            ),
            "avg_consistency_uncertainty": round(
                sum(r["uncertainty_consistency"] for r in ds_rows) / len(ds_rows),
                6
            ),
            "avg_groundedness_uncertainty_soft": round(
                sum(r["uncertainty_groundedness_soft"] for r in ds_rows) / len(ds_rows),
                6
            ),
            "avg_missing_box_penalty": round(
                sum(r["missing_box_penalty"] for r in ds_rows) / len(ds_rows),
                6
            ),
            "reason_counts": dict(Counter(r["groundedness_reason"] for r in ds_rows))
        }

    return {
        "num_items": len(rows),
        "score_key": score_key,
        "reason_counts": dict(reason_counter),
        "dataset_summary": dataset_summary,
        "top10_combined": [
            {
                "image_name": r["image_name"],
                "dataset_type": r["dataset_type"],
                "score": r[score_key],
                "consistency": r["consistency_score"],
                "g_soft": r["groundedness_effective_soft"],
                "missing_box_penalty": r["missing_box_penalty"],
                "reason": r["groundedness_reason"]
            }
            for r in top_combined
        ],
        "bottom10_combined": [
            {
                "image_name": r["image_name"],
                "dataset_type": r["dataset_type"],
                "score": r[score_key],
                "consistency": r["consistency_score"],
                "g_soft": r["groundedness_effective_soft"],
                "missing_box_penalty": r["missing_box_penalty"],
                "reason": r["groundedness_reason"]
            }
            for r in bottom_combined
        ]
    }


def main():
    input_file = find_latest_file("pseudo_boxes_*/pseudo_groundedness_results.json")

    print("=" * 80)
    print("[*] Input pseudo groundedness file:")
    print(input_file)
    print("=" * 80)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = [make_row(item) for item in data]

    # combined score 기준 내림차순 정렬
    rows = sorted(
        rows,
        key=lambda x: x["score_combined_soft_penalty"],
        reverse=True
    )

    output_dir = input_file.parent
    output_csv = output_dir / "priority_scores_pseudo.csv"
    output_json = output_dir / "priority_scores_pseudo.json"
    output_summary = output_dir / "priority_scores_summary.json"

    fieldnames = list(rows[0].keys())

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=4)

    summary = summarize(rows)

    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print("[완료] Priority score files saved:")
    print(f"CSV    : {output_csv}")
    print(f"JSON   : {output_json}")
    print(f"SUMMARY: {output_summary}")

    print("\n[TOP 10 combined priority]")
    for r in summary["top10_combined"]:
        print(r)

    print("\n[BOTTOM 10 combined priority]")
    for r in summary["bottom10_combined"]:
        print(r)


if __name__ == "__main__":
    main()