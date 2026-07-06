"""
===============================================================================
[File] compute_pseudo_groundedness.py

[Purpose]
Compute label-free pseudo groundedness by comparing:

1) VLM textual responses
   - location description
   - scale description
   - general defect description

2) OVD-generated pseudo bounding boxes
   - spatial_zone
   - scale_category
   - box_quality
   - area_ratio

[Why this file exists]
The previous groundedness implementation used GT VOC XML bounding boxes.
That violates the active learning assumption because the unlabeled pool should
not expose GT boxes.

This script uses pseudo boxes from OWL-ViT / Grounding DINO / YOLO-World-style
open-vocabulary detectors instead.

[Key Improvements]
This version separates the reasons for low groundedness:

- no_pseudo_box
- no_comparable_vlm_terms
- location_mismatch
- scale_mismatch
- partial_match
- matched

This prevents all zero scores from being interpreted in the same way.
===============================================================================
"""

import json
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# =============================================================================
# [0] Project path setup
# =============================================================================
PROJECT_ROOT = Path("/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project")
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))


# =============================================================================
# [1] File search utilities
# =============================================================================
def find_latest_file(pattern: str) -> Path:
    files = list(OUTPUT_BASE_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return max(files, key=lambda p: p.stat().st_mtime)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# =============================================================================
# [2] VLM result parsing
# =============================================================================
def get_image_name(item: Dict) -> str:
    """
    Handle different possible key names.
    """
    return (
        item.get("image_name")
        or item.get("image_id")
        or item.get("filename")
        or Path(item.get("image_path", "")).name
    )


def get_consistency_score(item: Dict) -> Optional[float]:
    """
    Robustly extract consistency score from previous result formats.
    """
    candidates = [
        item.get("consistency_score"),
        item.get("consistency"),
        item.get("semantic_consistency"),
        item.get("sbert_consistency"),
    ]

    for c in candidates:
        if c is not None:
            try:
                return float(c)
            except Exception:
                pass

    return None


def get_gt_oracle_groundedness(item: Dict) -> Optional[float]:
    """
    Extract GT-oracle groundedness if available.
    This is used only for comparison/diagnosis, not for acquisition scoring.
    """
    if "groundedness_gt_oracle" in item:
        return item.get("groundedness_gt_oracle")

    if "groundedness" in item and isinstance(item["groundedness"], dict):
        g = item["groundedness"]
        for key in ["total_score", "score", "groundedness_score"]:
            if key in g:
                return g[key]

    for key in ["groundedness_score", "gt_groundedness", "groundedness_gt"]:
        if key in item:
            return item[key]

    return None


def collect_vlm_text(item: Dict) -> Dict[str, str]:
    """
    Collect VLM responses from multiple possible result formats.

    Expected old format example:
        responses = {
            "location": "...",
            "scale": "...",
            "appearance": "..."
        }

    But this function also handles:
        response_1, response_2, ...
        descriptions list
        plain response string
    """
    texts = {
        "location": "",
        "scale": "",
        "appearance": "",
        "all": "",
    }

    responses = item.get("responses", None)

    if isinstance(responses, dict):
        for k, v in responses.items():
            if v is None:
                continue

            key = str(k).lower()
            val = str(v)

            if "location" in key or "position" in key or "where" in key:
                texts["location"] += " " + val
            elif "scale" in key or "size" in key:
                texts["scale"] += " " + val
            elif "appearance" in key or "shape" in key or "texture" in key:
                texts["appearance"] += " " + val

            texts["all"] += " " + val

    elif isinstance(responses, list):
        for v in responses:
            if v is not None:
                texts["all"] += " " + str(v)

    # fallback keys
    for key in ["response", "description", "caption", "vlm_response"]:
        if key in item and item[key] is not None:
            texts["all"] += " " + str(item[key])

    for key in ["response_1", "response_2", "response_3", "response_4"]:
        if key in item and item[key] is not None:
            texts["all"] += " " + str(item[key])

    # If specific fields are empty, use all text as weak fallback
    if not texts["location"]:
        texts["location"] = texts["all"]
    if not texts["scale"]:
        texts["scale"] = texts["all"]
    if not texts["appearance"]:
        texts["appearance"] = texts["all"]

    return {k: v.strip() for k, v in texts.items()}


# =============================================================================
# [3] Text term extraction
# =============================================================================
def extract_location_terms(text: str) -> List[str]:
    """
    Extract coarse spatial terms from VLM text.

    Returns terms among:
        top, bottom, left, right, center

    Notes:
        This is intentionally simple and interpretable.
        It can be replaced later by LLM-based parsing or regex expansion.
    """
    if not text:
        return []

    t = text.lower()

    terms = set()

    top_words = [
        "top", "upper", "upward", "above", "near the top",
        "upper-left", "upper right", "upper-right"
    ]
    bottom_words = [
        "bottom", "lower", "downward", "below", "near the bottom",
        "lower-left", "lower right", "lower-right"
    ]
    left_words = [
        "left", "left side", "left-side", "left region",
        "upper-left", "lower-left"
    ]
    right_words = [
        "right", "right side", "right-side", "right region",
        "upper-right", "lower-right"
    ]
    center_words = [
        "center", "central", "middle", "mid", "middle area",
        "central region"
    ]

    if any(w in t for w in top_words):
        terms.add("top")
    if any(w in t for w in bottom_words):
        terms.add("bottom")
    if any(w in t for w in left_words):
        terms.add("left")
    if any(w in t for w in right_words):
        terms.add("right")
    if any(w in t for w in center_words):
        terms.add("center")

    return sorted(list(terms))


def extract_scale_term(text: str) -> Optional[str]:
    """
    Extract scale category from VLM text.

    Returns:
        micro / small / large / None

    Current downstream pseudo boxes have:
        micro, small, large

    We keep this simple for interpretability.
    """
    if not text:
        return None

    t = text.lower()

    micro_words = [
        "tiny", "very small", "minute", "fine", "thin", "narrow",
        "hairline", "small dot", "speck"
    ]
    small_words = [
        "small", "localized", "minor", "short", "compact",
        "limited area", "small region"
    ]
    large_words = [
        "large", "wide", "broad", "long", "extended", "extensive",
        "large area", "large region", "across", "spanning"
    ]

    if any(w in t for w in micro_words):
        return "micro"
    if any(w in t for w in small_words):
        return "small"
    if any(w in t for w in large_words):
        return "large"

    return None


# =============================================================================
# [4] Matching logic
# =============================================================================
def zone_to_terms(zone: str) -> List[str]:
    """
    Convert pseudo bbox spatial_zone into comparable terms.
    Examples:
        top-right -> [top, right]
        center-left -> [center, left]
        center -> [center]
    """
    if not zone:
        return []

    zone = zone.lower()

    if zone == "center":
        return ["center"]

    return [z for z in zone.split("-") if z]


def compute_location_score(
    vlm_terms: List[str],
    pseudo_zone: str
) -> Optional[float]:
    """
    Return location match score:
        None: VLM did not provide location terms
        0.0 : mismatch
        0.5 : partial match
        1.0 : full match

    Example:
        VLM: ["left"], pseudo: "center-left" -> 1.0
        VLM: ["top", "left"], pseudo: "center-left" -> 0.5
        VLM: ["bottom"], pseudo: "top-right" -> 0.0
    """
    if not vlm_terms:
        return None

    pseudo_terms = zone_to_terms(pseudo_zone)
    if not pseudo_terms:
        return 0.0

    matched = 0
    for term in vlm_terms:
        if term in pseudo_terms:
            matched += 1

    return round(matched / len(vlm_terms), 4)


def compute_scale_score(
    vlm_scale: Optional[str],
    pseudo_scale: str
) -> Optional[float]:
    """
    Return scale match score:
        None: VLM did not provide scale expression
        1.0 : match
        0.5 : adjacent/compatible
        0.0 : mismatch

    Compatible examples:
        VLM small vs pseudo micro -> 0.5
        VLM micro vs pseudo small -> 0.5
    """
    if vlm_scale is None:
        return None

    if not pseudo_scale:
        return 0.0

    vlm_scale = vlm_scale.lower()
    pseudo_scale = pseudo_scale.lower()

    if vlm_scale == pseudo_scale:
        return 1.0

    compatible = {
        ("micro", "small"),
        ("small", "micro"),
        ("small", "large"),
        ("large", "small"),
    }

    if (vlm_scale, pseudo_scale) in compatible:
        return 0.5

    return 0.0


def choose_best_pseudo_box(
    pseudo_boxes: List[Dict],
    vlm_location_terms: List[str],
    vlm_scale_term: Optional[str]
) -> Tuple[Optional[Dict], Dict]:
    """
    Choose the best pseudo box for groundedness comparison.

    Priority:
        1. highest normalized match score
        2. highest box_quality
        3. highest OVD score

    If VLM has no comparable terms, choose the highest quality box but mark
    groundedness as not comparable.
    """
    if not pseudo_boxes:
        return None, {
            "location_score": None,
            "scale_score": None,
            "available_dims": 0,
            "groundedness_raw": 0.0,
            "groundedness_norm": None,
        }

    best_box = None
    best_info = None
    best_rank = (-1.0, -1.0, -1.0)

    for box in pseudo_boxes:
        loc_score = compute_location_score(
            vlm_terms=vlm_location_terms,
            pseudo_zone=box.get("spatial_zone", "")
        )

        scale_score = compute_scale_score(
            vlm_scale=vlm_scale_term,
            pseudo_scale=box.get("scale_category", "")
        )

        scores = []
        if loc_score is not None:
            scores.append(loc_score)
        if scale_score is not None:
            scores.append(scale_score)

        available_dims = len(scores)
        raw = round(sum(scores), 4) if scores else 0.0
        norm = round(raw / available_dims, 4) if available_dims > 0 else None

        box_quality = float(box.get("box_quality", 0.0))
        ovd_score = float(box.get("score", 0.0))

        rank = (
            norm if norm is not None else -1.0,
            box_quality,
            ovd_score,
        )

        if rank > best_rank:
            best_rank = rank
            best_box = box
            best_info = {
                "location_score": loc_score,
                "scale_score": scale_score,
                "available_dims": available_dims,
                "groundedness_raw": raw,
                "groundedness_norm": norm,
            }

    return best_box, best_info


def determine_reason(
    pseudo_box_found: bool,
    available_dims: int,
    location_score: Optional[float],
    scale_score: Optional[float],
    groundedness_norm: Optional[float]
) -> str:
    """
    Explain why groundedness score has its value.
    """
    if not pseudo_box_found:
        return "no_pseudo_box"

    if available_dims == 0:
        return "no_comparable_vlm_terms"

    if groundedness_norm is None:
        return "unknown"

    if groundedness_norm >= 0.999:
        return "matched"

    if groundedness_norm > 0.0:
        return "partial_match"

    # groundedness_norm == 0
    if location_score == 0.0 and scale_score in [None, 0.0]:
        return "location_mismatch"

    if scale_score == 0.0 and location_score in [None, 0.0]:
        return "scale_mismatch"

    return "mismatch"


# =============================================================================
# [5] Main groundedness computation
# =============================================================================
def compute_groundedness_for_item(
    vlm_item: Dict,
    pseudo_item: Dict
) -> Dict:
    image_name = get_image_name(vlm_item)

    vlm_texts = collect_vlm_text(vlm_item)
    location_terms = extract_location_terms(vlm_texts["location"])
    scale_term = extract_scale_term(vlm_texts["scale"])

    pseudo_boxes = pseudo_item.get("pseudo_boxes", [])
    pseudo_box_found = len(pseudo_boxes) > 0

    matched_box, match_info = choose_best_pseudo_box(
        pseudo_boxes=pseudo_boxes,
        vlm_location_terms=location_terms,
        vlm_scale_term=scale_term
    )

    available_dims = match_info["available_dims"]
    location_score = match_info["location_score"]
    scale_score = match_info["scale_score"]
    groundedness_raw = match_info["groundedness_raw"]
    groundedness_norm = match_info["groundedness_norm"]

    reason = determine_reason(
        pseudo_box_found=pseudo_box_found,
        available_dims=available_dims,
        location_score=location_score,
        scale_score=scale_score,
        groundedness_norm=groundedness_norm
    )

    pseudo_location_correct = (
        location_score is not None and location_score >= 0.5
    )
    pseudo_scale_correct = (
        scale_score is not None and scale_score >= 0.5
    )

    result = {
        "image_name": image_name,
        "dataset_type": vlm_item.get("dataset_type") or pseudo_item.get("dataset_type"),
        "image_path": pseudo_item.get("image_path"),
        "consistency_score": get_consistency_score(vlm_item),

        # GT oracle is kept only for diagnosis / upper-bound comparison
        "groundedness_gt_oracle": get_gt_oracle_groundedness(vlm_item),

        # VLM parsing info
        "vlm_location_terms": location_terms,
        "vlm_scale_term": scale_term,
        "vlm_location_available": len(location_terms) > 0,
        "vlm_scale_available": scale_term is not None,

        # Pseudo box info
        "pseudo_box_found": pseudo_box_found,
        "pseudo_box_count": len(pseudo_boxes),
        "pseudo_matched_box": matched_box,

        # Matching scores
        "location_score": location_score,
        "scale_score": scale_score,
        "pseudo_location_correct": pseudo_location_correct,
        "pseudo_scale_correct": pseudo_scale_correct,

        # Groundedness
        "groundedness_raw": groundedness_raw,
        "groundedness_available_dims": available_dims,
        "groundedness_norm": groundedness_norm,

        # Backward-compatible field
        "groundedness_pseudo": groundedness_raw,

        # Reason
        "groundedness_reason": reason,
    }

    # Convenience fields for later priority score generation
    if matched_box is not None:
        result.update({
            "best_box_score": matched_box.get("score"),
            "best_box_quality": matched_box.get("box_quality"),
            "best_box_area_ratio": matched_box.get("area_ratio"),
            "best_box_aspect_ratio": matched_box.get("aspect_ratio"),
            "best_box_spatial_zone": matched_box.get("spatial_zone"),
            "best_box_scale_category": matched_box.get("scale_category"),
            "best_box_prompt": matched_box.get("prompt"),
        })
    else:
        result.update({
            "best_box_score": None,
            "best_box_quality": None,
            "best_box_area_ratio": None,
            "best_box_aspect_ratio": None,
            "best_box_spatial_zone": None,
            "best_box_scale_category": None,
            "best_box_prompt": None,
        })

    return result


# =============================================================================
# [6] Summary and CSV output
# =============================================================================
def summarize_results(
    results: List[Dict],
    pseudo_total_count: int,
    vlm_total_count: int,
    matched_count: int
) -> Dict:
    reason_counter = Counter(r["groundedness_reason"] for r in results)
    dataset_counter = Counter(r.get("dataset_type", "unknown") for r in results)

    pseudo_found_count = sum(1 for r in results if r["pseudo_box_found"])
    location_available_count = sum(1 for r in results if r["vlm_location_available"])
    scale_available_count = sum(1 for r in results if r["vlm_scale_available"])

    comparable = [
        r for r in results
        if r["groundedness_norm"] is not None
    ]

    avg_groundedness_norm = None
    if comparable:
        avg_groundedness_norm = round(
            sum(float(r["groundedness_norm"]) for r in comparable) / len(comparable),
            4
        )

    avg_consistency = None
    valid_consistency = [
        r["consistency_score"] for r in results
        if r["consistency_score"] is not None
    ]
    if valid_consistency:
        avg_consistency = round(sum(valid_consistency) / len(valid_consistency), 4)

    dataset_summary = {}

    for ds in dataset_counter:
        ds_items = [r for r in results if r.get("dataset_type") == ds]
        ds_pseudo_found = sum(1 for r in ds_items if r["pseudo_box_found"])
        ds_comparable = [r for r in ds_items if r["groundedness_norm"] is not None]

        dataset_summary[ds] = {
            "num_items": len(ds_items),
            "pseudo_box_found": ds_pseudo_found,
            "pseudo_box_found_ratio": round(ds_pseudo_found / len(ds_items), 4) if ds_items else 0,
            "reason_counts": dict(Counter(r["groundedness_reason"] for r in ds_items)),
            "avg_groundedness_norm": (
                round(sum(float(r["groundedness_norm"]) for r in ds_comparable) / len(ds_comparable), 4)
                if ds_comparable else None
            )
        }

    return {
        "pseudo_total_count": pseudo_total_count,
        "vlm_total_count": vlm_total_count,
        "matched_overlap_count": matched_count,
        "result_count": len(results),

        "pseudo_box_found_count": pseudo_found_count,
        "pseudo_box_found_ratio": round(pseudo_found_count / len(results), 4) if results else 0,

        "vlm_location_available_count": location_available_count,
        "vlm_location_available_ratio": round(location_available_count / len(results), 4) if results else 0,

        "vlm_scale_available_count": scale_available_count,
        "vlm_scale_available_ratio": round(scale_available_count / len(results), 4) if results else 0,

        "avg_consistency_score": avg_consistency,
        "avg_groundedness_norm_comparable_only": avg_groundedness_norm,

        "reason_counts": dict(reason_counter),
        "dataset_summary": dataset_summary,
    }


def save_csv(results: List[Dict], output_file: Path):
    if not results:
        return

    fieldnames = [
        "image_name",
        "dataset_type",
        "consistency_score",
        "groundedness_gt_oracle",

        "vlm_location_terms",
        "vlm_scale_term",
        "vlm_location_available",
        "vlm_scale_available",

        "pseudo_box_found",
        "pseudo_box_count",

        "location_score",
        "scale_score",
        "pseudo_location_correct",
        "pseudo_scale_correct",

        "groundedness_raw",
        "groundedness_available_dims",
        "groundedness_norm",
        "groundedness_pseudo",
        "groundedness_reason",

        "best_box_score",
        "best_box_quality",
        "best_box_area_ratio",
        "best_box_aspect_ratio",
        "best_box_spatial_zone",
        "best_box_scale_category",
        "best_box_prompt",
        "image_path",
    ]

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {k: r.get(k) for k in fieldnames}

            # list field to string
            if isinstance(row.get("vlm_location_terms"), list):
                row["vlm_location_terms"] = "|".join(row["vlm_location_terms"])

            writer.writerow(row)


# =============================================================================
# [7] Main
# =============================================================================
def main():
    # Latest pseudo boxes
    pseudo_file = find_latest_file("pseudo_boxes_*/pseudo_boxes_owlvit.json")

    # Latest VLM consistency / GT-oracle pilot result
    # 필요하면 파일명 패턴을 본인 파일명에 맞게 추가하면 됨.
    vlm_file = find_latest_file("**/pilot_grounded_consistency_results.json")

    print("=" * 80)
    print("[*] Pseudo box file:")
    print(pseudo_file)
    print("[*] VLM result file:")
    print(vlm_file)
    print("=" * 80)

    pseudo_results = load_json(pseudo_file)
    vlm_results = load_json(vlm_file)

    pseudo_index = {
        get_image_name(item): item for item in pseudo_results
    }

    vlm_index = {
        get_image_name(item): item for item in vlm_results
    }

    overlap_names = sorted(set(pseudo_index.keys()) & set(vlm_index.keys()))

    print(f"[*] pseudo box items: {len(pseudo_results)}")
    print(f"[*] VLM result items: {len(vlm_results)}")
    print(f"[*] matched overlap items: {len(overlap_names)}")

    if len(overlap_names) == 0:
        print("[ERROR] 겹치는 image_name이 없습니다.")
        print("        pseudo box를 만든 이미지와 VLM consistency를 만든 이미지가 같은지 확인하세요.")
        return

    if len(overlap_names) < len(pseudo_results):
        print("[WARN] pseudo box 파일의 일부 이미지만 VLM 결과와 매칭됩니다.")
        print("       50장/100장 pseudo box를 계산했더라도 VLM 결과가 10장뿐이면 10장만 계산됩니다.")

    results = []

    for idx, image_name in enumerate(overlap_names):
        vlm_item = vlm_index[image_name]
        pseudo_item = pseudo_index[image_name]

        result = compute_groundedness_for_item(
            vlm_item=vlm_item,
            pseudo_item=pseudo_item
        )
        results.append(result)

        print(
            f"[{idx + 1}/{len(overlap_names)}] {image_name} | "
            f"box_found={result['pseudo_box_found']} | "
            f"reason={result['groundedness_reason']} | "
            f"G_norm={result['groundedness_norm']}"
        )

    summary = summarize_results(
        results=results,
        pseudo_total_count=len(pseudo_results),
        vlm_total_count=len(vlm_results),
        matched_count=len(overlap_names)
    )

    output_dir = pseudo_file.parent

    output_json = output_dir / "pseudo_groundedness_results.json"
    output_csv = output_dir / "pseudo_groundedness_results.csv"
    output_summary = output_dir / "pseudo_groundedness_summary.json"

    save_json(results, output_json)
    save_json(summary, output_summary)
    save_csv(results, output_csv)

    print("\n" + "=" * 80)
    print("[완료] pseudo groundedness 결과 저장")
    print(f"JSON   : {output_json}")
    print(f"CSV    : {output_csv}")
    print(f"SUMMARY: {output_summary}")
    print("=" * 80)

    print("\n[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()