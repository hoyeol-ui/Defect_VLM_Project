"""Post-hoc ONLY: sealed R4 box evaluation and label-free pair contact sheets.
This module is never imported by preflight.py or patch_signal.py.
"""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import html
import io
from pathlib import Path
import sys
import numpy as np
import preflight as p
import patch_signal as signal


def region_rect(index):
    p.require(type(index) is int and 0 <= index < 16, "Bad region index")
    row, col = divmod(index, 4)
    return [col / 4, row / 4, (col + 1) / 4, (row + 1) / 4]


def check_box(box):
    p.require(isinstance(box, list) and len(box) == 4, "Expected normalized xyxy")
    p.require(all(type(v) in (int, float) and np.isfinite(v) for v in box), "Nonfinite or nonnumeric box")
    x1, y1, x2, y2 = box
    p.require(0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1, "Box must be valid normalized ORIGINAL-image xyxy")


def union_area(boxes):
    """Exact rectangle union area. Overlapping GT boxes are counted once."""
    if not boxes:
        return 0.0
    xs = sorted({v for b in boxes for v in (b[0], b[2])})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        spans = sorted((b[1], b[3]) for b in boxes if b[0] < right and b[2] > left)
        height, end = 0.0, -float("inf")
        for bottom, top in spans:
            height += max(0.0, top - max(bottom, end))
            end = max(end, top)
        total += (right - left) * height
    return total


def box_fraction(rect, boxes):
    clipped = []
    for b in boxes:
        c = [max(rect[0], b[0]), max(rect[1], b[1]), min(rect[2], b[2]), min(rect[3], b[3])]
        if c[0] < c[2] and c[1] < c[3]:
            clipped.append(c)
    return union_area(clipped) / ((rect[2] - rect[0]) * (rect[3] - rect[1]))


def validate_record(rec):
    p.require(isinstance(rec["id"], str), "Bad ID")
    weights = np.asarray(rec["region_weights"], dtype=float)
    p.require(weights.shape == (16,) and np.isfinite(weights).all() and (weights >= 0).all(), "Invalid region contributions")
    valid, top = rec["valid_regions"], rec["top_regions"]
    p.require(len(valid) == len(set(valid)) > 0 and all(type(i) is int and 0 <= i < 16 for i in valid), "Invalid valid-region list")
    p.require(len(top) == len(set(top)) <= signal.TOP_K and set(top) <= set(valid), "Invalid highlighted regions")
    gain = rec["gain"]
    p.require(np.isfinite(gain) and gain >= 0 and abs(weights.sum() - gain) <= 1e-12, "Contribution sum mismatch")
    p.require(all(weights[i] == 0 for i in set(range(16)) - set(valid)), "Invalid regions have weight")
    expected = sorted((i for i in valid if weights[i] > 0), key=lambda i: (-weights[i], i))[:signal.TOP_K]
    if gain <= signal.GAIN_FLOOR:
        expected = []
    p.require(top == expected, "Top regions were changed after scoring")


def evaluate_r4(plan, scores, annotations):
    p.require(scores["kind"] == "R4_LEAVE_ONE_OUT_DIAGNOSTIC" and scores["labels_read"] is False, "Only sealed label-free R4 scores can be evaluated")
    p.require([r["id"] for r in scores["records"]] == plan["sample_ids"], "Sample IDs differ from the frozen plan")
    p.require(annotations["coordinate_system"] == "original_image_normalized_xyxy", "Do not use F518/padded/YOLO-xywh coordinates directly")
    boxes_by_id = annotations["boxes"]
    p.require(set(boxes_by_id) == set(plan["sample_ids"]) <= set(plan["r4"]),
              "GT must contain exactly the frozen R4 sample; extra IDs (including unlabeled) are rejected")
    for boxes in boxes_by_id.values():
        p.require(isinstance(boxes, list), "Boxes must be a list; [] means confirmed no boxes, not missing annotation")
        for box in boxes:
            check_box(box)
    rows = []
    for rec in scores["records"]:
        validate_record(rec)
        sid, top, valid = rec["id"], rec["top_regions"], rec["valid_regions"]
        boxes = boxes_by_id[sid]
        row = {"id": sid, "gain": rec["gain"], "top_regions": top, "box_count": len(boxes),
               "gt_union_area_fraction": union_area(boxes), "status": "SATURATED" if not top else "NO_GT_BOXES" if not boxes else "EVALUATED"}
        if top:
            fractions = {i: box_fraction(region_rect(i), boxes) for i in valid}
            observed = float(np.mean([fractions[i] for i in top]))
            seed = int(hashlib.sha256(("R4-spatial-null-v1\0" + sid).encode()).hexdigest()[:16], 16)
            rng = np.random.default_rng(seed)
            null = [float(np.mean([fractions[int(i)] for i in rng.choice(valid, size=len(top), replace=False)]))
                    for _ in range(signal.REPEATS)]
            # Under uniform same-k sampling the exact expected mean is the mean of all valid cells.
            expected = float(np.mean(list(fractions.values())))
            row.update(observed_box_fraction=observed, random_expected_box_fraction=expected,
                       difference_from_random=observed - expected, random_mc_mean=float(np.mean(null)),
                       random_mc_q05=float(np.quantile(null, .05)), random_mc_q95=float(np.quantile(null, .95)))
        rows.append(row)
    evaluable = [r for r in rows if r["status"] == "EVALUATED"]
    return {"kind": "R4_POSTHOC_BOX_OVERLAP", "records": rows,
            "sample_n": len(rows), "evaluated_n": len(evaluable),
            "saturated_n": sum(not r["top_regions"] for r in rows),
            "no_gt_boxes_n": sum(r["box_count"] == 0 for r in rows),
            "mean_difference_evaluated_only": float(np.mean([r["difference_from_random"] for r in evaluable])) if evaluable else None,
            "random_repeats": signal.REPEATS, "research_success": None,
            "interpretation": "Descriptive overlap only; boxes contain background. R4 is selected, small, and not an independent validation set. No detection/AP claim or automatic pass threshold."}


def image_svg(sample_id, rec, images, image_root, boxes=None):
    from PIL import Image  # optional until explicitly rendering existing images
    entry = images[sample_id]
    path = Path(entry["path"])
    if not path.is_absolute():
        path = image_root / path
    signal.hashed(path, entry["sha256"])
    with Image.open(path) as raw:
        # Never silently EXIF-rotate relative to annotation coordinates.
        p.require(raw.getexif().get(274, 1) == 1, "EXIF orientation needs explicit original-coordinate verification")
        image = raw.convert("RGB")
        image.thumbnail((640, 640))
        w, h = image.size
        buf = io.BytesIO()
        image.save(buf, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    pieces = [f'<svg viewBox="0 0 {w} {h}" role="img"><image width="{w}" height="{h}" href="{uri}"/>']
    for index in rec["top_regions"]:
        x1, y1, x2, y2 = region_rect(index)
        pieces.append(f'<rect x="{x1*w}" y="{y1*h}" width="{(x2-x1)*w}" height="{(y2-y1)*h}" fill="#ff9800" fill-opacity="0.16" stroke="#ff9800" stroke-width="2"/>')
    for b in boxes or []:
        x1, y1, x2, y2 = b
        pieces.append(f'<rect x="{x1*w}" y="{y1*h}" width="{(x2-x1)*w}" height="{(y2-y1)*h}" fill="none" stroke="#00ff88" stroke-width="2"/>')
    return "".join(pieces) + "</svg>"


def render(scores, images, image_root, out, boxes=None):
    pair_mode = scores["kind"] == "FROZEN_FINAL_PAIR_REVIEW"
    p.require(not pair_mode or boxes is None, "Pair review cannot receive GT")
    parts = ['<!doctype html><meta charset="utf-8"><title>Patch diagnostic review</title><style>body{font:16px system-ui;max-width:1200px;margin:24px auto;background:#fafafa}article{padding:16px;margin:16px 0;background:white;border:1px solid #ddd}.pair{display:flex;gap:16px}.side{width:50%}svg{width:100%;max-height:640px}p{line-height:1.5}</style>',
             '<h1>Patch 기여 위치 검토</h1><p>주황: 목적함수 기여 상위 지역. 초록: R4 사후 진단 box(쌍 검토에는 없음). 강조가 없으면 기여가 포화/극소임. 결함 위치 예측이 아님.</p>']
    review_ids = []
    for rec in scores["records"]:
        if pair_mode:
            rid = rec["pair_id"]
            parts.append(f'<article><h2>{html.escape(rid)}</h2><div class="pair">')
            for side, item in zip(("A", "B"), rec["sides"]):
                validate_record(item)
                parts.append(f'<div class="side"><h3>{side}</h3>' + image_svg(item["id"], item, images, image_root) + '</div>')
            parts.append('</div></article>')
        else:
            validate_record(rec)
            rid = rec["id"]
            parts.append(f'<article><h2>{html.escape(rid)}</h2>' + image_svg(rid, rec, images, image_root, boxes[rid] if boxes is not None else None) + '</article>')
        review_ids.append(rid)
    (out / "review.html").write_text("\n".join(parts), encoding="utf-8")
    with (out / "review_sheet.csv").open("x", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["review_id", "local_morphology_difference", "background_lighting", "padding_boundary", "near_duplicate", "unclear", "notes"])
        for rid in review_ids:
            writer.writerow([rid, "", "", "", "", "", ""])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("evaluate-r4", "render-pairs"))
    for name in ("plan", "scores"):
        ap.add_argument("--" + name, required=True, type=Path)
        ap.add_argument("--" + name + "-sha", required=True)
    for name in ("annotations", "images"):
        ap.add_argument("--" + name, type=Path)
        ap.add_argument("--" + name + "-sha")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    if args.command == "render-pairs" and (args.annotations or args.annotations_sha):
        ap.error("render-pairs forbids annotation inputs")
    if args.command == "evaluate-r4" and not (args.annotations and args.annotations_sha):
        ap.error("evaluate-r4 requires frozen-sample R4 annotations and SHA256")
    if bool(args.images) != bool(args.images_sha) or (args.command == "render-pairs" and not args.images):
        ap.error("Rendering requires --images and --images-sha")
    if args.out.exists():
        ap.error("Output exists; choose a new directory")
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        plan = signal.load_plan(args.plan, args.plan_sha)
        p.require(plan["posthoc_code_sha256"] == p.sha256(__file__), "Post-hoc code changed after freezing; review a new protocol version")
        scores = p.read_json(signal.hashed(args.scores, args.scores_sha))
        p.require(scores["plan_sha256"] == args.plan_sha and scores["labels_read"] is False, "Unsealed or label-dependent scores")
        p.require(scores["signal_code_sha256"] == plan["signal_code_sha256"], "Score code differs from frozen protocol")
        provenance = {"plan_sha256": args.plan_sha, "scores_sha256": args.scores_sha,
                      "posthoc_code_sha256": p.sha256(__file__), "numpy": np.__version__, "python": sys.version}
        boxes = None
        if args.command == "evaluate-r4":
            ann = p.read_json(signal.hashed(args.annotations, args.annotations_sha))
            result = evaluate_r4(plan, scores, ann)
            boxes = ann["boxes"]
            provenance["annotations_sha256"] = args.annotations_sha
            p.write_json(args.out / "box_overlap.json", result)
        else:
            p.require(scores["kind"] == "FROZEN_FINAL_PAIR_REVIEW", "Expected frozen pair review")
            for pair in scores["records"]:
                p.require(len(pair["sides"]) == 2, "Expected A/B pair")
                for item in pair["sides"]:
                    p.require(item["id"] in plan["train_ids"], "Non-train image")
        if args.images:
            images = p.read_json(signal.hashed(args.images, args.images_sha))["images"]
            provenance["images_manifest_sha256"] = args.images_sha
            render(scores, images, args.images.resolve().parent, args.out, boxes)
        p.write_json(args.out / "provenance.json", provenance)
        print("Post-hoc diagnostic complete; no selection, extraction or training executed.")
        return 0
    except (ValueError, OSError, KeyError, TypeError, IndexError, ImportError) as exc:
        p.write_json(args.out / "blocked.json", {"status": "BLOCKED", "error": str(exc)})
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
