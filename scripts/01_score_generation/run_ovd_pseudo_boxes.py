"""
===============================================================================
[File] run_ovd_pseudo_boxes.py

[Purpose]
Generate label-free pseudo bounding boxes from unlabeled industrial defect images
using an open-vocabulary detector.

[Research Motivation]
The previous groundedness implementation used GT VOC XML bounding boxes during
acquisition scoring. This violates the active learning assumption because the
unlabeled pool should not expose ground-truth boxes.

Therefore, this script replaces GT boxes with pseudo boxes predicted by an
open-vocabulary detector.

[References]
1. OWL-ViT
   - Hugging Face documentation:
     https://huggingface.co/docs/transformers/model_doc/owlvit
   - Used for: text-conditioned open-vocabulary object detection.

2. Grounding DINO
   - Official repository:
     https://github.com/IDEA-Research/GroundingDINO
   - Used conceptually for: text-conditioned pseudo bounding box generation.

3. YOLO-World
   - Official repository:
     https://github.com/AILab-CVC/YOLO-World
   - Candidate model for future real-time open-vocabulary detection.
===============================================================================
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection


# =============================================================================
# [0] Project path setup
# =============================================================================
PROJECT_ROOT = Path("/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project")
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# PyCharm에서 하위 폴더 파일을 직접 실행할 때 scripts/utils import가 안 되는 문제 방지
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from utils.utils_box_geometry import (
    xyxy_to_geometry,
    is_valid_pseudo_box,
    compute_box_quality
)
# =============================================================================
# [1] Basic configuration
# =============================================================================
OUTPUT_DIR = PROJECT_ROOT / "outputs" / f"pseudo_boxes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

MODEL_ID = "google/owlvit-base-patch32"

TEXT_PROMPTS = [
    "scratch defect",
    "scratches on metal surface",
    "pitted surface defect",
    "surface pit",
    "patch defect",
    "rolled-in scale defect",
    "inclusion defect",
    "crazing defect",
    "dark abnormal spot",
    "bright abnormal region",
    "thin line defect",
    "metal surface damage"
]

SCORE_THRESHOLD = 0.01
MAX_BOXES_PER_IMAGE = 5

# 처음 테스트는 작게. 성공하면 50으로 늘리기.
MAX_PER_DATASET = 50


# =============================================================================
# [2] Dataset collection
# =============================================================================
def collect_candidate_images(max_per_dataset: int = 5) -> List[Dict]:
    candidates = []

    neu_dir = PROJECT_ROOT / "data" / "NEU-DET" / "IMAGES"
    if neu_dir.exists():
        for img_path in list(neu_dir.glob("*.jpg"))[:max_per_dataset]:
            candidates.append({
                "image_path": img_path,
                "image_name": img_path.name,
                "dataset_type": "NEU-DET"
            })
    else:
        print(f"[WARN] NEU-DET 이미지 폴더 없음: {neu_dir}")

    gc10_dir = PROJECT_ROOT / "data" / "GC10-DET"
    if gc10_dir.exists():
        for img_path in list(gc10_dir.glob("**/img_*.jpg"))[:max_per_dataset]:
            candidates.append({
                "image_path": img_path,
                "image_name": img_path.name,
                "dataset_type": "GC10-DET"
            })
    else:
        print(f"[WARN] GC10-DET 이미지 폴더 없음: {gc10_dir}")

    return candidates


# =============================================================================
# [3] Model loading
# =============================================================================
def load_ovd_model():
    print(f"[*] Loading OWL-ViT model: {MODEL_ID}")
    print(f"[*] Device: {DEVICE}")

    processor = OwlViTProcessor.from_pretrained(MODEL_ID)
    model = OwlViTForObjectDetection.from_pretrained(MODEL_ID)
    model.to(DEVICE)
    model.eval()

    return processor, model


# =============================================================================
# [4] Inference
# =============================================================================
def predict_pseudo_boxes(
    processor,
    model,
    image_path: Path,
    prompts: List[str]
) -> Dict:
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    text_labels = [prompts]

    inputs = processor(
        text=text_labels,
        images=image,
        return_tensors="pt"
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Hugging Face 문서 기준: target_sizes는 (height, width)
    target_sizes = torch.tensor(
        [(image_height, image_width)],
        device=DEVICE
    )

    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=SCORE_THRESHOLD,
        text_labels=text_labels
    )[0]

    boxes = results["boxes"].detach().cpu().tolist()
    scores = results["scores"].detach().cpu().tolist()
    detected_text_labels = results["text_labels"]

    pseudo_boxes = []

    for box, score, detected_label in zip(boxes, scores, detected_text_labels):
        geometry = xyxy_to_geometry(
            xyxy=box,
            image_width=image_width,
            image_height=image_height
        )

        # 너무 큰 표면 영역이나 거의 전체 이미지 박스 제거
        if not is_valid_pseudo_box(
                geometry,
                min_area_ratio=0.0001,
                max_area_ratio=0.20,
                max_width_ratio=0.90,
                max_height_ratio=0.95
        ):
            continue

        box_quality = compute_box_quality(
            score=float(score),
            geometry=geometry,
            area_penalty_weight=1.0
        )

        pseudo_boxes.append({
            "xyxy": geometry["raw_bbox"],
            "score": round(float(score), 4),
            "box_quality": box_quality,
            "prompt": detected_label,
            **geometry
        })
    pseudo_boxes = sorted(
        pseudo_boxes,
        key=lambda x: x["box_quality"],
        reverse=True
    )[:MAX_BOXES_PER_IMAGE]

    return {
        "image_width": image_width,
        "image_height": image_height,
        "pseudo_boxes": pseudo_boxes
    }

# =============================================================================
# [5] Main
# =============================================================================
def main():
    candidates = collect_candidate_images(max_per_dataset=MAX_PER_DATASET)

    print(f"[!] Candidate images: {len(candidates)}")

    if not candidates:
        print("[ERROR] 이미지 후보가 없습니다. data 폴더 구조를 확인하세요.")
        return

    processor, model = load_ovd_model()

    all_results = []

    for idx, item in enumerate(candidates):
        img_path = item["image_path"]

        print(f"\n[{idx + 1}/{len(candidates)}] OVD inference: {img_path.name}")

        try:
            pred = predict_pseudo_boxes(
                processor=processor,
                model=model,
                image_path=img_path,
                prompts=TEXT_PROMPTS
            )

            result = {
                "image_name": item["image_name"],
                "dataset_type": item["dataset_type"],
                "image_path": str(img_path),
                "ovd_model": MODEL_ID,
                "text_prompts": TEXT_PROMPTS,
                "score_threshold": SCORE_THRESHOLD,
                **pred
            }

            all_results.append(result)

            print(f"  -> pseudo boxes: {len(pred['pseudo_boxes'])}")
            if pred["pseudo_boxes"]:
                top_box = pred["pseudo_boxes"][0]
                print(
                    f"  -> top box: {top_box['xyxy']} | "
                    f"score={top_box['score']} | "
                    f"zone={top_box['spatial_zone']} | "
                    f"scale={top_box['scale_category']}"
                )

        except Exception as e:
            print(f"  ❌ error: {e}")

            all_results.append({
                "image_name": item["image_name"],
                "dataset_type": item["dataset_type"],
                "image_path": str(img_path),
                "error": str(e),
                "pseudo_boxes": []
            })

    output_file = OUTPUT_DIR / "pseudo_boxes_owlvit.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print("\n" + "=" * 80)
    print("[완료] pseudo box 결과 저장")
    print(output_file)
    print("=" * 80)


if __name__ == "__main__":
    main()