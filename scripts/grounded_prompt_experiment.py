import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# 위 유틸리티 스크립트에서 파서 기능 로드
from utils_bbox_parser import parse_voc_xml, get_xml_path_from_image

# =============================================================================
# [1] 가속 기기 및 경로 설정
# =============================================================================
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
SBERT_ID = "sentence-transformers/all-MiniLM-L6-v2"

PROJECT_ROOT = Path("/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / f"grounded_experiment_{datetime.now().strftime('%Y%m%d')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# [2] Detection-Oriented 프롬프트 세트 정의
# =============================================================================
PROMPT_FAMILY = {
    "location": (
        "Identify the main defect in this image. Describe its location using standard spatial terms "
        "(e.g., top-left, top-center, top-right, center-left, center, center-right, bottom-left, bottom-center, bottom-right). "
        "Your response must include a phrase like: 'The defect is located in the [spatial term] region.'"
    ),
    "scale": (
        "Analyze the size of the defect relative to the entire image area. "
        "Categorize its scale as either 'micro' (tiny point), 'small' (localized region), or 'large' (covering significant portion). "
        "Your response must include a phrase like: 'The scale of the defect is [scale category].'"
    ),
    "appearance": (
        "As an industrial quality inspector, examine this image. "
        "Describe the shape, texture, and visual boundary of the defect in detail. Keep it brief."
    )
}


def load_models():
    print(f"[*] {DEVICE} 장치에 Qwen2-VL 로딩 중...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map="auto" if DEVICE != "mps" else None
    )
    if DEVICE == "mps":
        model = model.to(torch.device("mps"))

    print("[*] SBERT 임베딩 인코더 로딩 중...")
    sbert = SentenceTransformer(SBERT_ID)
    return processor, model, sbert


def query_vlm(processor, model, image_path: Path, prompt_text: str) -> str:
    # PIL 기반 가벼운 스케일 다운사이징을 통한 맥북 메모리 방어 및 가짜 1.0 점수 격파
    temp_img_path = PROJECT_ROOT / "temp_vlm_input.jpg"
    with Image.open(image_path) as img:
        img.thumbnail((768, 768))
        img.save(temp_img_path, "JPEG")

    messages = [{"role": "user",
                 "content": [{"type": "image", "image": str(temp_img_path)}, {"type": "text", "text": prompt_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    if temp_img_path.exists():
        temp_img_path.unlink()

    return output_text[0].strip()


def evaluate_image_grounded_consistency(processor, model, sbert, image_path: Path, xml_path: Path, dataset_type: str) -> \
Dict[str, Any]:
    gt_info = parse_voc_xml(xml_path, dataset_type=dataset_type)
    objects = gt_info["objects"]
    if not objects:
        return {"error": "No objects found in XML"}

    primary_object = max(objects, key=lambda x: x["area_ratio"])

    responses = {}
    for key, prompt in PROMPT_FAMILY.items():
        responses[key] = query_vlm(processor, model, image_path, prompt)

    embeddings = sbert.encode(list(responses.values()), convert_to_tensor=True)
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = torch.cosine_similarity(embeddings[i], embeddings[j], dim=0).item()
            similarities.append(sim)
    consistency_score = round(statistics.mean(similarities), 4) if similarities else 1.0

    loc_resp_lower = responses["location"].lower()
    scale_resp_lower = responses["scale"].lower()

    primary_pos_terms = primary_object["spatial_zone"].split("-")
    primary_loc_grounded = all(term in loc_resp_lower for term in primary_pos_terms)

    any_loc_grounded = False
    matched_label, matched_zone = None, None
    for obj in objects:
        pos_terms = obj["spatial_zone"].split("-")
        if all(term in loc_resp_lower for term in pos_terms):
            any_loc_grounded = True
            matched_label = obj["label"]
            matched_zone = obj["spatial_zone"]
            break

    primary_scale_grounded = primary_object["scale_category"] in scale_resp_lower
    groundedness_score = float(primary_loc_grounded) + float(primary_scale_grounded)

    return {
        "image_name": image_path.name,
        "dataset_type": dataset_type,
        "object_count": len(objects),
        "primary_object": primary_object,
        "responses": responses,
        "consistency_score": consistency_score,
        "groundedness": {
            "primary_location_correct": primary_loc_grounded,
            "primary_scale_correct": primary_scale_grounded,
            "any_location_correct": any_loc_grounded,
            "matched_object_info": {"label": matched_label, "zone": matched_zone} if any_loc_grounded else None,
            "total_score": groundedness_score
        }
    }


def main():
    processor, model, sbert = load_models()
    results = []
    pilot_targets = []

    # 💡 [규모 스케일업] NEU-DET에서 최대 50장 동적 선별
    neu_images = list((PROJECT_ROOT / "data" / "NEU-DET" / "IMAGES").glob("*.jpg"))[:50]
    for img in neu_images:
        xml_p = get_xml_path_from_image(img, dataset_type="NEU-DET")
        if xml_p and xml_p.exists(): pilot_targets.append((img, xml_p, "NEU-DET"))

    # 💡 [규모 스케일업] GC10-DET에서 최대 50장 동적 선별
    gc10_images = list((PROJECT_ROOT / "data" / "GC10-DET").glob("**/img_*.jpg"))[:50]
    for img in gc10_images:
        xml_p = get_xml_path_from_image(img, dataset_type="GC10-DET")
        if xml_p and xml_p.exists(): pilot_targets.append((img, xml_p, "GC10-DET"))

    print(f"[!] 총 {len(pilot_targets)}개의 본 실험용 대규모 이미지 풀 평가 시작.\n")

    for idx, (img_p, xml_p, d_type) in enumerate(pilot_targets):
        print(f"[{idx + 1}/{len(pilot_targets)}] 추론 중: {img_p.name} ({d_type})")
        try:
            res = evaluate_image_grounded_consistency(processor, model, sbert, img_p, xml_p, d_type)
            results.append(res)
            print(f"  -> 일관성: {res['consistency_score']} | 정합성: {res['groundedness']['total_score']}/2.0")
        except Exception as e:
            print(f"  ❌ 에러: {e}")

    output_file = OUTPUT_DIR / "pilot_grounded_consistency_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"\n[완료] 100장 스케일업 보고서 저장 완료: {output_file}")


if __name__ == "__main__":
    main()