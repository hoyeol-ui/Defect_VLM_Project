import itertools
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# =========================
# [1] 설정
# =========================
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
SBERT_ID = "all-MiniLM-L6-v2"

SAMPLES_PER_CLASS = 2   # 클래스당 몇 장 뽑을지
REPEAT_PER_IMAGE = 3    # 한 이미지당 몇 번 생성


# =========================
# [2] 유틸
# =========================
def cosine(a, b):
    return float(np.dot(a, b))


def get_project_root():
    return Path(__file__).resolve().parent.parent


def sample_images_by_class():
    """
    각 클래스 폴더에서 균등하게 이미지 샘플링
    """
    root = get_project_root()
    image_root = root / "data" / "NEU-DET" / "train" / "images"

    class_dirs = [d for d in image_root.iterdir() if d.is_dir()]

    print("CLASSES:", [d.name for d in class_dirs])

    selected = []

    for c in class_dirs:
        images = list(c.glob("*.jpg"))
        if len(images) == 0:
            continue

        sampled = random.sample(images, min(SAMPLES_PER_CLASS, len(images)))
        selected.extend(sampled)

    print(f"TOTAL SELECTED: {len(selected)}")
    return selected


def build_messages(image):
    return [
        {
            "role": "system",
            "content": [
                {"type": "text",
                 "text": "You are a careful inspector of metal surface defects."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",
                 "text": (
                     "Describe the visible defect pattern on the metal surface. "
                     "Mention its shape, texture, or appearance in one short sentence."
                 )}
            ]
        }
    ]


def generate_responses(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")
    responses = []

    for _ in range(REPEAT_PER_IMAGE):
        messages = build_messages(image)

        text_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=True,
                temperature=0.9,   # 🔥 다양성 증가
                num_beams=1
            )

        input_len = inputs["input_ids"].shape[1]
        gen_ids = generated_ids[:, input_len:]

        output = processor.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        if output == "":
            output = "[EMPTY]"

        responses.append(output)

    return responses


def get_consistency(embedder, responses):
    embeddings = embedder.encode(
        responses,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    sims = []
    for i, j in itertools.combinations(range(len(responses)), 2):
        sims.append(np.dot(embeddings[i], embeddings[j]))

    return float(np.mean(sims))


def prepare_logs():
    root = get_project_root()
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    return (
        log_dir / f"mixed_eval_{ts}.txt",
        log_dir / f"mixed_eval_{ts}.jsonl"
    )


def log_txt(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def log_jsonl(path, data):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# =========================
# [3] 실행
# =========================
if __name__ == "__main__":

    txt_log, jsonl_log = prepare_logs()

    print("DEVICE:", DEVICE)
    print("TXT LOG:", txt_log)

    print("\n⏳ 모델 로딩...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        attn_implementation="eager"
    ).to(DEVICE)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        use_fast=False
    )

    embedder = SentenceTransformer(SBERT_ID, device="cpu")

    print("✅ 모델 로딩 완료")

    images = sample_images_by_class()

    all_scores = []

    for idx, img_path in enumerate(images, 1):
        responses = generate_responses(model, processor, str(img_path))
        score = get_consistency(embedder, responses)

        all_scores.append(score)

        print("\n" + "=" * 60)
        print(f"[{idx}] {img_path.name}")
        for i, r in enumerate(responses, 1):
            print(f"  [{i}] {r}")
        print(f"CONSISTENCY: {score:.4f}")

        log_txt(txt_log, "=" * 60)
        log_txt(txt_log, f"{img_path}")
        for r in responses:
            log_txt(txt_log, r)
        log_txt(txt_log, f"score: {score:.4f}")

        log_jsonl(jsonl_log, {
            "image": str(img_path),
            "responses": responses,
            "consistency": round(score, 4)
        })

    print("\n" + "=" * 60)
    print("FINAL AVG:", np.mean(all_scores))