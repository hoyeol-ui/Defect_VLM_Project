import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
SBERT_ID = "all-MiniLM-L6-v2"


def cosine(a, b):
    return float(np.dot(a, b))


def get_project_root():
    return Path(__file__).resolve().parent.parent


def find_image_paths(limit=10):
    project_root = get_project_root()
    image_dir = project_root / "data" / "NEU-DET" / "train" / "images"

    print("PROJECT_ROOT:", project_root)
    print("IMAGE_DIR:", image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"❌ image directory not found: {image_dir}")

    image_paths = sorted(image_dir.rglob("*.jpg"))
    print("FOUND IMAGES:", len(image_paths))

    if len(image_paths) == 0:
        raise ValueError(f"❌ No .jpg images found under: {image_dir}")

    return image_paths[:limit]


def build_messages(image):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a careful inspector of metal surface defects."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Describe the visible defect in this metal surface image. "
                        "Focus on appearance. Answer in one short sentence."
                    )
                }
            ]
        }
    ]


def generate_responses(model, processor, image_path, n=3):
    image = Image.open(image_path).convert("RGB")
    responses = []

    for _ in range(n):
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
                temperature=0.8,
                num_beams=1
            )

        input_len = inputs["input_ids"].shape[1]
        generated_only_ids = generated_ids[:, input_len:]

        trimmed_output = processor.batch_decode(
            generated_only_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        if trimmed_output == "":
            trimmed_output = "[EMPTY OUTPUT]"

        responses.append(trimmed_output)

    return responses


def get_consistency_score(embedder, responses):
    embeddings = embedder.encode(
        responses,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    pairwise = []
    for i, j in itertools.combinations(range(len(responses)), 2):
        sim = cosine(embeddings[i], embeddings[j])
        pairwise.append(((i + 1, j + 1), sim))

    mean_score = float(np.mean([sim for (_, sim) in pairwise])) if pairwise else 0.0
    return pairwise, mean_score


def prepare_log_paths():
    project_root = get_project_root()
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = log_dir / f"consistency_run_{timestamp}.txt"
    jsonl_path = log_dir / f"consistency_run_{timestamp}.jsonl"

    return txt_path, jsonl_path


def append_txt_line(txt_path, text):
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def append_jsonl(jsonl_path, record):
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    image_paths = find_image_paths(limit=10)

    txt_log_path, jsonl_log_path = prepare_log_paths()

    print("DEVICE:", DEVICE)
    print("TXT LOG:", txt_log_path)
    print("JSONL LOG:", jsonl_log_path)

    append_txt_line(txt_log_path, f"DEVICE: {DEVICE}")
    append_txt_line(txt_log_path, f"MODEL_ID: {MODEL_ID}")
    append_txt_line(txt_log_path, f"SBERT_ID: {SBERT_ID}")
    append_txt_line(txt_log_path, f"TOTAL_IMAGES: {len(image_paths)}")
    append_txt_line(txt_log_path, "=" * 70)

    print("\n⏳ Qwen2-VL loading...")
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
    print("✅ Qwen2-VL loading 성공")

    print("\n⏳ SBERT loading...")
    embedder = SentenceTransformer(SBERT_ID, device="cpu")
    print("✅ SBERT loading 성공")

    all_scores = []

    print("\n🔍 10장 consistency 파일럿 시작...\n")

    for idx, image_path in enumerate(image_paths, 1):
        responses = generate_responses(model, processor, str(image_path), n=3)
        pairwise, mean_score = get_consistency_score(embedder, responses)
        all_scores.append(mean_score)

        print("=" * 70)
        print(f"[{idx}/10] IMAGE_PATH: {image_path}")

        print("\nRAW RESPONSES")
        for r_idx, r in enumerate(responses, 1):
            print(f"[{r_idx}] {r}")

        print("\nPAIRWISE SIMILARITY")
        for (i, j), sim in pairwise:
            print(f"({i}, {j}): {sim:.4f}")

        print(f"\nMEAN CONSISTENCY SCORE: {mean_score:.4f}")

        append_txt_line(txt_log_path, "=" * 70)
        append_txt_line(txt_log_path, f"[{idx}/10] IMAGE_PATH: {image_path}")
        append_txt_line(txt_log_path, "RAW RESPONSES")
        for r_idx, r in enumerate(responses, 1):
            append_txt_line(txt_log_path, f"[{r_idx}] {r}")

        append_txt_line(txt_log_path, "PAIRWISE SIMILARITY")
        for (i, j), sim in pairwise:
            append_txt_line(txt_log_path, f"({i}, {j}): {sim:.4f}")

        append_txt_line(txt_log_path, f"MEAN CONSISTENCY SCORE: {mean_score:.4f}")

        record = {
            "image_index": idx,
            "image_path": str(image_path),
            "responses": responses,
            "pairwise_similarity": [
                {"pair": [i, j], "score": round(sim, 6)}
                for (i, j), sim in pairwise
            ],
            "mean_consistency_score": round(mean_score, 6)
        }
        append_jsonl(jsonl_log_path, record)

    final_avg = float(np.mean(all_scores)) if all_scores else 0.0

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print(f"AVERAGE SCORE OVER 10 IMAGES: {final_avg:.4f}")
    print("=" * 70)

    append_txt_line(txt_log_path, "=" * 70)
    append_txt_line(txt_log_path, "FINAL SUMMARY")
    append_txt_line(txt_log_path, f"AVERAGE SCORE OVER 10 IMAGES: {final_avg:.4f}")
    append_txt_line(txt_log_path, "=" * 70)

    append_jsonl(jsonl_log_path, {
        "final_summary": True,
        "average_score_over_10_images": round(final_avg, 6)
    })

    print("\n✅ 로그 저장 완료")