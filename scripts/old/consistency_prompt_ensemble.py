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

SAMPLES_PER_CLASS = 2

PROMPTS = [
    "Describe the visible defect on this metal surface in one short sentence.",
    "Explain briefly what kind of surface damage or defect is visible in this image.",
    "Describe the defect pattern by focusing on its shape, texture, or appearance in one short sentence."
]


def cosine(a, b):
    return float(np.dot(a, b))


def get_project_root():
    return Path(__file__).resolve().parent.parent


def prepare_logs():
    root = get_project_root()
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = log_dir / f"prompt_ensemble_{ts}.txt"
    jsonl_path = log_dir / f"prompt_ensemble_{ts}.jsonl"
    return txt_path, jsonl_path


def log_txt(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def log_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def sample_images_by_class():
    root = get_project_root()
    image_root = root / "data" / "NEU-DET" / "train" / "images"

    class_dirs = sorted([d for d in image_root.iterdir() if d.is_dir()])
    print("CLASSES:", [d.name for d in class_dirs])

    selected = []
    for class_dir in class_dirs:
        images = sorted(class_dir.glob("*.jpg"))
        if not images:
            continue
        selected.extend(images[:SAMPLES_PER_CLASS])

    print("TOTAL SELECTED:", len(selected))
    return selected


def build_messages(image, prompt_text):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a careful inspector of metal surface defects. Avoid generic template phrases. Describe only what is visually observable."
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]


def generate_one_response(model, processor, image, prompt_text):
    messages = build_messages(image, prompt_text)

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
            max_new_tokens=32,
            do_sample=True,
            temperature=1.1,
            top_p=0.95,
            num_beams=1
        )

    input_len = inputs["input_ids"].shape[1]
    gen_ids = generated_ids[:, input_len:]

    output = processor.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0].strip()

    if not output:
        output = "[EMPTY OUTPUT]"

    return output


def get_pairwise_and_mean(embedder, responses):
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


if __name__ == "__main__":
    txt_log, jsonl_log = prepare_logs()
    image_paths = sample_images_by_class()

    print("DEVICE:", DEVICE)
    print("TXT LOG:", txt_log)
    print("JSONL LOG:", jsonl_log)

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

    for idx, image_path in enumerate(image_paths, 1):
        image = Image.open(image_path).convert("RGB")

        responses = []
        for prompt_text in PROMPTS:
            response = generate_one_response(model, processor, image, prompt_text)
            responses.append(response)

        pairwise, mean_score = get_pairwise_and_mean(embedder, responses)
        all_scores.append(mean_score)

        print("\n" + "=" * 70)
        print(f"[{idx}/{len(image_paths)}] IMAGE_PATH: {image_path}")

        print("\nPROMPT + RESPONSE")
        for p_idx, (prompt_text, response) in enumerate(zip(PROMPTS, responses), 1):
            print(f"[{p_idx}] PROMPT: {prompt_text}")
            print(f"    RESPONSE: {response}")

        print("\nPAIRWISE SIMILARITY")
        for (i, j), sim in pairwise:
            print(f"({i}, {j}): {sim:.4f}")

        print(f"\nMEAN CONSISTENCY SCORE: {mean_score:.4f}")

        log_txt(txt_log, "=" * 70)
        log_txt(txt_log, f"IMAGE_PATH: {image_path}")
        for p_idx, (prompt_text, response) in enumerate(zip(PROMPTS, responses), 1):
            log_txt(txt_log, f"[{p_idx}] PROMPT: {prompt_text}")
            log_txt(txt_log, f"[{p_idx}] RESPONSE: {response}")
        for (i, j), sim in pairwise:
            log_txt(txt_log, f"PAIR ({i}, {j}): {sim:.4f}")
        log_txt(txt_log, f"MEAN CONSISTENCY SCORE: {mean_score:.4f}")

        log_jsonl(jsonl_log, {
            "image_path": str(image_path),
            "prompts": PROMPTS,
            "responses": responses,
            "pairwise_similarity": [
                {"pair": [i, j], "score": round(sim, 6)}
                for (i, j), sim in pairwise
            ],
            "mean_consistency_score": round(mean_score, 6)
        })

    final_avg = float(np.mean(all_scores)) if all_scores else 0.0

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print(f"AVERAGE SCORE OVER {len(image_paths)} IMAGES: {final_avg:.4f}")
    print("=" * 70)

    log_txt(txt_log, "=" * 70)
    log_txt(txt_log, "FINAL SUMMARY")
    log_txt(txt_log, f"AVERAGE SCORE OVER {len(image_paths)} IMAGES: {final_avg:.4f}")
    log_txt(txt_log, "=" * 70)

    log_jsonl(jsonl_log, {
        "final_summary": True,
        "average_score": round(final_avg, 6),
        "num_images": len(image_paths)
    })

    print("\n✅ 로그 저장 완료")