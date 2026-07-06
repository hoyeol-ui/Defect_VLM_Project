import os
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def run_test():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(
        os.path.dirname(current_dir),
        "data",
        "NEU-DET",
        "train",
        "images",
        "pitted_surface",
        "pitted_surface_13.jpg"
    )

    print(f"IMAGE_PATH: {image_path}")
    print(f"DEVICE: {DEVICE}")

    # 1) Mac MPS에서 가장 안전하게: float32 + 명시적 to(device)
    print("⏳ model loading...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32
    )
    model.to(DEVICE)
    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("✅ model loading 성공")

    # 2) 문자열 경로 대신 PIL 이미지로 고정
    image = Image.open(image_path).convert("RGB")

    # 3) prompt를 더 안정적으로 축소
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a careful visual inspector for metal surface defects."
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "Identify the defect type in this metal surface image. "
                        "Reply with only the defect name in 1 to 3 words. "
                        "No punctuation. If unsure, say unknown defect."
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    print("🔍 Qwen2-VL 추론 시작...")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            num_beams=1
        )

    # 4) full decode와 generated-only decode를 분리
    raw_output_full = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]

    input_token_len = inputs["input_ids"].shape[1]
    generated_only_ids = generated_ids[:, input_token_len:]

    trimmed_output = processor.batch_decode(
        generated_only_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0].strip()

    print("\n" + "=" * 60)
    print("RAW OUTPUT:")
    print(raw_output_full)
    print("=" * 60)
    print("TRIMMED OUTPUT:")
    print(trimmed_output)
    print("=" * 60)

if __name__ == "__main__":
    run_test()