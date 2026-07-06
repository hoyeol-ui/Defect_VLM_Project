import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pathlib import Path
from PIL import Image

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

PROJECT_ROOT = Path.cwd()
IMAGE_PATH = list((PROJECT_ROOT / "data" / "NEU-DET").rglob("*.jpg"))[0]

print("IMAGE_PATH:", IMAGE_PATH)
print("DEVICE:", DEVICE)

print("⏳ model loading...")
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

print("✅ model loading 성공")

image = Image.open(IMAGE_PATH).convert("RGB")

messages = [
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
                    "This image belongs to one of the following metal surface defect classes: "
                    "crazing, inclusion, patches, pitted_surface, rolled_in_scale, scratches. "
                    "Choose exactly one label from the list and output only that label."
                )
            }
        ]
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
    return_tensors="pt"
)

inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

print("🔍 Qwen2-VL 추론 시작...")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        num_beams=1
    )

raw_output = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]

input_length = inputs["input_ids"].shape[1]
generated_only_ids = generated_ids[:, input_length:]

trimmed_output = processor.batch_decode(
    generated_only_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0].strip()

print("\n" + "=" * 60)
print("RAW OUTPUT:")
print(raw_output)
print("=" * 60)
print("TRIMMED OUTPUT:")
print(trimmed_output)
print("=" * 60)