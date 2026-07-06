import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os

# 1. 맥북 GPU(MPS) 사용 설정
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"사용 중인 디바이스: {device}")

# 2. 모델 및 프로세서 로드 (Florence-2-base)
model_id = "microsoft/Florence-2-base"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

def run_example(image_path, prompt="<CAPTION>"):
    if not os.path.exists(image_path):
        return f"File not found: {image_path}"

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=prompt, 
        image_size=(image.width, image.height)
    )

    return parsed_answer

if __name__ == "__main__":
    # 데이터 경로에 실제 이미지를 넣은 후 테스트하세요.
    test_image = "data/NEU-DET/IMAGES/scratches_1.jpg"
    result = run_example(test_image)
    print(f"--- 테스트 결과 ---")
    print(result)
