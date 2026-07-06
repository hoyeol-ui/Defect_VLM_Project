import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel  # [핵심] 어댑터 합체용
import os

# --- [1] 설정 ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BASE_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# 프로젝트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
ADAPTER_PATH = os.path.join(project_root, "models", "qwen_lora_adapter")
IMAGE_PATH = os.path.join(project_root, "data", "NEU-DET", "train", "images", "scratches", "scratches_1.jpg")


def run_test():
    print(f"🚀 디바이스: {DEVICE}")
    print(f"⏳ 베이스 모델 로딩 중: {BASE_MODEL_ID}")

    # 1. 베이스 모델 로드 (학습 때와 동일하게 bfloat16 사용)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="eager"  # MPS 안정성을 위해 eager 모드 유지
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    # 2. LoRA 어댑터 장착 (개조된 뇌 합체)
    print(f"🔗 LoRA 어댑터 장착 중: {ADAPTER_PATH}")
    if not os.path.exists(ADAPTER_PATH):
        print("❌ 에러: 학습된 어댑터를 찾을 수 없어! 학습이 끝났는지 확인해줘.")
        return

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()  # 추론 모드 전환
    print("✅ 튜닝된 모델 준비 완료!")

    # 3. 테스트 이미지 준비
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 에러: 이미지가 없어! 경로 확인: {IMAGE_PATH}")
        return

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE_PATH},
                {"type": "text", "text": "Describe the defect in this image."}
            ]
        }
    ]

    # 4. 추론 실행
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)

    print("🔍 [After Tuning] 모델 분석 시작...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=50)
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

    result = output_text[0].split("assistant")[-1].strip()

    print("\n" + "=" * 50)
    print(f"📊 최종 결과: {result}")
    print("=" * 50)

    if "scratch" in result.lower():
        print("🎉 대성공! 모델이 이제 'Crack'이 아니라 'Scratch'라고 대답해!")
    else:
        print("🤔 결과가 애매하다면 학습 횟수(Epoch)를 조금 더 늘려보자.")


if __name__ == "__main__":
    run_test()