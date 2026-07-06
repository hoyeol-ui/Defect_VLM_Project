import sys
import types
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os


# --- [1] flash_attn 완벽 위장 (맥북 전용 패치) ---
def mock_flash_attn():
    module_name = 'flash_attn'
    if module_name not in sys.modules:
        flash_mock = types.ModuleType(module_name)
        flash_mock.__spec__ = types.SimpleNamespace(loader=None, origin=None, submodule_search_locations=None)
        flash_mock.flash_attn_func = lambda *args, **kwargs: None
        flash_mock.flash_attn_varlen_func = lambda *args, **kwargs: None
        sys.modules[module_name] = flash_mock
        sys.modules[f"{module_name}.flash_attn_interface"] = flash_mock
        sys.modules[f"{module_name}.bert_padding"] = flash_mock
        print("🛡️ 맥북 전용 flash_attn 위장막 설치 완료!")


mock_flash_attn()

# --- [2] 모델 로드 및 설정 ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 사용 중인 디바이스: {device}")

model_id = "microsoft/Florence-2-base"

try:
    print("⏳ 모델 로딩 중...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="sdpa",  # 맥북 MPS 최적화
        torch_dtype=torch.float16 if device == "mps" else torch.float32
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("✅ 모델 로드 성공!")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    sys.exit(1)


# --- [3] 추론 함수 ---
def run_example(image_path, task_prompt):
    if not os.path.exists(image_path):
        return f"파일 없음: {os.path.abspath(image_path)}"

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)

    if device == "mps":
        inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in inputs.items()}

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
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer


# --- [4] 메인 실행부: 3가지 테스트 동시 진행 ---
if __name__ == "__main__":
    test_image = "data/NEU-DET/train/images/scratches/scratches_1.jpg"

    if os.path.exists(test_image):
        print(f"\n🔍 분석 시작 이미지: {test_image}")
        print("=" * 50)

        # 테스트할 3가지 작업 리스트
        tasks = [
            ("<CAPTION>", "1. 일반 설명"),
            ("<DETAILED_CAPTION>", "2. 상세 설명"),
            ("<OD>", "3. 결함 위치 탐지(Object Detection)")
        ]

        for prompt, title in tasks:
            print(f"\n[{title}] 분석 중...")
            result = run_example(test_image, prompt)
            print(f"결과: {result}")

        print("\n" + "=" * 50)
        print("💡 분석 완료! '<OD>' 결과의 좌표값은 결함의 위치를 나타냅니다.")
        print("💡 결과가 이상하다면(예: 혜성), 튜닝이 꼭 필요하다는 강력한 증거입니다.")

    else:
        print(f"❌ 파일을 찾을 수 없습니다: {os.path.abspath(test_image)}")