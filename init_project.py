import os


def create_project_structure():
    # 1. 생성할 디렉토리 목록
    directories = [
        "data/NEU-DET/IMAGES",
        "data/NEU-DET/ANNOTATIONS",
        "models",
        "scripts"
    ]

    # 2. 디렉토리 생성
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created: {directory}")

    # 3. 생성할 파일 및 초기 내용 정의
    files_content = {
        "main.py": "# 메인 실행 스크립트\n\ndef main():\n    print('Defect VLM Project Started!')\n\nif __name__ == '__main__':\n    main()\n",

        "scripts/data_loader.py": "# NEU-DET XML 어노테이션 파싱 및 데이터 로더 정의\n\ndef load_data():\n    pass\n",

        "scripts/test_model.py": """import torch
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
"""
    }

    # 4. 파일 생성 및 내용 쓰기
    for file_path, content in files_content.items():
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"File created: {file_path}")

    print("\n✅ 프로젝트 구조 생성이 완료되었습니다!")
    print("이제 다운로드하신 NEU-DET의 .jpg 파일들을 data/NEU-DET/IMAGES로,")
    print(".xml 파일들을 data/NEU-DET/ANNOTATIONS로 옮겨주세요.")


if __name__ == "__main__":
    create_project_structure()