import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm  # 진행률 표시용 (없으면 pip install tqdm)


def normalize_bbox(bbox, width, height):
    """
    픽셀 좌표를 Florence-2가 이해하는 0~1000 좌표계로 변환합니다.
    bbox: [xmin, ymin, xmax, ymax]
    """
    x1, y1, x2, y2 = bbox

    # 0~1000 범위로 스케일링 (정수형)
    x1 = int((x1 / width) * 1000)
    y1 = int((y1 / height) * 1000)
    x2 = int((x2 / width) * 1000)
    y2 = int((y2 / height) * 1000)

    # 좌표가 1000을 넘지 않도록 클리핑
    return [
        max(0, min(1000, x1)),
        max(0, min(1000, y1)),
        max(0, min(1000, x2)),
        max(0, min(1000, y2))
    ]


def create_jsonl_dataset(xml_root, img_root, output_file):
    dataset = []

    # XML 파일 목록 가져오기
    xml_files = [f for f in os.listdir(xml_root) if f.endswith('.xml')]
    print(f"📂 총 {len(xml_files)}개의 XML 파일을 찾았습니다.")

    success_count = 0

    for xml_file in tqdm(xml_files, desc="데이터 변환 중"):
        try:
            # 1. XML 파싱
            tree = ET.parse(os.path.join(xml_root, xml_file))
            root = tree.getroot()

            # 2. 이미지 파일 찾기 (NEU-DET 구조 대응)
            filename = root.find('filename').text

            # Kaggle 데이터셋 구조에 맞춰 경로 탐색
            # 예: scratches_1.jpg -> images/scratches/scratches_1.jpg
            # 결함 종류(class) 폴더명 추론 (파일명의 _ 앞부분)
            defect_class = filename.split('_')[0]

            # 이미지 경로 구성 (사용자의 폴더 구조에 맞춰 조정 가능)
            image_path = os.path.join(img_root, defect_class, filename)

            # 만약 클래스 폴더가 없다면 바로 이미지 폴더에서 찾기 시도
            if not os.path.exists(image_path):
                image_path = os.path.join(img_root, filename)

            if not os.path.exists(image_path):
                # print(f"⚠️ 이미지 없음: {image_path}") # 너무 많이 뜨면 주석 처리
                continue

            # 3. 이미지 크기 확인 (좌표 정규화를 위해 필수)
            with Image.open(image_path) as img:
                width, height = img.size

            # 4. 정답(Suffix) 생성: Florence-2 형식
            # 형식: label<loc_x1><loc_y1><loc_x2><loc_y2>
            suffix_parts = []

            for obj in root.findall('object'):
                name = obj.find('name').text.lower()  # 소문자로 통일
                bndbox = obj.find('bndbox')

                bbox = [
                    float(bndbox.find('xmin').text),
                    float(bndbox.find('ymin').text),
                    float(bndbox.find('xmax').text),
                    float(bndbox.find('ymax').text)
                ]

                # 좌표 정규화 (0~1000)
                norm_box = normalize_bbox(bbox, width, height)

                # Florence-2 위치 토큰 포맷: <loc_0> ~ <loc_999>
                loc_token = f"<loc_{norm_box[0]}><loc_{norm_box[1]}><loc_{norm_box[2]}><loc_{norm_box[3]}>"
                suffix_parts.append(f"{name}{loc_token}")

            # 하나의 이미지에 여러 결함이 있을 수 있음
            suffix = "".join(suffix_parts)

            # 5. 데이터셋에 추가
            # prefix: 모델에게 시킬 작업 (<OD>: 객체 탐지)
            entry = {
                "image": image_path,
                "prefix": "<OD>",
                "suffix": suffix
            }
            dataset.append(entry)
            success_count += 1

        except Exception as e:
            print(f"❌ 에러 발생 ({xml_file}): {e}")
            continue

    # JSONL 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')

    print(f"\n🎉 변환 완료! 총 {success_count}개의 데이터가 '{output_file}'에 저장되었습니다.")
    print(f"👉 첫 번째 데이터 예시:\n{json.dumps(dataset[0], indent=2)}")


if __name__ == "__main__":
    # 프로젝트 루트 경로 자동 계산
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # 입력/출력 경로 설정 (호열님의 NEU-DET 경로에 맞게 설정)
    # XML이 있는 폴더
    xml_dir = os.path.join(project_root, "data", "NEU-DET", "train", "annotations")
    # 이미지가 있는 최상위 폴더 (이 안에 scratches, pitted 등 폴더가 있을 것으로 예상)
    img_dir = os.path.join(project_root, "data", "NEU-DET", "train", "images")

    output_jsonl = os.path.join(project_root, "data", "train_dataset.jsonl")

    if os.path.exists(xml_dir) and os.path.exists(img_dir):
        create_jsonl_dataset(xml_dir, img_dir, output_jsonl)
    else:
        print("❌ 경로를 찾을 수 없습니다. 아래 경로를 확인해주세요.")
        print(f"XML Dir: {xml_dir}")
        print(f"IMG Dir: {img_dir}")