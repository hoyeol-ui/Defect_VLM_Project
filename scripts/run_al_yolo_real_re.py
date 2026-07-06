import os
import shutil
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import torch  # 💡 이 한 줄이 빠져서 에러가 났습니다! 꼭 추가해 주세요.

# 💡 Ultralytics YOLOv8 라이브러리 임포트
from ultralytics import YOLO

# =============================================================================
# [1] 기본 설정 및 클래스 매핑
# =============================================================================
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path("/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project")
OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs"
YOLO_DATASET_DIR = PROJECT_ROOT / "datasets" / "al_dynamic_yolo"

# 시뮬레이션 하이퍼파라미터 (논문 스케일에 맞게 조정 가능)
INITIAL_SEED_SIZE = 30  # 처음 YOLO를 웜업 시킬 데이터 개수
AL_ROUNDS = 3  # 능동 학습 추가 라운드 수
SAMPLES_PER_ROUND = 15  # 매 라운드 추가할 데이터 개수
EPOCHS_PER_ROUND = 50  # 각 라운드별 훈련 에폭 (실험 속도를 위해 15로 세팅, 본 학습시 50 추천)

# NEU-DET & GC10-DET 통합 클래스 리스트 (YOLO 학습용)
CLASS_NAMES = [
    "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches",
    "punching_hole", "welding_line", "crescent_gap", "water_spot", "oil_spot",
    "silk_spot", "rolled_pit", "crease", "waist_folding"
]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# GC10 원본 라벨을 통합 영어 클래스로 변환하는 헬퍼 딕셔너리
GC10_RAW_MAP = {
    "1_chongkong": "punching_hole", "2_hanfeng": "welding_line", "3_yueyawan": "crescent_gap",
    "4_shuiban": "water_spot", "5_youban": "oil_spot", "6_siban": "silk_spot",
    "7_yiwu": "inclusion", "8_yahen": "rolled_pit", "9_zhehen": "crease", "10_yaozhe": "waist_folding"
}


# =============================================================================
# [2] 유틸리티 함수: XML -> YOLO TXT 변환 및 데이터셋 구축
# =============================================================================
def get_latest_results_file():
    json_files = list(OUTPUT_BASE_DIR.glob("**/pilot_grounded_consistency_results.json"))
    return max(json_files, key=lambda f: f.stat().st_mtime)


def calculate_priority_score(row):
    return (1.0 - row['consistency_score']) + (1.0 - (row['groundedness_score'] / 2.0))


def convert_xml_to_yolo(xml_path: Path, txt_path: Path, dataset_type: str):
    """XML 라벨을 읽어 YOLOv8 규격의 txt 파일로 변환합니다."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    dw = 1.0 / int(size.find("width").text)
    dh = 1.0 / int(size.find("height").text)

    with open(txt_path, "w") as f:
        for obj in root.findall("object"):
            raw_label = obj.find("name").text
            # 라벨 정규화
            if "GC10" in dataset_type.upper():
                class_name = GC10_RAW_MAP.get(raw_label, raw_label)
            else:
                class_name = raw_label

            if class_name not in CLASS_MAP:
                continue

            class_id = CLASS_MAP[class_name]
            bndbox = obj.find("bndbox")
            xmin, ymin = float(bndbox.find("xmin").text), float(bndbox.find("ymin").text)
            xmax, ymax = float(bndbox.find("xmax").text), float(bndbox.find("ymax").text)

            x_center = (xmin + xmax) / 2.0 * dw
            y_center = (ymin + ymax) / 2.0 * dh
            w = (xmax - xmin) * dw
            h = (ymax - ymin) * dh

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


def build_yolo_dataset(train_df: pd.DataFrame, val_df: pd.DataFrame, strategy: str, round_idx: int) -> Path:
    """YOLOv8이 학습할 수 있는 폴더 구조와 data.yaml을 동적으로 생성합니다."""
    dataset_path = YOLO_DATASET_DIR / f"{strategy}_round_{round_idx}"

    for split in ["train", "val"]:
        (dataset_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    def copy_and_convert(df, split):
        for _, row in df.iterrows():
            img_name = row['image_name']
            d_type = row['dataset_type']

            # 1. 원본 이미지 찾기 및 복사
            img_src = PROJECT_ROOT / "data" / d_type / ("IMAGES" if "NEU" in d_type else "") / img_name
            if not img_src.exists():  # GC10 하위 폴더 탐색
                img_src = list((PROJECT_ROOT / "data" / d_type).glob(f"**/{img_name}"))[0]
            shutil.copy(img_src, dataset_path / "images" / split / img_name)

            # 2. 원본 XML 찾기 및 YOLO txt 변환
            xml_dir = "ANNOTATIONS" if "NEU" in d_type else "lable"
            xml_src = PROJECT_ROOT / "data" / d_type / xml_dir / f"{Path(img_name).stem}.xml"
            txt_dst = dataset_path / "labels" / split / f"{Path(img_name).stem}.txt"
            convert_xml_to_yolo(xml_src, txt_dst, d_type)

    copy_and_convert(train_df, "train")
    copy_and_convert(val_df, "val")

    # 3. yaml 파일 생성
    yaml_path = dataset_path / "data.yaml"
    yaml_content = {
        "path": str(dataset_path.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)

    return yaml_path


# =============================================================================
# [3] 실제 YOLO 모델 학습 및 평가 래퍼 함수
# =============================================================================
def train_and_eval_real_yolo(yaml_path: Path, strategy: str, round_idx: int) -> float:
    print(f"\n🚀 [YOLOv8 Engine] 학습 시작: {strategy} - Round {round_idx}")

    # YOLOv8 nano 모델 로드 (가장 가볍고 빠른 모델)
    model = YOLO('yolov8n.pt')

    # 모델 학습
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS_PER_ROUND,
        imgsz=640,
        project=str(PROJECT_ROOT / "runs" / "active_learning"),
        name=f"{strategy}_R{round_idx}",
        device="mps" if torch.backends.mps.is_available() else "cpu",
        verbose=False  # 터미널 로그 최소화
    )

    # mAP50 추출
    map50 = round(results.box.map50, 4)
    print(f"🎯 [YOLOv8 Engine] 학습 완료! 검증 mAP@50: {map50}")
    return map50


# =============================================================================
# [4] 능동 학습 루프 (Active Learning Loop) 메인
# =============================================================================
def main():
    print(f"[*] 100% 리얼 물리 연산 YOLO Active Learning 파이프라인 가동")

    latest_json = get_latest_results_file()
    with open(latest_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    df_all = pd.DataFrame([{
        "image_name": item["image_name"],
        "dataset_type": item["dataset_type"],
        "consistency_score": item["consistency_score"],
        "groundedness_score": item["groundedness"]["total_score"]
    } for item in data])

    df_all["priority_score"] = df_all.apply(calculate_priority_score, axis=1)

    # 공정한 성능 비교를 위한 고정 검증셋 분리 (전체의 약 20%)
    val_size = int(len(df_all) * 0.2)
    df_val = df_all.sample(n=val_size, random_state=42)
    df_pool_initial = df_all.drop(df_val.index)

    print(f"[!] 고정 Validation Set 구축 완료: {len(df_val)}장")
    print(f"[!] 초기 Unlabeled Pool 규모: {len(df_pool_initial)}장")

    results = {"Random": {"budget": [], "mAP": []}, "VLM-Guided": {"budget": [], "mAP": []}}

    for strategy in ["Random", "VLM-Guided"]:
        print(f"\n" + "=" * 50)
        print(f"🌟 시작 전략: {strategy}")
        print("=" * 50)

        current_pool = df_pool_initial.copy()
        labeled_set = pd.DataFrame()

        # [Step 1] Initial Seed 선별 (양쪽 전략 모두 동일한 랜덤 조건 출발)
        seed_samples = current_pool.sample(n=min(INITIAL_SEED_SIZE, len(current_pool)), random_state=100)
        labeled_set = pd.concat([labeled_set, seed_samples])
        current_pool = current_pool.drop(seed_samples.index)

        # 데이터셋 폴더 구축 및 YOLO 학습
        yaml_path = build_yolo_dataset(labeled_set, df_val, strategy, round_idx=0)
        current_map = train_and_eval_real_yolo(yaml_path, strategy, 0)

        current_budget = len(labeled_set)
        results[strategy]["budget"].append(current_budget)
        results[strategy]["mAP"].append(current_map)

        # [Step 2] Active Learning Rounds
        for round_idx in range(1, AL_ROUNDS + 1):
            if len(current_pool) == 0:
                break

            sample_size = min(SAMPLES_PER_ROUND, len(current_pool))

            if strategy == "VLM-Guided":
                # 💡 순수 Hard Sample만 넣는 대신, VLM 추천(50%) + Random(50%) 하이브리드 전략
                n_hard = sample_size // 2
                n_rand = sample_size - n_hard

                hard_samples = current_pool.sort_values(by="priority_score", ascending=False).head(n_hard)
                current_pool_temp = current_pool.drop(hard_samples.index)
                rand_samples = current_pool_temp.sample(n=n_rand, random_state=100 + round_idx)

                selected = pd.concat([hard_samples, rand_samples])
            else:
                selected = current_pool.sample(n=sample_size, random_state=100 + round_idx)

            labeled_set = pd.concat([labeled_set, selected])
            current_pool = current_pool.drop(selected.index)

            yaml_path = build_yolo_dataset(labeled_set, df_val, strategy, round_idx)
            current_map = train_and_eval_real_yolo(yaml_path, strategy, round_idx)

            current_budget = len(labeled_set)
            results[strategy]["budget"].append(current_budget)
            results[strategy]["mAP"].append(current_map)

    # =========================================================================
    # [5] 실제 성능 곡선 (Learning Curve) 시각화
    # =========================================================================
    plt.figure(figsize=(8, 6))
    plt.plot(results["Random"]["budget"], results["Random"]["mAP"], marker='o', linestyle='--', color='gray',
             label='Random Sampling')
    plt.plot(results["VLM-Guided"]["budget"], results["VLM-Guided"]["mAP"], marker='s', linestyle='-', color='red',
             linewidth=2, label='VLM-Guided (Proposed)')

    plt.title("실제 YOLOv8 능동 학습 성능 비교 (Real Learning Curve)", fontsize=16)
    plt.xlabel("라벨링 예산 (사용된 훈련 이미지 수)", fontsize=12)
    plt.ylabel("YOLO 검출 성능 (mAP@50)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plot_path = latest_json.parent / "real_yolo_al_learning_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[최종 완료] 100% 실제 물리 연산 학습 곡선 저장 완료: {plot_path.name}")


if __name__ == "__main__":
    main()