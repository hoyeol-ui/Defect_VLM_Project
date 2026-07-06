"""
===============================================================================
[Project] VLM-Guided Active Learning for Industrial Defect Detection
[Version] V3 - Dynamic Hybrid Annealing & Advanced Visualization
===============================================================================
"""

import os
import shutil
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import torch

# Ultralytics YOLOv8
from ultralytics import YOLO

# ── 메서드 출처 메타 ──
METHOD_SOURCE_META = {
    "dynamic_hybrid_active_learning": {
        "official_repo": ["https://github.com/JordanAsh/badge", "https://github.com/we1pingyu/CALD"],
        "paper": [
            "Ash et al., 'Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds', ICLR 2020.",
            "Yu et al., 'Consistency-based Active Learning for Object Detection', CVPR 2022."
        ],
        "adaptation_note": (
            "초기 라운드에서 발생하는 Catastrophic Forgetting을 방지하기 위해, "
            "VLM 기반 Hard Sample 주입 비율을 라운드 진행에 따라 점진적으로 증가시키는 "
            "Dynamic Annealing Heuristic (Exploration to Exploitation) 스케줄링을 구현함."
        )
    },
    "yolov8_training_backend": {
        "official_repo": ["https://github.com/ultralytics/ultralytics"],
        "adaptation_note": "YOLOv8n 기반 On-the-fly Fine-tuning을 통한 능동 학습 효용성 물리 연산 증명."
    }
}

# =============================================================================
# [1] 기본 설정 및 클래스 매핑
# =============================================================================
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path("/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project")
OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs"
YOLO_DATASET_DIR = PROJECT_ROOT / "datasets" / "al_dynamic_yolo_v3"

# 시뮬레이션 하이퍼파라미터
INITIAL_SEED_SIZE = 30  # 초기 기초 체력용 Random Seed
AL_ROUNDS = 3  # 추가 라운드
SAMPLES_PER_ROUND = 15  # 회당 추가 데이터
EPOCHS_PER_ROUND = 50  # 모델 수렴을 위한 훈련 에폭

CLASS_NAMES = [
    "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches",
    "punching_hole", "welding_line", "crescent_gap", "water_spot", "oil_spot",
    "silk_spot", "rolled_pit", "crease", "waist_folding"
]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

GC10_RAW_MAP = {
    "1_chongkong": "punching_hole", "2_hanfeng": "welding_line", "3_yueyawan": "crescent_gap",
    "4_shuiban": "water_spot", "5_youban": "oil_spot", "6_siban": "silk_spot",
    "7_yiwu": "inclusion", "8_yahen": "rolled_pit", "9_zhehen": "crease", "10_yaozhe": "waist_folding"
}


# =============================================================================
# [2] 유틸리티 함수
# =============================================================================
def get_latest_results_file():
    json_files = list(OUTPUT_BASE_DIR.glob("**/pilot_grounded_consistency_results.json"))
    return max(json_files, key=lambda f: f.stat().st_mtime)


def calculate_priority_score(row):
    return (1.0 - row['consistency_score']) + (1.0 - (row['groundedness_score'] / 2.0))


def convert_xml_to_yolo(xml_path: Path, txt_path: Path, dataset_type: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    dw = 1.0 / int(size.find("width").text)
    dh = 1.0 / int(size.find("height").text)

    with open(txt_path, "w") as f:
        for obj in root.findall("object"):
            raw_label = obj.find("name").text
            class_name = GC10_RAW_MAP.get(raw_label, raw_label) if "GC10" in dataset_type.upper() else raw_label
            if class_name not in CLASS_MAP: continue

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
    dataset_path = YOLO_DATASET_DIR / f"{strategy}_round_{round_idx}"
    for split in ["train", "val"]:
        (dataset_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    def copy_and_convert(df, split):
        for _, row in df.iterrows():
            img_name = row['image_name']
            d_type = row['dataset_type']
            img_src = PROJECT_ROOT / "data" / d_type / ("IMAGES" if "NEU" in d_type else "") / img_name
            if not img_src.exists():
                img_src = list((PROJECT_ROOT / "data" / d_type).glob(f"**/{img_name}"))[0]
            shutil.copy(img_src, dataset_path / "images" / split / img_name)

            xml_dir = "ANNOTATIONS" if "NEU" in d_type else "lable"
            xml_src = PROJECT_ROOT / "data" / d_type / xml_dir / f"{Path(img_name).stem}.xml"
            txt_dst = dataset_path / "labels" / split / f"{Path(img_name).stem}.txt"
            convert_xml_to_yolo(xml_src, txt_dst, d_type)

    copy_and_convert(train_df, "train")
    copy_and_convert(val_df, "val")

    yaml_path = dataset_path / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(
            {"path": str(dataset_path.absolute()), "train": "images/train", "val": "images/val", "nc": len(CLASS_NAMES),
             "names": CLASS_NAMES}, f)
    return yaml_path


# =============================================================================
# [3] 실제 YOLO 모델 학습 및 평가 래퍼 함수 (NMS 딜레이 방어 포함)
# =============================================================================
def train_and_eval_real_yolo(yaml_path: Path, strategy: str, round_idx: int) -> float:
    print(f"\n🚀 [YOLOv8 Engine] 학습 시작: {strategy} - Round {round_idx}")
    model = YOLO('yolov8n.pt')

    # max_det=100 적용하여 NMS 폭주 경고 및 타임아웃 방지
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS_PER_ROUND,
        imgsz=640,
        project=str(PROJECT_ROOT / "runs" / "active_learning_v3"),
        name=f"{strategy}_R{round_idx}",
        device="mps" if torch.backends.mps.is_available() else "cpu",
        max_det=100,
        verbose=False
    )
    map50 = round(results.box.map50, 4)
    print(f"🎯 [YOLOv8 Engine] 학습 완료! 검증 mAP@50: {map50}")
    return map50


# =============================================================================
# [4] 능동 학습 루프 (Active Learning Loop) 메인
# =============================================================================
def main():
    print(f"[*] V3 파이프라인 가동: Dynamic Annealing & 랩미팅 리포터")

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

    val_size = int(len(df_all) * 0.2)
    df_val = df_all.sample(n=val_size, random_state=42)
    df_pool_initial = df_all.drop(df_val.index)

    # 보고서용 데이터 저장소
    results = {"Random": {"budget": [], "mAP": [], "vlm_added": [], "rand_added": []},
               "VLM-Guided": {"budget": [], "mAP": [], "vlm_added": [], "rand_added": []}}

    for strategy in ["Random", "VLM-Guided"]:
        print(f"\n" + "=" * 50)
        print(f"🌟 시작 전략: {strategy}")
        print("=" * 50)

        current_pool = df_pool_initial.copy()
        labeled_set = pd.DataFrame()

        # [Round 0] Initial Seed (웜업)
        seed_samples = current_pool.sample(n=min(INITIAL_SEED_SIZE, len(current_pool)), random_state=100)
        labeled_set = pd.concat([labeled_set, seed_samples])
        current_pool = current_pool.drop(seed_samples.index)

        yaml_path = build_yolo_dataset(labeled_set, df_val, strategy, round_idx=0)
        current_map = train_and_eval_real_yolo(yaml_path, strategy, 0)

        results[strategy]["budget"].append(len(labeled_set))
        results[strategy]["mAP"].append(current_map)
        results[strategy]["vlm_added"].append(0)
        results[strategy]["rand_added"].append(len(seed_samples))

        # [Round 1~3] Active Learning Rounds
        for round_idx in range(1, AL_ROUNDS + 1):
            if len(current_pool) == 0: break
            sample_size = min(SAMPLES_PER_ROUND, len(current_pool))

            if strategy == "VLM-Guided":
                # 💡 [핵심] Dynamic Annealing 스케줄링 (Round 1: 30%, Round 2: 50%, Round 3: 70%)
                vlm_ratio = min(0.1 + (0.2 * round_idx), 0.8)  # 0.3 -> 0.5 -> 0.7
                n_hard = int(sample_size * vlm_ratio)
                n_rand = sample_size - n_hard

                hard_samples = current_pool.sort_values(by="priority_score", ascending=False).head(n_hard)
                current_pool_temp = current_pool.drop(hard_samples.index)
                rand_samples = current_pool_temp.sample(n=n_rand, random_state=100 + round_idx)
                selected = pd.concat([hard_samples, rand_samples])

                results[strategy]["vlm_added"].append(n_hard)
                results[strategy]["rand_added"].append(n_rand)
            else:
                selected = current_pool.sample(n=sample_size, random_state=100 + round_idx)
                results[strategy]["vlm_added"].append(0)
                results[strategy]["rand_added"].append(sample_size)

            labeled_set = pd.concat([labeled_set, selected])
            current_pool = current_pool.drop(selected.index)

            yaml_path = build_yolo_dataset(labeled_set, df_val, strategy, round_idx)
            current_map = train_and_eval_real_yolo(yaml_path, strategy, round_idx)

            results[strategy]["budget"].append(len(labeled_set))
            results[strategy]["mAP"].append(current_map)

    # =========================================================================
    # [5] 랩미팅 보고용 다중 시각화 및 리포트 자동 생성
    # =========================================================================
    output_folder = latest_json.parent

    # 1. Learning Curve (mAP)
    plt.figure(figsize=(9, 6))
    plt.plot(results["Random"]["budget"], results["Random"]["mAP"], marker='o', linestyle='--', color='gray',
             label='Random Sampling')
    plt.plot(results["VLM-Guided"]["budget"], results["VLM-Guided"]["mAP"], marker='s', linestyle='-', color='red',
             linewidth=2, label='VLM-Guided (Dynamic Hybrid)')
    plt.title("YOLOv8 능동 학습 성능 검증 (Dynamic Annealing 적용)", fontsize=16, pad=15)
    plt.xlabel("라벨링 예산 (사용된 이미지 수)", fontsize=12)
    plt.ylabel("YOLO 검출 성능 (mAP@50)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    lc_path = output_folder / "v3_learning_curve.png"
    plt.savefig(lc_path, dpi=300, bbox_inches='tight')

    # 2. VLM-Guided Sampling Distribution (스케줄링 증명용 막대그래프)
    plt.figure(figsize=(9, 6))
    rounds = [f"Round {i}" for i in range(AL_ROUNDS + 1)]
    vlm_counts = results["VLM-Guided"]["vlm_added"]
    rand_counts = results["VLM-Guided"]["rand_added"]

    bars1 = plt.bar(rounds, rand_counts, color='lightgray', edgecolor='black', label='Random Exploration (탐색)')
    bars2 = plt.bar(rounds, vlm_counts, bottom=rand_counts, color='salmon', edgecolor='black',
                    label='VLM Hard Samples (활용)')

    plt.title("VLM-Guided 전략의 라운드별 데이터 획득 비율 (Dynamic Annealing)", fontsize=15, pad=15)
    plt.ylabel("획득된 이미지 수", fontsize=12)
    plt.legend(fontsize=11)

    # 막대 위에 퍼센트 라벨 추가
    for i in range(1, len(rounds)):
        total = vlm_counts[i] + rand_counts[i]
        vlm_pct = int((vlm_counts[i] / total) * 100)
        plt.text(i, rand_counts[i] + (vlm_counts[i] / 2), f"{vlm_pct}%", ha='center', va='center', color='black',
                 weight='bold')

    dist_path = output_folder / "v3_sampling_distribution.png"
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')

    # 3. CSV Report Export
    report_df = pd.DataFrame({
        "Round": rounds,
        "Total_Budget": results["Random"]["budget"],
        "Random_mAP": results["Random"]["mAP"],
        "VLM_Hybrid_mAP": results["VLM-Guided"]["mAP"],
        "VLM_Hard_Sample_Added": results["VLM-Guided"]["vlm_added"]
    })

    # 성능 격차 컬럼 추가
    report_df["mAP_Gap (VLM - Random)"] = report_df["VLM_Hybrid_mAP"] - report_df["Random_mAP"]

    csv_path = output_folder / "v3_al_performance_report.csv"
    report_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print("🎉 [V3 파이프라인 실험 종료 및 리포트 생성 완료]")
    print(f" 1. 성능 곡선 그래프: {lc_path.name}")
    print(f" 2. 샘플링 분포 그래프: {dist_path.name}")
    print(f" 3. 수치 리포트 CSV: {csv_path.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()