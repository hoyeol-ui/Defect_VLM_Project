"""
===============================================================================
[File] run_al_yolo_ablation_v2.py

[Purpose]
Run YOLOv8 Active Learning ablation with stronger experimental controls.

V2 improvements:
1. Shared Round-0 baseline
   - Round 0 is trained/evaluated once and copied to all strategies.
   - This prevents strategy-wise baseline mismatch.

2. Capped selection strategies
   - no_pseudo_box means missing visual evidence, not necessarily hard sample.
   - Limit no_pseudo_box ratio per acquisition round.

3. DRY_RUN_ONLY support
   - Current default: True
   - Builds datasets and checks selection without training YOLO.

Strategies:
1. Random
2. ConsistencyOnly
3. CombinedStrict
4. CombinedSoftPenalty
5. CombinedSoftPenaltyCapped
6. SoftPenaltyHybrid50
7. SoftPenaltyHybridCapped

Input:
    outputs/pseudo_boxes_*/priority_scores_pseudo.csv

Output:
    runs/active_learning_ablation_v2/al_ablation_v2_YYYYMMDD_HHMMSS/
        config.json
        fixed_validation_split.csv
        initial_pool_split.csv
        shared_initial_seed_samples.csv
        shared_validation_samples.csv
        al_round_results.csv
        selected_samples_by_round.csv
        dataset_build_log.csv
        selection_reason_distribution.png
        selection_dataset_distribution.png
        selection_class_hint_distribution.png
        al_ablation_v2_summary.md

Important:
    GT XML labels are used only after samples are selected,
    simulating annotation. GT is not used for acquisition scoring.
===============================================================================
"""

import re
import json
import yaml
import shutil
import random
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image

try:
    import torch
except Exception:
    torch = None

try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError(
        "ultralytics가 설치되어 있지 않습니다. 먼저 아래 명령으로 설치하세요:\n"
        "pip install ultralytics"
    ) from e


# =============================================================================
# [0] Project settings
# =============================================================================
PROJECT_ROOT = Path("/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project")
OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs"
DATA_ROOT = PROJECT_ROOT / "data"

RUNS_ROOT = PROJECT_ROOT / "runs" / "active_learning_ablation_v2"
YOLO_DATASETS_ROOT = PROJECT_ROOT / "datasets" / "al_yolo_ablation_v2"

# -------------------------------------------------------------------------
# V2 pilot hyperparameters
# -------------------------------------------------------------------------
RANDOM_SEED = 42
VAL_RATIO = 0.20

INITIAL_SEED_SIZE = 30
AL_ROUNDS = 3
QUERY_SIZE = 10

EPOCHS_PER_ROUND = 30
IMGSZ = 640
BATCH_SIZE = 4
WORKERS = 0

YOLO_MODEL_NAME = "yolov8n.pt"

# 핵심: 지금은 반드시 True로 시작
DRY_RUN_ONLY = False

# True 권장: label 변환 문제를 초기에 잡기 위함
STRICT_LABEL_CHECK = True

# Capped strategy 설정
MAX_NO_PSEUDO_RATIO = 0.4

STRATEGIES_TO_RUN = [
    "Random",
    "CombinedSoftPenalty",
    "CombinedSoftPenaltyCapped",
    "SoftPenaltyHybridCapped",
]


# =============================================================================
# [1] Class names and mapping
# =============================================================================
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
    "punching_hole",
    "welding_line",
    "crescent_gap",
    "water_spot",
    "oil_spot",
    "silk_spot",
    "rolled_pit",
    "crease",
    "waist_folding",
]

CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

GC10_DIGIT_MAP = {
    "1": "punching_hole",
    "2": "welding_line",
    "3": "crescent_gap",
    "4": "water_spot",
    "5": "oil_spot",
    "6": "silk_spot",
    "7": "inclusion",
    "8": "rolled_pit",
    "9": "crease",
    "10": "waist_folding",
}

GC10_RAW_MAP = {
    "1_chongkong": "punching_hole",
    "2_hanfeng": "welding_line",
    "3_yueyawan": "crescent_gap",
    "4_shuiban": "water_spot",
    "5_youban": "oil_spot",
    "6_siban": "silk_spot",
    "7_yiwu": "inclusion",
    "8_yahen": "rolled_pit",
    "9_zhehen": "crease",
    "10_yaozhe": "waist_folding",

    "chongkong": "punching_hole",
    "hanfeng": "welding_line",
    "yueyawan": "crescent_gap",
    "shuiban": "water_spot",
    "youban": "oil_spot",
    "siban": "silk_spot",
    "yiwu": "inclusion",
    "yahen": "rolled_pit",
    "zhehen": "crease",
    "yaozhe": "waist_folding",
}


# =============================================================================
# [2] Font / reproducibility
# =============================================================================
def set_korean_font():
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    candidates = [
        "AppleGothic",
        "NanumGothic",
        "Malgun Gothic",
        "Noto Sans CJK KR",
        "DejaVu Sans",
    ]

    for font in candidates:
        if font in available_fonts:
            plt.rcParams["font.family"] = font
            break

    plt.rcParams["axes.unicode_minus"] = False


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def get_device():
    if torch is None:
        return "cpu"

    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    try:
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass

    return "cpu"


# =============================================================================
# [3] File loading
# =============================================================================
def find_latest_file(pattern: str) -> Path:
    files = list(OUTPUT_BASE_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return max(files, key=lambda p: p.stat().st_mtime)


def load_priority_scores():
    priority_csv = find_latest_file("pseudo_boxes_*/priority_scores_pseudo.csv")
    df = pd.read_csv(priority_csv)

    required_cols = [
        "image_name",
        "dataset_type",
        "image_path",
        "consistency_score",
        "groundedness_reason",
        "score_consistency_only",
        "score_combined_strict",
        "score_combined_soft_penalty",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"priority_scores_pseudo.csv에 필요한 컬럼이 없습니다: {missing_cols}")

    return priority_csv, prepare_priority_dataframe(df)


def prepare_priority_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "consistency_score",
        "groundedness_norm",
        "groundedness_effective_strict",
        "groundedness_effective_soft",
        "missing_box_penalty",
        "uncertainty_consistency",
        "uncertainty_groundedness_strict",
        "uncertainty_groundedness_soft",
        "score_consistency_only",
        "score_groundedness_strict",
        "score_combined_strict",
        "score_combined_soft_missing",
        "score_combined_soft_penalty",
        "best_box_score",
        "best_box_quality",
        "best_box_area_ratio",
        "best_box_aspect_ratio",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["dataset_type"] = df["dataset_type"].fillna("unknown")
    df["groundedness_reason"] = df["groundedness_reason"].fillna("unknown")
    df["image_path"] = df["image_path"].astype(str)

    df["class_hint"] = df.apply(infer_class_hint, axis=1)

    df = df.drop_duplicates(subset=["image_name", "dataset_type"]).reset_index(drop=True)

    return df


# =============================================================================
# [4] Dataset / XML utilities
# =============================================================================
def infer_class_hint(row: pd.Series) -> str:
    dataset_type = str(row.get("dataset_type", ""))
    image_name = str(row.get("image_name", ""))
    image_path = Path(str(row.get("image_path", "")))

    if "NEU" in dataset_type.upper():
        stem = Path(image_name).stem
        return re.sub(r"_\d+$", "", stem)

    if "GC10" in dataset_type.upper():
        parent = image_path.parent.name
        return GC10_DIGIT_MAP.get(parent, f"GC10_{parent}")

    return "unknown"


def find_image_path(row: pd.Series) -> Path:
    image_path = Path(str(row["image_path"]))

    if image_path.exists():
        return image_path

    dataset_type = str(row["dataset_type"])
    image_name = str(row["image_name"])

    candidates = []

    if "NEU" in dataset_type.upper():
        candidates.append(DATA_ROOT / "NEU-DET" / "IMAGES" / image_name)
    elif "GC10" in dataset_type.upper():
        candidates.extend(list((DATA_ROOT / "GC10-DET").glob(f"**/{image_name}")))

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(f"Image not found: {image_name} / {image_path}")


def find_xml_path(row: pd.Series, image_path: Path):
    dataset_type = str(row["dataset_type"])
    image_name = str(row["image_name"])
    stem = Path(image_name).stem

    candidates = []

    if "NEU" in dataset_type.upper():
        candidates.extend([
            DATA_ROOT / "NEU-DET" / "ANNOTATIONS" / f"{stem}.xml",
            DATA_ROOT / "NEU-DET" / "Annotations" / f"{stem}.xml",
            DATA_ROOT / "NEU-DET" / "annotations" / f"{stem}.xml",
        ])

    elif "GC10" in dataset_type.upper():
        candidates.extend([
            DATA_ROOT / "GC10-DET" / "lable" / f"{stem}.xml",
            DATA_ROOT / "GC10-DET" / "label" / f"{stem}.xml",
            DATA_ROOT / "GC10-DET" / "labels" / f"{stem}.xml",
            DATA_ROOT / "GC10-DET" / "ANNOTATIONS" / f"{stem}.xml",
            DATA_ROOT / "GC10-DET" / "Annotations" / f"{stem}.xml",
            DATA_ROOT / "GC10-DET" / "annotations" / f"{stem}.xml",
            image_path.with_suffix(".xml"),
        ])

        candidates.extend(list((DATA_ROOT / "GC10-DET").glob(f"**/{stem}.xml")))

    else:
        candidates.extend(list(DATA_ROOT.glob(f"**/{stem}.xml")))

    for c in candidates:
        if c.exists():
            return c

    return None


def get_image_size_from_xml_or_image(root, image_path: Path):
    size = root.find("size")

    if size is not None:
        w_node = size.find("width")
        h_node = size.find("height")

        if w_node is not None and h_node is not None:
            try:
                w = int(float(w_node.text))
                h = int(float(h_node.text))
                if w > 0 and h > 0:
                    return w, h
            except Exception:
                pass

    with Image.open(image_path) as img:
        return img.size


def map_class_name(raw_label: str, dataset_type: str, image_path: Path):
    raw = str(raw_label).strip()
    raw_lower = raw.lower()

    if raw in CLASS_MAP:
        return raw

    if raw_lower in CLASS_MAP:
        return raw_lower

    if raw in GC10_RAW_MAP:
        return GC10_RAW_MAP[raw]

    if raw_lower in GC10_RAW_MAP:
        return GC10_RAW_MAP[raw_lower]

    if "GC10" in dataset_type.upper():
        parent = image_path.parent.name

        if parent in GC10_DIGIT_MAP:
            return GC10_DIGIT_MAP[parent]

        m = re.match(r"^(\d+)", raw)
        if m:
            digit = m.group(1)
            if digit in GC10_DIGIT_MAP:
                return GC10_DIGIT_MAP[digit]

    return None


def convert_xml_to_yolo(
    xml_path: Path,
    txt_path: Path,
    dataset_type: str,
    image_path: Path,
) -> dict:
    if xml_path is None or not xml_path.exists():
        if STRICT_LABEL_CHECK:
            raise FileNotFoundError(f"XML label not found for image: {image_path}")

        txt_path.write_text("", encoding="utf-8")
        return {
            "xml_found": False,
            "num_objects": 0,
            "num_written": 0,
            "unknown_labels": [],
        }

    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_width, image_height = get_image_size_from_xml_or_image(root, image_path)

    yolo_lines = []
    unknown_labels = []
    num_objects = 0

    for obj in root.findall("object"):
        num_objects += 1

        name_node = obj.find("name")
        if name_node is None:
            unknown_labels.append("NO_NAME_NODE")
            continue

        raw_label = name_node.text
        class_name = map_class_name(raw_label, dataset_type, image_path)

        if class_name not in CLASS_MAP:
            unknown_labels.append(str(raw_label))
            continue

        class_id = CLASS_MAP[class_name]

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        except Exception:
            continue

        xmin = max(0.0, min(xmin, image_width - 1))
        ymin = max(0.0, min(ymin, image_height - 1))
        xmax = max(0.0, min(xmax, image_width - 1))
        ymax = max(0.0, min(ymax, image_height - 1))

        if xmax <= xmin or ymax <= ymin:
            continue

        x_center = ((xmin + xmax) / 2.0) / image_width
        y_center = ((ymin + ymax) / 2.0) / image_height
        box_w = (xmax - xmin) / image_width
        box_h = (ymax - ymin) / image_height

        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
        )

    txt_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

    if STRICT_LABEL_CHECK and num_objects > 0 and len(yolo_lines) == 0:
        raise ValueError(
            f"XML은 있으나 YOLO label로 변환된 객체가 없습니다.\n"
            f"image={image_path}\n"
            f"xml={xml_path}\n"
            f"unknown_labels={unknown_labels}"
        )

    return {
        "xml_found": True,
        "num_objects": num_objects,
        "num_written": len(yolo_lines),
        "unknown_labels": unknown_labels,
    }


def make_safe_file_stem(row: pd.Series) -> str:
    dataset = str(row["dataset_type"]).replace("-", "_")
    stem = Path(str(row["image_name"])).stem
    return f"{dataset}__{stem}"


def clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_yolo_dataset(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    dataset_dir: Path,
):
    clean_dir(dataset_dir)

    for split in ["train", "val"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    build_logs = []

    def copy_and_convert(df_split: pd.DataFrame, split: str):
        for _, row in df_split.iterrows():
            image_path = find_image_path(row)
            xml_path = find_xml_path(row, image_path)

            safe_stem = make_safe_file_stem(row)
            img_ext = image_path.suffix.lower()

            if img_ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
                img_ext = ".jpg"

            img_dst = dataset_dir / "images" / split / f"{safe_stem}{img_ext}"
            txt_dst = dataset_dir / "labels" / split / f"{safe_stem}.txt"

            shutil.copy2(image_path, img_dst)

            status = convert_xml_to_yolo(
                xml_path=xml_path,
                txt_path=txt_dst,
                dataset_type=str(row["dataset_type"]),
                image_path=image_path,
            )

            build_logs.append({
                "split": split,
                "image_name": row["image_name"],
                "dataset_type": row["dataset_type"],
                "class_hint": row.get("class_hint", "unknown"),
                "image_src": str(image_path),
                "image_dst": str(img_dst),
                "xml_path": str(xml_path) if xml_path else None,
                "label_dst": str(txt_dst),
                "xml_found": status["xml_found"],
                "num_objects": status["num_objects"],
                "num_written": status["num_written"],
                "unknown_labels": "|".join(status["unknown_labels"]),
            })

    copy_and_convert(train_df, "train")
    copy_and_convert(val_df, "val")

    yaml_path = dataset_dir / "data.yaml"

    data_yaml = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    build_log_df = pd.DataFrame(build_logs)

    if STRICT_LABEL_CHECK:
        empty_train = build_log_df[
            (build_log_df["split"] == "train") &
            (build_log_df["num_written"] <= 0)
        ]

        if len(empty_train) > 0:
            raise ValueError(
                f"Train split에 label이 없는 이미지가 있습니다.\n"
                f"{empty_train[['image_name', 'dataset_type', 'xml_path', 'unknown_labels']].head(20)}"
            )

    return yaml_path, build_log_df


# =============================================================================
# [5] Active Learning selection strategies
# =============================================================================
def select_capped_by_reason(
    current_pool: pd.DataFrame,
    score_col: str,
    sample_size: int,
    max_no_pseudo_ratio: float = 0.4,
    random_fill: bool = False,
    round_idx: int = 0,
) -> pd.DataFrame:
    """
    Select high-priority samples while limiting no_pseudo_box samples.

    Motivation:
        no_pseudo_box means missing visual evidence.
        It should not dominate the entire acquisition batch.

    Example:
        query_size=10, max_no_pseudo_ratio=0.4
        -> no_pseudo_box at most 4 samples if possible.
    """
    if len(current_pool) == 0:
        return current_pool.copy()

    sample_size = min(sample_size, len(current_pool))

    max_no_pseudo = int(sample_size * max_no_pseudo_ratio)
    min_non_missing = sample_size - max_no_pseudo

    pool_sorted = current_pool.sort_values(score_col, ascending=False)

    non_missing = pool_sorted[
        pool_sorted["groundedness_reason"] != "no_pseudo_box"
    ]

    missing = pool_sorted[
        pool_sorted["groundedness_reason"] == "no_pseudo_box"
    ]

    selected_parts = []

    selected_non_missing = non_missing.head(min_non_missing)
    selected_parts.append(selected_non_missing)

    selected_missing = missing.head(max_no_pseudo)
    selected_parts.append(selected_missing)

    selected = pd.concat(selected_parts).drop_duplicates()

    if len(selected) < sample_size:
        remaining = current_pool.drop(selected.index)
        n_fill = sample_size - len(selected)

        if random_fill:
            fill = remaining.sample(
                n=min(n_fill, len(remaining)),
                random_state=RANDOM_SEED + round_idx * 107,
            )
        else:
            fill = remaining.sort_values(score_col, ascending=False).head(n_fill)

        selected = pd.concat([selected, fill]).drop_duplicates()

    selected = selected.sort_values(score_col, ascending=False).head(sample_size)

    return selected


def select_samples(
    strategy: str,
    current_pool: pd.DataFrame,
    sample_size: int,
    round_idx: int,
) -> pd.DataFrame:
    if len(current_pool) == 0:
        return current_pool.copy()

    sample_size = min(sample_size, len(current_pool))

    if strategy == "Random":
        return current_pool.sample(
            n=sample_size,
            random_state=RANDOM_SEED + round_idx * 101,
        )

    if strategy == "ConsistencyOnly":
        return current_pool.sort_values(
            "score_consistency_only",
            ascending=False,
        ).head(sample_size)

    if strategy == "CombinedStrict":
        return current_pool.sort_values(
            "score_combined_strict",
            ascending=False,
        ).head(sample_size)

    if strategy == "CombinedSoftPenalty":
        return current_pool.sort_values(
            "score_combined_soft_penalty",
            ascending=False,
        ).head(sample_size)

    if strategy == "CombinedSoftPenaltyCapped":
        return select_capped_by_reason(
            current_pool=current_pool,
            score_col="score_combined_soft_penalty",
            sample_size=sample_size,
            max_no_pseudo_ratio=MAX_NO_PSEUDO_RATIO,
            random_fill=False,
            round_idx=round_idx,
        )

    if strategy == "SoftPenaltyHybrid50":
        n_hard = sample_size // 2
        n_random = sample_size - n_hard

        hard = current_pool.sort_values(
            "score_combined_soft_penalty",
            ascending=False,
        ).head(n_hard)

        remaining = current_pool.drop(hard.index)

        if len(remaining) > 0 and n_random > 0:
            rand = remaining.sample(
                n=min(n_random, len(remaining)),
                random_state=RANDOM_SEED + round_idx * 103,
            )
            return pd.concat([hard, rand]).drop_duplicates().head(sample_size)

        return hard

    if strategy == "SoftPenaltyHybridCapped":
        n_hard = sample_size // 2
        n_random = sample_size - n_hard

        hard = select_capped_by_reason(
            current_pool=current_pool,
            score_col="score_combined_soft_penalty",
            sample_size=n_hard,
            max_no_pseudo_ratio=MAX_NO_PSEUDO_RATIO,
            random_fill=False,
            round_idx=round_idx,
        )

        remaining = current_pool.drop(hard.index)

        if len(remaining) > 0 and n_random > 0:
            rand = remaining.sample(
                n=min(n_random, len(remaining)),
                random_state=RANDOM_SEED + round_idx * 109,
            )
            return pd.concat([hard, rand]).drop_duplicates().head(sample_size)

        return hard

    raise ValueError(f"Unknown strategy: {strategy}")


# =============================================================================
# [6] YOLO training and validation
# =============================================================================
def train_and_eval_yolo(
    yaml_path: Path,
    run_output_dir: Path,
    strategy: str,
    round_idx: int,
) -> dict:
    device = get_device()

    print("\n" + "-" * 80)
    print(f"[YOLO] strategy={strategy} | round={round_idx}")
    print(f"[YOLO] yaml={yaml_path}")
    print(f"[YOLO] device={device}")
    print("-" * 80)

    if DRY_RUN_ONLY:
        print("[DRY RUN] YOLO 학습 생략")
        return {
            "map50": np.nan,
            "map5095": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "train_status": "dry_run",
            "train_run_dir": None,
            "error": None,
        }

    try:
        model = YOLO(YOLO_MODEL_NAME)

        train_results = model.train(
            data=str(yaml_path),
            epochs=EPOCHS_PER_ROUND,
            imgsz=IMGSZ,
            batch=BATCH_SIZE,
            workers=WORKERS,
            device=device,
            project=str(run_output_dir / "yolo_train_runs"),
            name=f"{strategy}_R{round_idx}",
            exist_ok=True,
            patience=5,
            cache=False,
            verbose=False,
            seed=RANDOM_SEED,
        )

        train_run_dir = getattr(train_results, "save_dir", None)
        train_run_dir = Path(train_run_dir) if train_run_dir is not None else None

        best_weight = None

        if train_run_dir is not None:
            candidate = train_run_dir / "weights" / "best.pt"
            if candidate.exists():
                best_weight = candidate

        if best_weight is not None:
            eval_model = YOLO(str(best_weight))
        else:
            eval_model = model

        metrics = eval_model.val(
            data=str(yaml_path),
            imgsz=IMGSZ,
            batch=BATCH_SIZE,
            workers=WORKERS,
            device=device,
            split="val",
            verbose=False,
        )

        map50 = float(metrics.box.map50)
        map5095 = float(metrics.box.map)
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)

        print(f"[YOLO] 완료 | mAP50={map50:.4f}, mAP50-95={map5095:.4f}")

        return {
            "map50": round(map50, 6),
            "map5095": round(map5095, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "train_status": "success",
            "train_run_dir": str(train_run_dir) if train_run_dir else None,
            "error": None,
        }

    except Exception as e:
        print("[ERROR] YOLO 학습/평가 중 오류 발생")
        print(str(e))
        traceback.print_exc()

        return {
            "map50": np.nan,
            "map5095": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "train_status": "failed",
            "train_run_dir": None,
            "error": str(e),
        }


# =============================================================================
# [7] Plotting and report
# =============================================================================
def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {path}")


def plot_learning_curve(results_df: pd.DataFrame, save_dir: Path, metric: str):
    plt.figure(figsize=(9, 6))

    for strategy, sub in results_df.groupby("strategy"):
        sub = sub.sort_values("round")
        plt.plot(
            sub["labeled_budget"],
            sub[metric],
            marker="o",
            linewidth=2,
            label=strategy,
        )

    title_metric = "mAP@50" if metric == "map50" else "mAP@50-95"
    plt.title(f"YOLOv8 Active Learning V2: {title_metric}", fontsize=15, fontweight="bold")
    plt.xlabel("Number of labeled training samples")
    plt.ylabel(title_metric)
    plt.grid(alpha=0.3)
    plt.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")

    save_fig(save_dir / f"learning_curve_{metric}.png")


def plot_final_map_comparison(results_df: pd.DataFrame, save_dir: Path):
    final_rows = (
        results_df.sort_values("round")
        .groupby("strategy")
        .tail(1)
        .sort_values("map50", ascending=False)
    )

    plt.figure(figsize=(10, 5))
    bars = plt.bar(final_rows["strategy"], final_rows["map50"])
    plt.title("Final round mAP@50 by acquisition strategy", fontsize=15, fontweight="bold")
    plt.xlabel("Strategy")
    plt.ylabel("Final mAP@50")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    save_fig(save_dir / "final_map_comparison.png")


def plot_selection_distribution(
    selected_df: pd.DataFrame,
    save_dir: Path,
    group_col: str,
    filename: str,
):
    if len(selected_df) == 0:
        return

    pivot = (
        selected_df.groupby(["strategy", "round", group_col])
        .size()
        .reset_index(name="count")
    )

    pivot["strategy_round"] = pivot["strategy"] + "_R" + pivot["round"].astype(str)

    plot_df = pivot.pivot(
        index="strategy_round",
        columns=group_col,
        values="count",
    ).fillna(0)

    ax = plot_df.plot(
        kind="bar",
        stacked=True,
        figsize=(13, 6),
        edgecolor="white",
    )

    plt.title(f"Selected sample distribution by {group_col}", fontsize=15, fontweight="bold")
    plt.xlabel("Strategy / round")
    plt.ylabel("Number of selected samples")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc="upper left")

    for container in ax.containers:
        ax.bar_label(container, label_type="center", fontsize=7)

    save_fig(save_dir / filename)


def write_summary_md(
    save_dir: Path,
    priority_csv: Path,
    results_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    config: dict,
):
    lines = []
    lines.append("# YOLO Active Learning Ablation V2 Summary\n")

    lines.append("## 1. 목적\n")
    lines.append(
        "V2 실험은 shared Round-0 baseline과 no_pseudo_box capped selection을 추가하여, "
        "기존 실험에서 관찰된 baseline mismatch와 missing evidence 편향 가능성을 검증한다.\n"
    )

    lines.append("## 2. 입력 파일\n")
    lines.append(f"- priority score CSV: `{priority_csv}`\n")

    lines.append("## 3. 실험 설정\n")
    for k, v in config.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    if len(results_df) > 0:
        lines.append("## 4. Round 결과 요약\n")
        for _, row in results_df.sort_values(["strategy", "round"]).iterrows():
            lines.append(
                f"- {row['strategy']} R{int(row['round'])}: "
                f"budget={int(row['labeled_budget'])}, "
                f"mAP50={row['map50']}, "
                f"mAP50-95={row['map5095']}, "
                f"status={row['train_status']}"
            )
        lines.append("")

    if len(selected_df) > 0:
        acquired = selected_df[selected_df["round"] > 0]
        lines.append("## 5. 선택 샘플 분포 요약\n")

        for strategy, sub in acquired.groupby("strategy"):
            reason_counts = dict(Counter(sub["groundedness_reason"]))
            dataset_counts = dict(Counter(sub["dataset_type"]))
            class_counts = dict(Counter(sub["class_hint"]))

            lines.append(f"### {strategy}")
            lines.append(f"- reason_counts: {reason_counts}")
            lines.append(f"- dataset_counts: {dataset_counts}")
            lines.append(f"- class_hint_counts: {class_counts}")
            lines.append("")

    lines.append("## 6. 해석 가이드\n")
    lines.append(
        "- CombinedSoftPenaltyCapped가 CombinedSoftPenalty보다 안정적이면, "
        "no_pseudo_box missing evidence는 acquisition에 반영하되 round별 비율 제한이 필요하다는 근거가 된다."
    )
    lines.append(
        "- SoftPenaltyHybridCapped가 SoftPenaltyHybrid50보다 안정적이면, "
        "hybrid selection에도 missing evidence cap이 필요하다는 해석이 가능하다."
    )
    lines.append(
        "- DRY_RUN_ONLY=True일 때 mAP 값은 NaN이며, 이 단계의 목적은 학습이 아니라 "
        "selection 분포와 label 변환 검증이다."
    )

    summary_path = save_dir / "al_ablation_v2_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVE] {summary_path}")


# =============================================================================
# [8] Main active learning loop
# =============================================================================
def make_fixed_val_split(df: pd.DataFrame):
    val_parts = []

    for dataset_type, sub in df.groupby("dataset_type"):
        n = max(1, int(len(sub) * VAL_RATIO))
        val_parts.append(sub.sample(n=min(n, len(sub)), random_state=RANDOM_SEED))

    val_df = pd.concat(val_parts).drop_duplicates()
    pool_df = df.drop(val_df.index)

    return pool_df.reset_index(drop=True), val_df.reset_index(drop=True)


def add_selection_metadata(
    selected: pd.DataFrame,
    strategy: str,
    round_idx: int,
    selection_type: str,
) -> pd.DataFrame:
    selected = selected.copy()
    selected.insert(0, "strategy", strategy)
    selected.insert(1, "round", round_idx)
    selected.insert(2, "selection_type", selection_type)
    selected.insert(3, "rank_in_selection", range(1, len(selected) + 1))
    return selected


def summarize_selected_batch(selected: pd.DataFrame, strategy: str, round_idx: int):
    reason_counts = dict(Counter(selected["groundedness_reason"]))
    dataset_counts = dict(Counter(selected["dataset_type"]))
    class_counts = dict(Counter(selected["class_hint"]))

    print(f"[SELECT] {strategy} R{round_idx}")
    print(f"  reason_counts : {reason_counts}")
    print(f"  dataset_counts: {dataset_counts}")
    print(f"  class_counts  : {class_counts}")


def main():
    set_korean_font()
    seed_everything(RANDOM_SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    priority_csv, df_all = load_priority_scores()

    run_output_dir = RUNS_ROOT / f"al_ablation_v2_{timestamp}"
    dataset_output_root = YOLO_DATASETS_ROOT / f"al_ablation_v2_{timestamp}"

    run_output_dir.mkdir(parents=True, exist_ok=True)
    dataset_output_root.mkdir(parents=True, exist_ok=True)

    config = {
        "VERSION": "v2",
        "YOLO_MODEL_NAME": YOLO_MODEL_NAME,
        "STRATEGIES_TO_RUN": ", ".join(STRATEGIES_TO_RUN),
        "num_scored_samples": len(df_all),
        "VAL_RATIO": VAL_RATIO,
        "INITIAL_SEED_SIZE": INITIAL_SEED_SIZE,
        "AL_ROUNDS": AL_ROUNDS,
        "QUERY_SIZE": QUERY_SIZE,
        "EPOCHS_PER_ROUND": EPOCHS_PER_ROUND,
        "IMGSZ": IMGSZ,
        "BATCH_SIZE": BATCH_SIZE,
        "DEVICE": get_device(),
        "DRY_RUN_ONLY": DRY_RUN_ONLY,
        "STRICT_LABEL_CHECK": STRICT_LABEL_CHECK,
        "MAX_NO_PSEUDO_RATIO": MAX_NO_PSEUDO_RATIO,
        "RANDOM_SEED": RANDOM_SEED,
    }

    with open(run_output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    print("=" * 100)
    print("[ACTIVE LEARNING YOLO ABLATION V2]")
    print(f"Priority CSV: {priority_csv}")
    print(f"Run output : {run_output_dir}")
    print(f"Dataset dir: {dataset_output_root}")
    print(f"DRY_RUN_ONLY: {DRY_RUN_ONLY}")
    print(f"Config     : {config}")
    print("=" * 100)

    df_pool_initial, df_val = make_fixed_val_split(df_all)

    print(f"Total scored samples : {len(df_all)}")
    print(f"Validation samples   : {len(df_val)}")
    print(f"Initial pool samples : {len(df_pool_initial)}")

    df_val.to_csv(run_output_dir / "fixed_validation_split.csv", index=False, encoding="utf-8-sig")
    df_pool_initial.to_csv(run_output_dir / "initial_pool_split.csv", index=False, encoding="utf-8-sig")

    initial_seed_df = df_pool_initial.sample(
        n=min(INITIAL_SEED_SIZE, len(df_pool_initial)),
        random_state=RANDOM_SEED + 999,
    )

    initial_seed_df[
        ["image_name", "dataset_type", "class_hint", "groundedness_reason"]
    ].to_csv(
        run_output_dir / "shared_initial_seed_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df_val[
        ["image_name", "dataset_type", "class_hint"]
    ].to_csv(
        run_output_dir / "shared_validation_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_round_results = []
    all_selected_rows = []
    all_build_logs = []

    # -------------------------------------------------------------------------
    # Shared Round-0 baseline
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("[SHARED BASELINE] Train initial seed only once")
    print("=" * 100)

    baseline_dataset_dir = dataset_output_root / "_SHARED_BASELINE" / "round_0"

    baseline_yaml_path, baseline_build_log_df = build_yolo_dataset(
        train_df=initial_seed_df,
        val_df=df_val,
        dataset_dir=baseline_dataset_dir,
    )

    baseline_build_log_df.insert(0, "strategy", "_SHARED_BASELINE")
    baseline_build_log_df.insert(1, "round", 0)
    all_build_logs.append(baseline_build_log_df)

    baseline_eval_result = train_and_eval_yolo(
        yaml_path=baseline_yaml_path,
        run_output_dir=run_output_dir,
        strategy="_SHARED_BASELINE",
        round_idx=0,
    )

    # -------------------------------------------------------------------------
    # Strategy loops
    # -------------------------------------------------------------------------
    for strategy in STRATEGIES_TO_RUN:
        print("\n" + "=" * 100)
        print(f"[STRATEGY START] {strategy}")
        print("=" * 100)

        current_pool = df_pool_initial.drop(initial_seed_df.index).copy()
        labeled_df = initial_seed_df.copy()

        seed_selected_log = add_selection_metadata(
            selected=initial_seed_df,
            strategy=strategy,
            round_idx=0,
            selection_type="shared_initial_seed_random",
        )
        all_selected_rows.append(seed_selected_log)

        for round_idx in range(0, AL_ROUNDS + 1):
            if round_idx > 0:
                if len(current_pool) == 0:
                    print(f"[{strategy}] current_pool empty. Stop.")
                    break

                selected = select_samples(
                    strategy=strategy,
                    current_pool=current_pool,
                    sample_size=QUERY_SIZE,
                    round_idx=round_idx,
                )

                summarize_selected_batch(selected, strategy, round_idx)

                selected_log = add_selection_metadata(
                    selected=selected,
                    strategy=strategy,
                    round_idx=round_idx,
                    selection_type=strategy,
                )
                all_selected_rows.append(selected_log)

                labeled_df = pd.concat([labeled_df, selected])
                current_pool = current_pool.drop(selected.index)

            print(f"\n[{strategy}] Round {round_idx}")
            print(f"  labeled samples: {len(labeled_df)}")
            print(f"  current pool   : {len(current_pool)}")
            print(f"  validation     : {len(df_val)}")

            dataset_dir = dataset_output_root / strategy / f"round_{round_idx}"

            yaml_path, build_log_df = build_yolo_dataset(
                train_df=labeled_df,
                val_df=df_val,
                dataset_dir=dataset_dir,
            )

            build_log_df.insert(0, "strategy", strategy)
            build_log_df.insert(1, "round", round_idx)
            all_build_logs.append(build_log_df)

            if round_idx == 0:
                eval_result = baseline_eval_result.copy()
                eval_result["train_status"] = "shared_baseline_dry_run" if DRY_RUN_ONLY else "shared_baseline"
                eval_result["error"] = None

                print(
                    f"[{strategy}] Round 0 uses shared baseline | "
                    f"mAP50={eval_result['map50']}, "
                    f"mAP50-95={eval_result['map5095']}"
                )
            else:
                eval_result = train_and_eval_yolo(
                    yaml_path=yaml_path,
                    run_output_dir=run_output_dir,
                    strategy=strategy,
                    round_idx=round_idx,
                )

            all_round_results.append({
                "strategy": strategy,
                "round": round_idx,
                "labeled_budget": len(labeled_df),
                "pool_remaining": len(current_pool),
                "val_size": len(df_val),
                "yaml_path": str(yaml_path),
                **eval_result,
            })

            pd.DataFrame(all_round_results).to_csv(
                run_output_dir / "al_round_results.csv",
                index=False,
                encoding="utf-8-sig",
            )

            if all_selected_rows:
                pd.concat(all_selected_rows, ignore_index=True).to_csv(
                    run_output_dir / "selected_samples_by_round.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            if all_build_logs:
                pd.concat(all_build_logs, ignore_index=True).to_csv(
                    run_output_dir / "dataset_build_log.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

    results_df = pd.DataFrame(all_round_results)
    selected_df = pd.concat(all_selected_rows, ignore_index=True) if all_selected_rows else pd.DataFrame()
    build_log_all_df = pd.concat(all_build_logs, ignore_index=True) if all_build_logs else pd.DataFrame()

    results_csv = run_output_dir / "al_round_results.csv"
    selected_csv = run_output_dir / "selected_samples_by_round.csv"
    build_log_csv = run_output_dir / "dataset_build_log.csv"

    results_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
    selected_df.to_csv(selected_csv, index=False, encoding="utf-8-sig")
    build_log_all_df.to_csv(build_log_csv, index=False, encoding="utf-8-sig")

    if not DRY_RUN_ONLY and len(results_df) > 0:
        plot_learning_curve(results_df, run_output_dir, metric="map50")
        plot_learning_curve(results_df, run_output_dir, metric="map5095")
        plot_final_map_comparison(results_df, run_output_dir)

    if len(selected_df) > 0:
        acquired_only = selected_df[selected_df["round"] > 0].copy()

        if len(acquired_only) > 0:
            plot_selection_distribution(
                acquired_only,
                run_output_dir,
                group_col="groundedness_reason",
                filename="selection_reason_distribution.png",
            )
            plot_selection_distribution(
                acquired_only,
                run_output_dir,
                group_col="dataset_type",
                filename="selection_dataset_distribution.png",
            )
            plot_selection_distribution(
                acquired_only,
                run_output_dir,
                group_col="class_hint",
                filename="selection_class_hint_distribution.png",
            )

    write_summary_md(
        save_dir=run_output_dir,
        priority_csv=priority_csv,
        results_df=results_df,
        selected_df=selected_df,
        config=config,
    )

    print("\n" + "=" * 100)
    print("[완료] YOLO Active Learning Ablation V2 종료")
    print(f"Results CSV : {results_csv}")
    print(f"Selected CSV: {selected_csv}")
    print(f"Build log   : {build_log_csv}")
    print(f"Output dir  : {run_output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()