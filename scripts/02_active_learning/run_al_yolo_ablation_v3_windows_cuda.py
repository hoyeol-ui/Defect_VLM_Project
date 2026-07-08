"""
===============================================================================
[File] run_al_yolo_ablation_v3_windows_cuda.py

[Purpose]
Windows/CUDA entrypoint for the YOLOv8 Active Learning ablation.

This file keeps the same research logic as run_al_yolo_ablation_v3_minimal.py,
but avoids macOS-specific path/device assumptions for a Windows desktop GPU.

Recommended PowerShell run:
    $env:AL_PROJECT_ROOT="C:\path\to\Defect_VLM_Project"
    $env:AL_PRIORITY_CSV="outputs\priority_sensitivity_20260706_152020\penalty_0\priority_scores_pseudo.csv"
    $env:AL_YOLO_DEVICE="0"
    $env:AL_BATCH_SIZE="8"
    $env:AL_WORKERS="4"
    $env:AL_STRATEGIES="Random,RandomClassDatasetBalanced,ConsistencyOnly,ConsistencyOnlyClassDatasetBalanced"
    $env:AL_SEEDS="42,43,44,45,46,47,48,49"
    .\.venv\Scripts\python.exe scripts\02_active_learning\run_al_yolo_ablation_v3_windows_cuda.py

The script still follows the professor-feedback validation design:

1. Detector-level validation
   - Evaluate acquisition strategies by YOLOv8 mAP, not VLM error.

2. High / Random / Low comparison
   - Random
   - CombinedSoftPenalty: high-priority
   - LowPrioritySoft: low-priority

3. Consistency-only / Groundedness-only / Combined ablation
   - ConsistencyOnly
   - GroundednessOnlySoft
   - CombinedSoftPenalty

4. AULC reporting
   - Compute area under learning curve for mAP@50 and mAP@50-95.

5. Seed repetition
   - Repeat the same experiment over multiple seeds.

Recommended first run:
    $env:AL_DRY_RUN_ONLY="1"

After checking selection distributions and label conversion:
    DRY_RUN_ONLY = False

Input:
    outputs/pseudo_boxes_*/priority_scores_pseudo.csv

Output:
    runs/active_learning_ablation_v3_minimal/al_ablation_v3_minimal_YYYYMMDD_HHMMSS/
        config.json
        all_round_results.csv
        all_selected_samples_by_round.csv
        all_dataset_build_log.csv
        seed_strategy_metric_summary.csv
        aggregate_strategy_metric_summary.csv
        aggregate_learning_curve_map50.png
        aggregate_learning_curve_map5095.png
        final_map50_mean_std.png
        final_map5095_mean_std.png
        aulc_map50_mean_std.png
        aulc_map5095_mean_std.png
        selection_reason_distribution.png
        selection_dataset_distribution.png
        selection_class_hint_distribution.png
        al_ablation_v3_minimal_summary.md

Important:
    GT XML labels are used only after samples are selected,
    simulating annotation. GT is not used for acquisition scoring.
===============================================================================
"""

import re
import os
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

try:
    from strategy_metadata import (
        STRATEGY_METADATA,
        add_strategy_metadata_columns,
        get_strategy_metadata,
    )
except Exception:
    STRATEGY_METADATA = {}

    def get_strategy_metadata(strategy: str) -> dict:
        return {
            "display_name": strategy,
            "family": "Uncategorized",
            "role": "No strategy metadata registered",
        }

    def add_strategy_metadata_columns(df: pd.DataFrame, strategy_col: str = "strategy") -> pd.DataFrame:
        return df


# =============================================================================
# [0] Project settings
# =============================================================================
def resolve_project_root() -> Path:
    override = os.environ.get("AL_PROJECT_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"AL_PROJECT_ROOT does not exist: {root}")
        return root

    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = resolve_project_root()
OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs"
DATA_ROOT = PROJECT_ROOT / "data"

RUNS_ROOT = PROJECT_ROOT / "runs" / "active_learning_ablation_v3_minimal"
YOLO_DATASETS_ROOT = PROJECT_ROOT / "datasets" / "al_yolo_ablation_v3_minimal"

# -------------------------------------------------------------------------
# 핵심 설정
# -------------------------------------------------------------------------
def parse_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_int_list_env(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name)
    if not value:
        return default
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_strategy_env(default: list[str]) -> list[str]:
    value = os.environ.get("AL_STRATEGIES")
    if not value:
        return default
    return [v.strip() for v in value.split(",") if v.strip()]


DRY_RUN_ONLY = parse_bool_env("AL_DRY_RUN_ONLY", False)
STRICT_LABEL_CHECK = True

SEEDS = parse_int_list_env("AL_SEEDS", [42, 43, 44])

VAL_RATIO = 0.20

INITIAL_SEED_SIZE = 30
AL_ROUNDS = 3
QUERY_SIZE = 10

EPOCHS_PER_ROUND = int(os.environ.get("AL_EPOCHS_PER_ROUND", "30"))

IMGSZ = int(os.environ.get("AL_IMGSZ", "640"))
BATCH_SIZE = int(os.environ.get("AL_BATCH_SIZE", "8"))
WORKERS = int(os.environ.get("AL_WORKERS", "4"))

YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_DEVICE_OVERRIDE = os.environ.get("AL_YOLO_DEVICE", "0")

# 교수님 피드백 대응용 최소 전략 세트
DEFAULT_STRATEGIES_TO_RUN = [
    "Random",
    "ConsistencyOnly",
    "GroundednessOnlySoft",
    "CombinedSoftPenalty",
    "LowPrioritySoft",
]

STRATEGIES_TO_RUN = parse_strategy_env(DEFAULT_STRATEGIES_TO_RUN)

PRIORITY_CSV_OVERRIDE = os.environ.get("AL_PRIORITY_CSV")

WEIGHTED_ALPHA = float(os.environ.get("AL_WEIGHTED_ALPHA", "1.0"))
WEIGHTED_BETA = float(os.environ.get("AL_WEIGHTED_BETA", "0.5"))
WEIGHTED_GAMMA = float(os.environ.get("AL_WEIGHTED_GAMMA", "0.2"))
SUPPRESS_NO_PSEUDO_GAMMA = float(os.environ.get("AL_SUPPRESS_NO_PSEUDO_GAMMA", "0.2"))


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
# [2] Utilities
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


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def parse_device_override(value: str):
    normalized = value.strip().lower()
    if normalized in {"cpu", "mps", "cuda"}:
        return normalized
    if normalized.startswith("cuda:"):
        return normalized
    if normalized.isdigit():
        return int(normalized)
    return value.strip()


def get_device():
    if YOLO_DEVICE_OVERRIDE:
        return parse_device_override(YOLO_DEVICE_OVERRIDE)

    if torch is None:
        return "cpu"

    try:
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass

    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def find_latest_file(pattern: str) -> Path:
    files = list(OUTPUT_BASE_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return max(files, key=lambda p: p.stat().st_mtime)


def safe_numeric(df: pd.DataFrame, col: str, default: float = 0.0):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index)


# =============================================================================
# [3] Load priority score dataframe
# =============================================================================
def load_priority_scores():
    if PRIORITY_CSV_OVERRIDE:
        priority_csv = Path(PRIORITY_CSV_OVERRIDE).expanduser().resolve()
        if not priority_csv.exists():
            priority_csv = (PROJECT_ROOT / PRIORITY_CSV_OVERRIDE).expanduser().resolve()
        if not priority_csv.exists():
            raise FileNotFoundError(f"AL_PRIORITY_CSV does not exist: {priority_csv}")
    else:
        priority_csv = find_latest_file("pseudo_boxes_*/priority_scores_pseudo.csv")
    df = pd.read_csv(priority_csv)
    df = prepare_priority_dataframe(df)
    return priority_csv, df


def stable_sample_order(df: pd.DataFrame) -> pd.DataFrame:
    """Keep split/random sampling independent of priority CSV row order."""
    sort_cols = [
        c
        for c in ["dataset_type", "image_name", "image_path"]
        if c in df.columns
    ]
    if not sort_cols:
        return df.copy()
    return df.sort_values(sort_cols, kind="mergesort").copy()


def prepare_priority_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_cols = [
        "image_name",
        "dataset_type",
        "image_path",
        "consistency_score",
        "groundedness_reason",
        "score_consistency_only",
        "score_combined_soft_penalty",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"priority_scores_pseudo.csv에 필요한 컬럼이 없습니다: {missing}")

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

    # Groundedness-only soft score 보완 생성
    if "groundedness_effective_soft" not in df.columns:
        if "groundedness_norm" in df.columns:
            df["groundedness_effective_soft"] = df["groundedness_norm"].fillna(0.5)
        else:
            df["groundedness_effective_soft"] = 0.5

    if "missing_box_penalty" not in df.columns:
        df["missing_box_penalty"] = np.where(
            df["groundedness_reason"].astype(str) == "no_pseudo_box",
            0.2,
            0.0,
        )

    df["groundedness_effective_soft"] = pd.to_numeric(
        df["groundedness_effective_soft"],
        errors="coerce",
    ).fillna(0.5)

    df["missing_box_penalty"] = pd.to_numeric(
        df["missing_box_penalty"],
        errors="coerce",
    ).fillna(0.0)

    df["score_groundedness_only_soft"] = (
        1.0 - df["groundedness_effective_soft"]
    ) + df["missing_box_penalty"]

    df["score_combined_no_penalty"] = (
        safe_numeric(df, "uncertainty_consistency", default=0.0)
        + safe_numeric(df, "uncertainty_groundedness_soft", default=0.5)
    )

    df["score_combined_no_groundedness"] = safe_numeric(
        df,
        "uncertainty_consistency",
        default=0.0,
    )

    no_pseudo_indicator = (
        df["groundedness_reason"].astype(str).eq("no_pseudo_box").astype(float)
    )

    df["score_combined_suppress_no_pseudo"] = (
        df["score_combined_no_penalty"]
        - SUPPRESS_NO_PSEUDO_GAMMA * no_pseudo_indicator
    )

    df["score_combined_weighted"] = (
        WEIGHTED_ALPHA * safe_numeric(df, "uncertainty_consistency", default=0.0)
        + WEIGHTED_BETA * safe_numeric(df, "uncertainty_groundedness_soft", default=0.5)
        + WEIGHTED_GAMMA * no_pseudo_indicator
    )

    df["score_combined_rank_calibrated"] = (
        safe_numeric(df, "uncertainty_consistency", default=0.0).rank(method="average", pct=True)
        + safe_numeric(df, "uncertainty_groundedness_soft", default=0.5).rank(method="average", pct=True)
        + WEIGHTED_GAMMA * no_pseudo_indicator
    )

    # class hint 생성
    df["class_hint"] = df.apply(infer_class_hint, axis=1)

    # 중복 제거 후 stable order 고정:
    # penalty sensitivity CSV가 score 기준으로 재정렬되어도 val/seed/random split이 바뀌지 않게 한다.
    df = df.drop_duplicates(subset=["image_name", "dataset_type"])
    df = stable_sample_order(df).reset_index(drop=True)

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
# [5] Selection strategies
# =============================================================================
def class_balanced_select(
    current_pool: pd.DataFrame,
    sample_size: int,
    score_col: str | None,
    ascending: bool,
    seed: int,
    round_idx: int,
) -> pd.DataFrame:
    return group_balanced_select(
        current_pool=current_pool,
        sample_size=sample_size,
        score_col=score_col,
        ascending=ascending,
        seed=seed,
        round_idx=round_idx,
        group_cols=["class_hint"],
    )


def group_balanced_select(
    current_pool: pd.DataFrame,
    sample_size: int,
    score_col: str | None,
    ascending: bool,
    seed: int,
    round_idx: int,
    group_cols: list[str],
) -> pd.DataFrame:
    if len(current_pool) == 0:
        return current_pool.copy()

    current_pool = stable_sample_order(current_pool)
    sample_size = min(sample_size, len(current_pool))

    available_group_cols = [c for c in group_cols if c in current_pool.columns]
    if not available_group_cols:
        if score_col is None:
            return current_pool.sample(n=sample_size, random_state=seed + round_idx * 101)
        return sort_select(current_pool, sample_size, score_col, ascending=ascending)

    group_key = available_group_cols[0] if len(available_group_cols) == 1 else available_group_cols
    groups = list(current_pool.groupby(group_key, dropna=False, sort=True))
    if not groups:
        return current_pool.head(sample_size).copy()

    base_quota = sample_size // len(groups)
    remainder = sample_size % len(groups)
    group_sizes = sorted(
        [(group_name, len(sub)) for group_name, sub in groups],
        key=lambda x: x[1],
        reverse=True,
    )
    extra_groups = {group_name for group_name, _ in group_sizes[:remainder]}

    selected_parts = []
    selected_indices = set()
    for group_name, sub in groups:
        quota = base_quota + (1 if group_name in extra_groups else 0)
        quota = min(quota, len(sub))
        if quota <= 0:
            continue
        if score_col is None:
            picked = stable_sample_order(sub).sample(n=quota, random_state=seed + round_idx * 101)
        else:
            picked = sort_select(sub, quota, score_col, ascending=ascending)
        selected_parts.append(picked)
        selected_indices.update(picked.index.tolist())

    selected = pd.concat(selected_parts) if selected_parts else current_pool.iloc[0:0].copy()

    if len(selected) < sample_size:
        remaining = current_pool.drop(index=list(selected_indices), errors="ignore")
        need = min(sample_size - len(selected), len(remaining))
        if need > 0:
            if score_col is None:
                fill = stable_sample_order(remaining).sample(n=need, random_state=seed + round_idx * 131)
            else:
                fill = sort_select(remaining, need, score_col, ascending=ascending)
            selected = pd.concat([selected, fill])

    return selected.head(sample_size)


def dataset_then_class_balanced_select(
    current_pool: pd.DataFrame,
    sample_size: int,
    score_col: str | None,
    ascending: bool,
    seed: int,
    round_idx: int,
) -> pd.DataFrame:
    """Balance dataset_type first, then class_hint inside each dataset slice."""
    if len(current_pool) == 0:
        return current_pool.copy()
    if "dataset_type" not in current_pool.columns:
        return class_balanced_select(current_pool, sample_size, score_col, ascending, seed, round_idx)

    current_pool = stable_sample_order(current_pool)
    sample_size = min(sample_size, len(current_pool))
    dataset_groups = list(current_pool.groupby("dataset_type", dropna=False, sort=True))
    if not dataset_groups:
        return current_pool.head(sample_size).copy()

    base_quota = sample_size // len(dataset_groups)
    remainder = sample_size % len(dataset_groups)
    dataset_sizes = sorted(
        [(dataset_name, len(sub)) for dataset_name, sub in dataset_groups],
        key=lambda x: x[1],
        reverse=True,
    )
    extra_datasets = {dataset_name for dataset_name, _ in dataset_sizes[:remainder]}

    selected_parts = []
    selected_indices = set()
    for dataset_name, sub in dataset_groups:
        quota = base_quota + (1 if dataset_name in extra_datasets else 0)
        quota = min(quota, len(sub))
        if quota <= 0:
            continue
        picked = class_balanced_select(
            current_pool=sub,
            sample_size=quota,
            score_col=score_col,
            ascending=ascending,
            seed=seed,
            round_idx=round_idx,
        )
        selected_parts.append(picked)
        selected_indices.update(picked.index.tolist())

    selected = pd.concat(selected_parts) if selected_parts else current_pool.iloc[0:0].copy()

    if len(selected) < sample_size:
        remaining = current_pool.drop(index=list(selected_indices), errors="ignore")
        need = min(sample_size - len(selected), len(remaining))
        if need > 0:
            if score_col is None:
                fill = stable_sample_order(remaining).sample(n=need, random_state=seed + round_idx * 137)
            else:
                fill = sort_select(remaining, need, score_col, ascending=ascending)
            selected = pd.concat([selected, fill])

    return selected.head(sample_size)


def sort_select(current_pool: pd.DataFrame, sample_size: int, score_col: str, ascending: bool) -> pd.DataFrame:
    if score_col not in current_pool.columns:
        raise ValueError(
            f"Score column not found for selection: {score_col}\n"
            f"Available columns: {list(current_pool.columns)}"
        )
    stable_cols = [
        c
        for c in ["dataset_type", "image_name", "image_path"]
        if c in current_pool.columns and c != score_col
    ]
    return current_pool.sort_values(
        [score_col] + stable_cols,
        ascending=[ascending] + [True] * len(stable_cols),
        kind="mergesort",
    ).head(sample_size)


def select_samples(
    strategy: str,
    current_pool: pd.DataFrame,
    sample_size: int,
    round_idx: int,
    seed: int,
) -> pd.DataFrame:
    if len(current_pool) == 0:
        return current_pool.copy()

    sample_size = min(sample_size, len(current_pool))
    current_pool = stable_sample_order(current_pool)

    if strategy == "Random":
        return current_pool.sample(
            n=sample_size,
            random_state=seed + round_idx * 101,
        )

    if strategy == "RandomClassBalanced":
        return class_balanced_select(
            current_pool=current_pool,
            sample_size=sample_size,
            score_col=None,
            ascending=False,
            seed=seed,
            round_idx=round_idx,
        )

    if strategy == "RandomClassDatasetBalanced":
        return dataset_then_class_balanced_select(
            current_pool=current_pool,
            sample_size=sample_size,
            score_col=None,
            ascending=False,
            seed=seed,
            round_idx=round_idx,
        )

    if strategy == "ConsistencyOnly":
        return sort_select(current_pool, sample_size, "score_consistency_only", ascending=False)

    if strategy == "ConsistencyOnlyClassBalanced":
        return class_balanced_select(
            current_pool=current_pool,
            sample_size=sample_size,
            score_col="score_consistency_only",
            ascending=False,
            seed=seed,
            round_idx=round_idx,
        )

    if strategy == "ConsistencyOnlyDatasetBalanced":
        return group_balanced_select(
            current_pool=current_pool,
            sample_size=sample_size,
            score_col="score_consistency_only",
            ascending=False,
            seed=seed,
            round_idx=round_idx,
            group_cols=["dataset_type"],
        )

    if strategy == "ConsistencyOnlyClassDatasetBalanced":
        return dataset_then_class_balanced_select(
            current_pool=current_pool,
            sample_size=sample_size,
            score_col="score_consistency_only",
            ascending=False,
            seed=seed,
            round_idx=round_idx,
        )

    if strategy == "GroundednessOnlySoft":
        return sort_select(current_pool, sample_size, "score_groundedness_only_soft", ascending=False)

    if strategy == "CombinedSoftPenalty":
        return sort_select(current_pool, sample_size, "score_combined_soft_penalty", ascending=False)

    if strategy == "LowPrioritySoft":
        return sort_select(current_pool, sample_size, "score_combined_soft_penalty", ascending=True)

    if strategy == "CombinedNoPenalty":
        return sort_select(current_pool, sample_size, "score_combined_no_penalty", ascending=False)

    if strategy == "CombinedNoGroundedness":
        return sort_select(current_pool, sample_size, "score_combined_no_groundedness", ascending=False)

    if strategy == "CombinedWeighted":
        return sort_select(current_pool, sample_size, "score_combined_weighted", ascending=False)

    if strategy == "CombinedRankCalibrated":
        return sort_select(current_pool, sample_size, "score_combined_rank_calibrated", ascending=False)

    if strategy == "CombinedSuppressNoPseudo":
        return sort_select(current_pool, sample_size, "score_combined_suppress_no_pseudo", ascending=False)

    if strategy == "CombinedSoftPenaltyClassBalanced":
        return class_balanced_select(
            current_pool=current_pool,
            sample_size=sample_size,
            score_col="score_combined_soft_penalty",
            ascending=False,
            seed=seed,
            round_idx=round_idx,
        )

    if strategy == "CombinedSuppressNoPseudoClassBalanced":
        return class_balanced_select(
            current_pool=current_pool,
            sample_size=sample_size,
            score_col="score_combined_suppress_no_pseudo",
            ascending=False,
            seed=seed,
            round_idx=round_idx,
        )

    if strategy == "LowPrioritySoftClassBalanced":
        return class_balanced_select(
            current_pool=current_pool,
            sample_size=sample_size,
            score_col="score_combined_soft_penalty",
            ascending=True,
            seed=seed,
            round_idx=round_idx,
        )

    raise ValueError(f"Unknown strategy: {strategy}")


# =============================================================================
# [6] YOLO train/eval
# =============================================================================
def train_and_eval_yolo(
    yaml_path: Path,
    run_output_dir: Path,
    strategy: str,
    round_idx: int,
    seed: int,
) -> dict:
    device = get_device()

    print("\n" + "-" * 80)
    print(f"[YOLO] seed={seed} | strategy={strategy} | round={round_idx}")
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
            name=f"seed{seed}_{strategy}_R{round_idx}",
            exist_ok=True,
            patience=5,
            cache=False,
            verbose=False,
            seed=seed,
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
# [7] Split and metric utilities
# =============================================================================
def make_fixed_val_split(df: pd.DataFrame, seed: int):
    val_parts = []
    df = stable_sample_order(df)

    for dataset_type, sub in df.groupby("dataset_type"):
        n = max(1, int(len(sub) * VAL_RATIO))
        sub = stable_sample_order(sub)
        val_parts.append(sub.sample(n=min(n, len(sub)), random_state=seed + 17))

    val_df = pd.concat(val_parts).drop_duplicates()
    pool_df = df.drop(val_df.index)

    return pool_df.reset_index(drop=True), val_df.reset_index(drop=True)


def compute_aulc(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    mask = ~(np.isnan(x) | np.isnan(y))

    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    area = 0.0
    for i in range(1, len(x)):
        width = x[i] - x[i - 1]
        height = (y[i] + y[i - 1]) / 2.0
        area += width * height

    return float(area)


def add_selection_metadata(
    selected: pd.DataFrame,
    seed: int,
    strategy: str,
    round_idx: int,
    selection_type: str,
) -> pd.DataFrame:
    selected = selected.copy()
    selected.insert(0, "seed", seed)
    selected.insert(1, "strategy", strategy)
    selected.insert(2, "round", round_idx)
    selected.insert(3, "selection_type", selection_type)
    selected.insert(4, "rank_in_selection", range(1, len(selected) + 1))
    return selected


def summarize_selected_batch(selected: pd.DataFrame, seed: int, strategy: str, round_idx: int):
    reason_counts = dict(Counter(selected["groundedness_reason"]))
    dataset_counts = dict(Counter(selected["dataset_type"]))
    class_counts = dict(Counter(selected["class_hint"]))

    print(f"[SELECT] seed={seed} | {strategy} R{round_idx}")
    print(f"  reason_counts : {reason_counts}")
    print(f"  dataset_counts: {dataset_counts}")
    print(f"  class_counts  : {class_counts}")


# =============================================================================
# [8] Plotting
# =============================================================================
def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {path}")


def plot_aggregate_learning_curve(results_df: pd.DataFrame, save_dir: Path, metric: str):
    if results_df[metric].isna().all():
        return

    plt.figure(figsize=(9, 6))

    for strategy, sub in results_df.groupby("strategy"):
        grouped = (
            sub.groupby("labeled_budget")[metric]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("labeled_budget")
        )

        x = grouped["labeled_budget"].values
        mean = grouped["mean"].values
        std = grouped["std"].fillna(0.0).values

        plt.plot(x, mean, marker="o", linewidth=2, label=strategy)
        plt.fill_between(x, mean - std, mean + std, alpha=0.15)

    title_metric = "mAP@50" if metric == "map50" else "mAP@50-95"
    plt.title(f"Aggregate YOLOv8 AL curve: {title_metric}", fontsize=15, fontweight="bold")
    plt.xlabel("Number of labeled training samples")
    plt.ylabel(title_metric)
    plt.grid(alpha=0.3)
    plt.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")

    save_fig(save_dir / f"aggregate_learning_curve_{metric}.png")


def plot_bar_mean_std(summary_df: pd.DataFrame, metric: str, save_dir: Path):
    if summary_df[metric].isna().all():
        return

    agg = (
        summary_df.groupby("strategy")[metric]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    plt.figure(figsize=(10, 5))
    bars = plt.bar(agg["strategy"], agg["mean"], yerr=agg["std"].fillna(0.0), capsize=4)

    plt.title(f"{metric} mean ± std over seeds", fontsize=15, fontweight="bold")
    plt.xlabel("Strategy")
    plt.ylabel(metric)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for bar, mean in zip(bars, agg["mean"]):
        if not np.isnan(mean):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                mean,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    save_fig(save_dir / f"{metric}_mean_std.png")


def plot_selection_distribution(
    selected_df: pd.DataFrame,
    save_dir: Path,
    group_col: str,
    filename: str,
):
    if len(selected_df) == 0:
        return

    acquired = selected_df[selected_df["round"] > 0].copy()

    if len(acquired) == 0:
        return

    pivot = (
        acquired.groupby(["strategy", group_col])
        .size()
        .reset_index(name="count")
    )

    plot_df = pivot.pivot(
        index="strategy",
        columns=group_col,
        values="count",
    ).fillna(0)

    ax = plot_df.plot(
        kind="bar",
        stacked=True,
        figsize=(11, 6),
        edgecolor="white",
    )

    plt.title(f"Selected sample distribution by {group_col}", fontsize=15, fontweight="bold")
    plt.xlabel("Strategy")
    plt.ylabel("Number of selected samples across all seeds/rounds")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc="upper left")

    for container in ax.containers:
        ax.bar_label(container, label_type="center", fontsize=7)

    save_fig(save_dir / filename)


# =============================================================================
# [9] Summaries
# =============================================================================
def make_seed_strategy_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (seed, strategy), sub in results_df.groupby(["seed", "strategy"]):
        sub = sub.sort_values("labeled_budget")

        rows.append({
            "seed": seed,
            "strategy": strategy,
            "final_budget": int(sub["labeled_budget"].iloc[-1]),
            "final_map50": sub["map50"].iloc[-1],
            "final_map5095": sub["map5095"].iloc[-1],
            "best_map50": sub["map50"].max(),
            "best_map5095": sub["map5095"].max(),
            "best_round_map50": int(sub.loc[sub["map50"].idxmax(), "round"]) if not sub["map50"].isna().all() else np.nan,
            "best_round_map5095": int(sub.loc[sub["map5095"].idxmax(), "round"]) if not sub["map5095"].isna().all() else np.nan,
            "aulc_map50": compute_aulc(sub["labeled_budget"], sub["map50"]),
            "aulc_map5095": compute_aulc(sub["labeled_budget"], sub["map5095"]),
        })

    return add_strategy_metadata_columns(pd.DataFrame(rows))


def make_aggregate_strategy_summary(seed_summary_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "final_map50",
        "final_map5095",
        "best_map50",
        "best_map5095",
        "aulc_map50",
        "aulc_map5095",
    ]

    rows = []

    for strategy, sub in seed_summary_df.groupby("strategy"):
        meta = get_strategy_metadata(strategy)
        row = {
            "strategy": strategy,
            "strategy_display_name": meta["display_name"],
            "strategy_family": meta["family"],
            "strategy_role": meta["role"],
            "num_seeds": sub["seed"].nunique(),
        }

        for m in metrics:
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_std"] = sub[m].std()
            row[f"{m}_min"] = sub[m].min()
            row[f"{m}_max"] = sub[m].max()

        rows.append(row)

    out = pd.DataFrame(rows)

    if "final_map50_mean" in out.columns:
        out = out.sort_values("final_map50_mean", ascending=False)

    return out


def write_summary_md(
    save_dir: Path,
    priority_csv: Path,
    config: dict,
    seed_summary_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
):
    lines = []

    lines.append("# YOLO Active Learning Ablation V3 Minimal Summary\n")

    lines.append("## 1. 목적\n")
    lines.append(
        "이 실험은 교수님 피드백을 닫기 위한 최소 검증 실험이다. "
        "Random / high-priority / low-priority 비교와 "
        "Consistency-only / Groundedness-only / Combined ablation을 수행하고, "
        "AULC와 seed 반복을 통해 active learning 효율을 평가한다.\n"
    )

    lines.append("## 2. 입력 파일\n")
    lines.append(f"- priority score CSV: `{priority_csv}`\n")

    lines.append("## 3. 실험 설정\n")
    for k, v in config.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## 4. 전략 의미\n")
    for strategy in config.get("STRATEGIES_TO_RUN", "").split(", "):
        if not strategy:
            continue
        meta = get_strategy_metadata(strategy)
        lines.append(
            f"- `{strategy}`: {meta['display_name']} | {meta['family']} | {meta['role']}"
        )
    lines.append("")

    if len(seed_summary_df) > 0:
        lines.append("## 5. Seed별 결과 요약\n")
        lines.append(seed_summary_df.to_markdown(index=False))
        lines.append("")

    if len(aggregate_df) > 0:
        lines.append("## 6. Strategy별 평균 ± 표준편차 요약\n")
        lines.append(aggregate_df.to_markdown(index=False))
        lines.append("")

    lines.append("## 7. Consistency-centered interpretation\n")
    lines.append("### Core hypothesis\n")
    lines.append(
        "- `ConsistencyOnly` tests whether expert-designed VLM prompt-family "
        "inconsistency is useful as a GT-free acquisition signal."
    )
    lines.append(
        "- This is the main research hypothesis, not merely a baseline."
    )
    lines.append("")
    lines.append("### Auxiliary extension\n")
    lines.append(
        "- `GroundednessOnlySoft`, `CombinedSoftPenalty`, and "
        "`CombinedSuppressNoPseudo` test whether weak pseudo visual evidence "
        "improves or destabilizes the core consistency signal."
    )
    lines.append(
        "- OWL-ViT pseudo boxes are weak visual evidence, not ground-truth boxes."
    )
    lines.append("")
    lines.append("### Failure analysis\n")
    lines.append("- `LowPrioritySoft` is a reverse-direction diagnostic control.")
    lines.append(
        "- `no_pseudo_box` analysis checks whether the auxiliary visual signal "
        "introduces direction/calibration artifacts."
    )
    lines.append(
        "- Current conclusion should be framed as promising but not closed."
    )
    lines.append("")

    lines.append("## 8. 해석 가이드\n")
    lines.append(
        "- CombinedSoftPenalty가 Random보다 final mAP와 AULC에서 높으면, "
        "제안 점수가 detector 학습 효율에 기여할 가능성이 있다."
    )
    lines.append(
        "- CombinedSoftPenalty가 LowPrioritySoft보다 높으면, "
        "priority score의 방향성이 맞다는 근거가 된다."
    )
    lines.append(
        "- CombinedSoftPenalty가 ConsistencyOnly와 GroundednessOnlySoft보다 높으면, "
        "두 축을 결합하는 것이 단일 축보다 유리하다는 근거가 된다."
    )
    lines.append(
        "- final mAP는 높지만 AULC가 낮으면, 후반에는 좋지만 active learning 전체 구간에서는 "
        "초기 안정성이 부족하다는 뜻이다."
    )
    lines.append(
        "- DRY_RUN_ONLY=True인 경우 mAP/AULC는 NaN이며, 이 단계의 목적은 selection 분포와 label 변환 검증이다."
    )

    summary_path = save_dir / "al_ablation_v3_minimal_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVE] {summary_path}")


# =============================================================================
# [10] Main loop
# =============================================================================
def run_one_seed(
    seed: int,
    df_all: pd.DataFrame,
    run_output_dir: Path,
    dataset_output_root: Path,
):
    print("\n" + "=" * 100)
    print(f"[SEED START] seed={seed}")
    print("=" * 100)

    seed_everything(seed)

    seed_run_dir = run_output_dir / f"seed_{seed}"
    seed_dataset_root = dataset_output_root / f"seed_{seed}"

    seed_run_dir.mkdir(parents=True, exist_ok=True)
    seed_dataset_root.mkdir(parents=True, exist_ok=True)

    df_pool_initial, df_val = make_fixed_val_split(df_all, seed=seed)

    df_val.to_csv(seed_run_dir / "fixed_validation_split.csv", index=False, encoding="utf-8-sig")
    df_pool_initial.to_csv(seed_run_dir / "initial_pool_split.csv", index=False, encoding="utf-8-sig")

    initial_seed_df = stable_sample_order(df_pool_initial).sample(
        n=min(INITIAL_SEED_SIZE, len(df_pool_initial)),
        random_state=seed + 999,
    )

    initial_seed_df[
        ["image_name", "dataset_type", "class_hint", "groundedness_reason"]
    ].to_csv(
        seed_run_dir / "shared_initial_seed_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df_val[
        ["image_name", "dataset_type", "class_hint"]
    ].to_csv(
        seed_run_dir / "shared_validation_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_round_results = []
    all_selected_rows = []
    all_build_logs = []

    # -------------------------------------------------------------------------
    # Shared baseline
    # -------------------------------------------------------------------------
    print("\n" + "-" * 100)
    print(f"[SHARED BASELINE] seed={seed}")
    print("-" * 100)

    baseline_dataset_dir = seed_dataset_root / "_SHARED_BASELINE" / "round_0"

    baseline_yaml_path, baseline_build_log_df = build_yolo_dataset(
        train_df=initial_seed_df,
        val_df=df_val,
        dataset_dir=baseline_dataset_dir,
    )

    baseline_build_log_df.insert(0, "seed", seed)
    baseline_build_log_df.insert(1, "strategy", "_SHARED_BASELINE")
    baseline_build_log_df.insert(2, "round", 0)
    all_build_logs.append(baseline_build_log_df)

    baseline_eval_result = train_and_eval_yolo(
        yaml_path=baseline_yaml_path,
        run_output_dir=seed_run_dir,
        strategy="_SHARED_BASELINE",
        round_idx=0,
        seed=seed,
    )

    # -------------------------------------------------------------------------
    # Strategies
    # -------------------------------------------------------------------------
    for strategy in STRATEGIES_TO_RUN:
        print("\n" + "-" * 100)
        print(f"[STRATEGY START] seed={seed} | strategy={strategy}")
        print("-" * 100)

        current_pool = df_pool_initial.drop(initial_seed_df.index).copy()
        labeled_df = initial_seed_df.copy()

        seed_selected_log = add_selection_metadata(
            selected=initial_seed_df,
            seed=seed,
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
                    seed=seed,
                )

                summarize_selected_batch(selected, seed, strategy, round_idx)

                selected_log = add_selection_metadata(
                    selected=selected,
                    seed=seed,
                    strategy=strategy,
                    round_idx=round_idx,
                    selection_type=strategy,
                )
                all_selected_rows.append(selected_log)

                labeled_df = pd.concat([labeled_df, selected])
                current_pool = current_pool.drop(selected.index)

            print(f"\n[ROUND] seed={seed} | strategy={strategy} | round={round_idx}")
            print(f"  labeled samples: {len(labeled_df)}")
            print(f"  current pool   : {len(current_pool)}")
            print(f"  validation     : {len(df_val)}")

            dataset_dir = seed_dataset_root / strategy / f"round_{round_idx}"

            yaml_path, build_log_df = build_yolo_dataset(
                train_df=labeled_df,
                val_df=df_val,
                dataset_dir=dataset_dir,
            )

            build_log_df.insert(0, "seed", seed)
            build_log_df.insert(1, "strategy", strategy)
            build_log_df.insert(2, "round", round_idx)
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
                    run_output_dir=seed_run_dir,
                    strategy=strategy,
                    round_idx=round_idx,
                    seed=seed,
                )

            all_round_results.append({
                "seed": seed,
                "strategy": strategy,
                "round": round_idx,
                "labeled_budget": len(labeled_df),
                "pool_remaining": len(current_pool),
                "val_size": len(df_val),
                "yaml_path": str(yaml_path),
                **eval_result,
            })

            # seed 단위 중간 저장
            pd.DataFrame(all_round_results).to_csv(
                seed_run_dir / "round_results.csv",
                index=False,
                encoding="utf-8-sig",
            )

            if all_selected_rows:
                pd.concat(all_selected_rows, ignore_index=True).to_csv(
                    seed_run_dir / "selected_samples_by_round.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            if all_build_logs:
                pd.concat(all_build_logs, ignore_index=True).to_csv(
                    seed_run_dir / "dataset_build_log.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

    results_df = pd.DataFrame(all_round_results)
    selected_df = pd.concat(all_selected_rows, ignore_index=True) if all_selected_rows else pd.DataFrame()
    build_log_df = pd.concat(all_build_logs, ignore_index=True) if all_build_logs else pd.DataFrame()

    return results_df, selected_df, build_log_df


def main():
    set_korean_font()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    priority_csv, df_all = load_priority_scores()

    run_output_dir = RUNS_ROOT / f"al_ablation_v3_minimal_{timestamp}"
    dataset_output_root = YOLO_DATASETS_ROOT / f"al_ablation_v3_minimal_{timestamp}"

    run_output_dir.mkdir(parents=True, exist_ok=True)
    dataset_output_root.mkdir(parents=True, exist_ok=True)

    config = {
        "VERSION": "v3_windows_cuda",
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "YOLO_MODEL_NAME": YOLO_MODEL_NAME,
        "STRATEGIES_TO_RUN": ", ".join(STRATEGIES_TO_RUN),
        "SEEDS": SEEDS,
        "num_scored_samples": len(df_all),
        "VAL_RATIO": VAL_RATIO,
        "INITIAL_SEED_SIZE": INITIAL_SEED_SIZE,
        "AL_ROUNDS": AL_ROUNDS,
        "QUERY_SIZE": QUERY_SIZE,
        "EPOCHS_PER_ROUND": EPOCHS_PER_ROUND,
        "IMGSZ": IMGSZ,
        "BATCH_SIZE": BATCH_SIZE,
        "WORKERS": WORKERS,
        "YOLO_DEVICE_OVERRIDE": YOLO_DEVICE_OVERRIDE,
        "DEVICE": get_device(),
        "DRY_RUN_ONLY": DRY_RUN_ONLY,
        "STRICT_LABEL_CHECK": STRICT_LABEL_CHECK,
        "PRIORITY_CSV_OVERRIDE": PRIORITY_CSV_OVERRIDE,
        "WEIGHTED_ALPHA": WEIGHTED_ALPHA,
        "WEIGHTED_BETA": WEIGHTED_BETA,
        "WEIGHTED_GAMMA": WEIGHTED_GAMMA,
        "SUPPRESS_NO_PSEUDO_GAMMA": SUPPRESS_NO_PSEUDO_GAMMA,
    }

    with open(run_output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    print("=" * 100)
    print("[YOLO ACTIVE LEARNING ABLATION V3 MINIMAL]")
    print(f"Priority CSV: {priority_csv}")
    print(f"Run output : {run_output_dir}")
    print(f"Dataset dir: {dataset_output_root}")
    print(f"DRY_RUN_ONLY: {DRY_RUN_ONLY}")
    print(f"Strategies: {STRATEGIES_TO_RUN}")
    print(f"Seeds: {SEEDS}")
    print("=" * 100)

    all_results = []
    all_selected = []
    all_build_logs = []

    for seed in SEEDS:
        results_df, selected_df, build_log_df = run_one_seed(
            seed=seed,
            df_all=df_all,
            run_output_dir=run_output_dir,
            dataset_output_root=dataset_output_root,
        )

        all_results.append(results_df)
        all_selected.append(selected_df)
        all_build_logs.append(build_log_df)

    all_results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    all_selected_df = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    all_build_log_df = pd.concat(all_build_logs, ignore_index=True) if all_build_logs else pd.DataFrame()

    # -------------------------------------------------------------------------
    # Save combined raw results
    # -------------------------------------------------------------------------
    all_results_path = run_output_dir / "all_round_results.csv"
    all_selected_path = run_output_dir / "all_selected_samples_by_round.csv"
    all_build_log_path = run_output_dir / "all_dataset_build_log.csv"

    all_results_df.to_csv(all_results_path, index=False, encoding="utf-8-sig")
    all_selected_df.to_csv(all_selected_path, index=False, encoding="utf-8-sig")
    all_build_log_df.to_csv(all_build_log_path, index=False, encoding="utf-8-sig")

    print(f"[SAVE] {all_results_path}")
    print(f"[SAVE] {all_selected_path}")
    print(f"[SAVE] {all_build_log_path}")

    # -------------------------------------------------------------------------
    # Metric summaries
    # -------------------------------------------------------------------------
    seed_summary_df = make_seed_strategy_summary(all_results_df)
    aggregate_df = make_aggregate_strategy_summary(seed_summary_df)

    seed_summary_path = run_output_dir / "seed_strategy_metric_summary.csv"
    aggregate_path = run_output_dir / "aggregate_strategy_metric_summary.csv"

    seed_summary_df.to_csv(seed_summary_path, index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(aggregate_path, index=False, encoding="utf-8-sig")

    print(f"[SAVE] {seed_summary_path}")
    print(f"[SAVE] {aggregate_path}")

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    if not DRY_RUN_ONLY and len(all_results_df) > 0:
        plot_aggregate_learning_curve(all_results_df, run_output_dir, metric="map50")
        plot_aggregate_learning_curve(all_results_df, run_output_dir, metric="map5095")

        for metric in [
            "final_map50",
            "final_map5095",
            "aulc_map50",
            "aulc_map5095",
        ]:
            plot_bar_mean_std(seed_summary_df, metric, run_output_dir)

    if len(all_selected_df) > 0:
        plot_selection_distribution(
            all_selected_df,
            run_output_dir,
            group_col="groundedness_reason",
            filename="selection_reason_distribution.png",
        )
        plot_selection_distribution(
            all_selected_df,
            run_output_dir,
            group_col="dataset_type",
            filename="selection_dataset_distribution.png",
        )
        plot_selection_distribution(
            all_selected_df,
            run_output_dir,
            group_col="class_hint",
            filename="selection_class_hint_distribution.png",
        )

    # -------------------------------------------------------------------------
    # Markdown summary
    # -------------------------------------------------------------------------
    write_summary_md(
        save_dir=run_output_dir,
        priority_csv=priority_csv,
        config=config,
        seed_summary_df=seed_summary_df,
        aggregate_df=aggregate_df,
    )

    print("\n" + "=" * 100)
    print("[완료] YOLO Active Learning Ablation V3 Minimal 종료")
    print(f"Output dir: {run_output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
