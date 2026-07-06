"""
===============================================================================
[File] run_yolo_full_supervised_baseline.py

[Purpose]
Run non-active-learning YOLO baselines for comparison.

This script provides two detector-level baselines:

1. InitialSeedOnly
   - Train YOLO only with the shared initial seed samples.
   - This is the lower-bound before active learning acquisition.

2. FullSupervised
   - Train YOLO with the full available training pool after fixed validation split.
   - This is the upper-bound using all available labeled training samples.

Why this is needed:
    Active Learning results should be interpreted with:
    - Random same-budget baseline
    - Low-priority negative control
    - Full supervised upper-bound
    - Initial seed lower-bound

Input:
    outputs/pseudo_boxes_*/priority_scores_pseudo.csv

Output:
    runs/yolo_full_supervised_baseline/yolo_full_baseline_YYYYMMDD_HHMMSS/
        config.json
        all_baseline_results.csv
        all_dataset_build_log.csv
        seed_baseline_metric_summary.csv
        aggregate_baseline_metric_summary.csv
        baseline_final_map50_mean_std.png
        baseline_final_map5095_mean_std.png
        yolo_full_supervised_baseline_summary.md

Important:
    GT XML labels are used only for YOLO training/evaluation labels.
    They are not used for active learning acquisition scoring.
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
        "ultralytics가 설치되어 있지 않습니다. 먼저 설치하세요:\n"
        "pip install ultralytics"
    ) from e


# =============================================================================
# [0] Project settings
# =============================================================================
PROJECT_ROOT = Path("/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project")
OUTPUT_BASE_DIR = PROJECT_ROOT / "outputs"
DATA_ROOT = PROJECT_ROOT / "data"

RUNS_ROOT = PROJECT_ROOT / "runs" / "yolo_full_supervised_baseline"
YOLO_DATASETS_ROOT = PROJECT_ROOT / "datasets" / "yolo_full_supervised_baseline"

# 처음에는 True로 실행해서 라벨 변환/데이터셋 생성만 확인
DRY_RUN_ONLY = False
STRICT_LABEL_CHECK = True

SEEDS = [42, 43, 44]

VAL_RATIO = 0.20
INITIAL_SEED_SIZE = 30

EPOCHS = 30
IMGSZ = 640
BATCH_SIZE = 4
WORKERS = 0

YOLO_MODEL_NAME = "yolov8n.pt"

BASELINES_TO_RUN = [
    "InitialSeedOnly",
    "FullSupervised",
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
# [2] Basic utilities
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


def clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def find_latest_file(pattern: str) -> Path:
    files = list(OUTPUT_BASE_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return max(files, key=lambda p: p.stat().st_mtime)


# =============================================================================
# [3] Load scored sample manifest
# =============================================================================
def load_priority_scores():
    priority_csv = find_latest_file("pseudo_boxes_*/priority_scores_pseudo.csv")
    df = pd.read_csv(priority_csv)
    df = prepare_dataframe(df)
    return priority_csv, df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_cols = [
        "image_name",
        "dataset_type",
        "image_path",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"priority_scores_pseudo.csv에 필요한 컬럼이 없습니다: {missing}")

    df["dataset_type"] = df["dataset_type"].fillna("unknown")
    df["image_path"] = df["image_path"].astype(str)
    df["class_hint"] = df.apply(infer_class_hint, axis=1)

    if "groundedness_reason" not in df.columns:
        df["groundedness_reason"] = "unknown"
    else:
        df["groundedness_reason"] = df["groundedness_reason"].fillna("unknown")

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
# [5] Split
# =============================================================================
def make_fixed_val_split(df: pd.DataFrame, seed: int):
    val_parts = []

    for dataset_type, sub in df.groupby("dataset_type"):
        n = max(1, int(len(sub) * VAL_RATIO))
        val_parts.append(sub.sample(n=min(n, len(sub)), random_state=seed + 17))

    val_df = pd.concat(val_parts).drop_duplicates()
    train_pool_df = df.drop(val_df.index)

    return train_pool_df.reset_index(drop=True), val_df.reset_index(drop=True)


def make_initial_seed_df(train_pool_df: pd.DataFrame, seed: int):
    return train_pool_df.sample(
        n=min(INITIAL_SEED_SIZE, len(train_pool_df)),
        random_state=seed + 999,
    ).copy()


# =============================================================================
# [6] YOLO train/eval
# =============================================================================
def train_and_eval_yolo(
    yaml_path: Path,
    run_output_dir: Path,
    baseline_name: str,
    seed: int,
) -> dict:
    device = get_device()

    print("\n" + "-" * 80)
    print(f"[YOLO BASELINE] seed={seed} | baseline={baseline_name}")
    print(f"[YOLO BASELINE] yaml={yaml_path}")
    print(f"[YOLO BASELINE] device={device}")
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
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH_SIZE,
            workers=WORKERS,
            device=device,
            project=str(run_output_dir / "yolo_train_runs"),
            name=f"seed{seed}_{baseline_name}",
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

        print(
            f"[YOLO BASELINE] 완료 | "
            f"mAP50={map50:.4f}, mAP50-95={map5095:.4f}, "
            f"P={precision:.4f}, R={recall:.4f}"
        )

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
# [7] Plot / summary
# =============================================================================
def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {path}")


def plot_baseline_mean_std(results_df: pd.DataFrame, metric: str, save_dir: Path):
    if results_df[metric].isna().all():
        return

    agg = (
        results_df.groupby("baseline")[metric]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        agg["baseline"],
        agg["mean"],
        yerr=agg["std"].fillna(0.0),
        capsize=4,
    )

    plt.title(f"YOLO baseline {metric} mean ± std", fontsize=15, fontweight="bold")
    plt.xlabel("Baseline")
    plt.ylabel(metric)
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

    save_fig(save_dir / f"baseline_{metric}_mean_std.png")


def make_seed_baseline_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (seed, baseline), sub in results_df.groupby(["seed", "baseline"]):
        row = sub.iloc[-1].to_dict()

        rows.append({
            "seed": seed,
            "baseline": baseline,
            "train_size": int(row["train_size"]),
            "val_size": int(row["val_size"]),
            "map50": row["map50"],
            "map5095": row["map5095"],
            "precision": row["precision"],
            "recall": row["recall"],
            "train_status": row["train_status"],
            "train_run_dir": row["train_run_dir"],
            "error": row["error"],
        })

    return pd.DataFrame(rows)


def make_aggregate_baseline_summary(seed_summary_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["map50", "map5095", "precision", "recall"]

    rows = []

    for baseline, sub in seed_summary_df.groupby("baseline"):
        row = {
            "baseline": baseline,
            "num_seeds": sub["seed"].nunique(),
            "train_size_mean": sub["train_size"].mean(),
            "train_size_min": sub["train_size"].min(),
            "train_size_max": sub["train_size"].max(),
            "val_size_mean": sub["val_size"].mean(),
        }

        for m in metrics:
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_std"] = sub[m].std()
            row[f"{m}_min"] = sub[m].min()
            row[f"{m}_max"] = sub[m].max()

        rows.append(row)

    out = pd.DataFrame(rows)

    if "map50_mean" in out.columns:
        out = out.sort_values("map50_mean", ascending=False)

    return out

def dataframe_to_markdown_safe(df: pd.DataFrame) -> str:
    """
    Convert dataframe to markdown table without requiring tabulate.
    This function safely converts every value to string.
    """
    if df is None or len(df) == 0:
        return "_No data available._"

    df = df.copy()

    # 보기 좋게 float 값 반올림
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].round(6)

    header = "| " + " | ".join([str(c) for c in df.columns]) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"

    rows = []
    for _, row in df.iterrows():
        values = []
        for v in row.values.tolist():
            if pd.isna(v):
                values.append("")
            else:
                values.append(str(v).replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator] + rows)


def write_summary_md(
    save_dir: Path,
    priority_csv: Path,
    config: dict,
    seed_summary_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
):
    lines = []

    lines.append("# YOLO Full Supervised Baseline Summary\n")

    lines.append("## 1. 목적\n")
    lines.append(
        "이 실험은 Active Learning을 적용하지 않은 순수 YOLO baseline을 만들기 위한 것이다. "
        "InitialSeedOnly는 라벨 30장만 사용한 lower-bound이고, "
        "FullSupervised는 validation set을 제외한 train pool 전체를 사용한 upper-bound이다.\n"
    )

    lines.append("## 2. 입력 파일\n")
    lines.append(f"- priority score CSV: `{priority_csv}`\n")

    lines.append("## 3. 실험 설정\n")
    for k, v in config.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## 4. Baseline 의미\n")
    lines.append("- InitialSeedOnly: Active Learning 시작 전 초기 라벨 데이터만 사용한 YOLO 성능")
    lines.append("- FullSupervised: 사용 가능한 train pool 전체 라벨을 사용한 YOLO upper-bound")
    lines.append("")

    if len(seed_summary_df) > 0:
        lines.append("## 5. Seed별 결과\n")
        lines.append(dataframe_to_markdown_safe(seed_summary_df))
        lines.append("")

    if len(aggregate_df) > 0:
        lines.append("## 6. 평균 ± 표준편차 결과\n")
        lines.append(dataframe_to_markdown_safe(aggregate_df))
        lines.append("")

    lines.append("## 7. 해석 가이드\n")
    lines.append(
        "- Active Learning 전략은 같은 labeled budget에서 Random/LowPriority와 비교해야 한다."
    )
    lines.append(
        "- FullSupervised는 이겨야 하는 대상이 아니라, 모든 사용 가능 라벨을 사용했을 때의 upper-bound이다."
    )
    lines.append(
        "- AL 60장 성능이 FullSupervised 성능에 얼마나 접근하는지 upper-bound 도달률로 해석할 수 있다."
    )
    lines.append(
        "- DRY_RUN_ONLY=True인 경우 mAP는 NaN이며, 데이터셋 생성과 라벨 변환 검증만 수행한다."
    )

    summary_path = save_dir / "yolo_full_supervised_baseline_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVE] {summary_path}")


# =============================================================================
# [8] Main
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

    train_pool_df, val_df = make_fixed_val_split(df_all, seed=seed)
    initial_seed_df = make_initial_seed_df(train_pool_df, seed=seed)

    train_pool_df.to_csv(seed_run_dir / "full_train_pool_split.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(seed_run_dir / "fixed_validation_split.csv", index=False, encoding="utf-8-sig")
    initial_seed_df.to_csv(seed_run_dir / "initial_seed_split.csv", index=False, encoding="utf-8-sig")

    results = []
    build_logs = []

    for baseline in BASELINES_TO_RUN:
        print("\n" + "-" * 100)
        print(f"[BASELINE START] seed={seed} | baseline={baseline}")
        print("-" * 100)

        if baseline == "InitialSeedOnly":
            train_df = initial_seed_df.copy()

        elif baseline == "FullSupervised":
            train_df = train_pool_df.copy()

        else:
            raise ValueError(f"Unknown baseline: {baseline}")

        dataset_dir = seed_dataset_root / baseline

        yaml_path, build_log_df = build_yolo_dataset(
            train_df=train_df,
            val_df=val_df,
            dataset_dir=dataset_dir,
        )

        build_log_df.insert(0, "seed", seed)
        build_log_df.insert(1, "baseline", baseline)
        build_logs.append(build_log_df)

        eval_result = train_and_eval_yolo(
            yaml_path=yaml_path,
            run_output_dir=seed_run_dir,
            baseline_name=baseline,
            seed=seed,
        )

        result_row = {
            "seed": seed,
            "baseline": baseline,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "yaml_path": str(yaml_path),
            **eval_result,
        }

        results.append(result_row)

        pd.DataFrame(results).to_csv(
            seed_run_dir / "baseline_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

        pd.concat(build_logs, ignore_index=True).to_csv(
            seed_run_dir / "dataset_build_log.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return pd.DataFrame(results), pd.concat(build_logs, ignore_index=True)


def main():
    set_korean_font()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    priority_csv, df_all = load_priority_scores()

    run_output_dir = RUNS_ROOT / f"yolo_full_baseline_{timestamp}"
    dataset_output_root = YOLO_DATASETS_ROOT / f"yolo_full_baseline_{timestamp}"

    run_output_dir.mkdir(parents=True, exist_ok=True)
    dataset_output_root.mkdir(parents=True, exist_ok=True)

    config = {
        "VERSION": "full_supervised_baseline",
        "YOLO_MODEL_NAME": YOLO_MODEL_NAME,
        "BASELINES_TO_RUN": ", ".join(BASELINES_TO_RUN),
        "SEEDS": SEEDS,
        "num_scored_samples": len(df_all),
        "VAL_RATIO": VAL_RATIO,
        "INITIAL_SEED_SIZE": INITIAL_SEED_SIZE,
        "EPOCHS": EPOCHS,
        "IMGSZ": IMGSZ,
        "BATCH_SIZE": BATCH_SIZE,
        "DEVICE": get_device(),
        "DRY_RUN_ONLY": DRY_RUN_ONLY,
        "STRICT_LABEL_CHECK": STRICT_LABEL_CHECK,
    }

    with open(run_output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    print("=" * 100)
    print("[YOLO FULL SUPERVISED BASELINE]")
    print(f"Priority CSV: {priority_csv}")
    print(f"Run output : {run_output_dir}")
    print(f"Dataset dir: {dataset_output_root}")
    print(f"DRY_RUN_ONLY: {DRY_RUN_ONLY}")
    print(f"Baselines: {BASELINES_TO_RUN}")
    print(f"Seeds: {SEEDS}")
    print("=" * 100)

    all_results = []
    all_build_logs = []

    for seed in SEEDS:
        results_df, build_log_df = run_one_seed(
            seed=seed,
            df_all=df_all,
            run_output_dir=run_output_dir,
            dataset_output_root=dataset_output_root,
        )

        all_results.append(results_df)
        all_build_logs.append(build_log_df)

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_build_log_df = pd.concat(all_build_logs, ignore_index=True)

    all_results_path = run_output_dir / "all_baseline_results.csv"
    all_build_log_path = run_output_dir / "all_dataset_build_log.csv"

    all_results_df.to_csv(all_results_path, index=False, encoding="utf-8-sig")
    all_build_log_df.to_csv(all_build_log_path, index=False, encoding="utf-8-sig")

    print(f"[SAVE] {all_results_path}")
    print(f"[SAVE] {all_build_log_path}")

    seed_summary_df = make_seed_baseline_summary(all_results_df)
    aggregate_df = make_aggregate_baseline_summary(seed_summary_df)

    seed_summary_path = run_output_dir / "seed_baseline_metric_summary.csv"
    aggregate_path = run_output_dir / "aggregate_baseline_metric_summary.csv"

    seed_summary_df.to_csv(seed_summary_path, index=False, encoding="utf-8-sig")
    aggregate_df.to_csv(aggregate_path, index=False, encoding="utf-8-sig")

    print(f"[SAVE] {seed_summary_path}")
    print(f"[SAVE] {aggregate_path}")

    if not DRY_RUN_ONLY:
        plot_baseline_mean_std(seed_summary_df, "map50", run_output_dir)
        plot_baseline_mean_std(seed_summary_df, "map5095", run_output_dir)
        plot_baseline_mean_std(seed_summary_df, "precision", run_output_dir)
        plot_baseline_mean_std(seed_summary_df, "recall", run_output_dir)

    write_summary_md(
        save_dir=run_output_dir,
        priority_csv=priority_csv,
        config=config,
        seed_summary_df=seed_summary_df,
        aggregate_df=aggregate_df,
    )

    print("\n" + "=" * 100)
    print("[완료] YOLO Full Supervised Baseline 종료")
    print(f"Output dir: {run_output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()