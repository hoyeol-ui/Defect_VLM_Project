"""
===============================================================================
[File] run_al_yolo_direction_check.py

[Purpose]
Run a focused direction-check experiment for the VLM-based Active Learning study.

This script reuses the already validated V3 minimal pipeline:
    scripts/02_active_learning/run_al_yolo_ablation_v3_minimal.py

Instead of running all ablation strategies again, this focused script only runs:

    1. Random
    2. CombinedSoftPenalty
    3. LowPrioritySoft

with additional seeds:

    SEEDS = [45, 46]

Why this experiment is needed:
    The previous V3 minimal experiment showed:

    - CombinedSoftPenalty had the best AULC.
    - LowPrioritySoft had the best final mAP.
    - Random was lower than the proposed/high-priority strategies.

Therefore, the remaining question is not the full ablation again.
The remaining question is:

    "Is the acquisition score direction stable?"

Specifically:
    - Does CombinedSoftPenalty remain better than Random?
    - Does CombinedSoftPenalty keep strong AULC?
    - Does LowPrioritySoft still win final mAP?
    - Is the final-mAP direction issue a seed artifact or a repeated pattern?

Important:
    This script DOES NOT define a new method.
    It is a focused robustness/direction-check experiment.

Recommended usage:
    1. Keep DRY_RUN_ONLY = True first.
    2. Run this script once and check dataset generation.
    3. If no error occurs, set DRY_RUN_ONLY = False.
    4. Run actual YOLO training.

Expected training count:
    2 seeds × 3 strategies × 4 rounds
    Round 0 shared baseline is reused within each seed.

===============================================================================
"""

from pathlib import Path
import importlib.util
import sys


# =============================================================================
# [0] Project path
# =============================================================================

PROJECT_ROOT = Path("/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project")

BASE_SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "02_active_learning"
    / "run_al_yolo_ablation_v3_minimal.py"
)


# =============================================================================
# [1] Direction-check settings
# =============================================================================

# 처음에는 True로 실행해서 데이터셋 생성/라벨 변환/전략 선택만 확인
# 정상 통과 후 False로 바꿔서 실제 YOLO 학습 실행
DRY_RUN_ONLY = False

# 방향성 확인용 추가 seed
SEEDS = [45, 46]

# 핵심 비교 전략만 남김
STRATEGIES_TO_RUN = [
    "Random",
    "CombinedSoftPenalty",
    "LowPrioritySoft",
]

# 기존 V3 minimal과 동일하게 유지
INITIAL_SEED_SIZE = 30
AL_ROUNDS = 3
QUERY_SIZE = 10

EPOCHS_PER_ROUND = 30
IMGSZ = 640
BATCH_SIZE = 4
WORKERS = 0

YOLO_MODEL_NAME = "yolov8n.pt"
STRICT_LABEL_CHECK = True

# 출력 폴더를 V3 minimal과 분리
RUNS_ROOT = PROJECT_ROOT / "runs" / "active_learning_direction_check"
YOLO_DATASETS_ROOT = PROJECT_ROOT / "datasets" / "al_yolo_direction_check"


# =============================================================================
# [2] Load base V3 minimal module
# =============================================================================

def load_base_module():
    """
    Dynamically load the existing V3 minimal script as a Python module.

    This lets us reuse all validated functions without copying the whole code.
    """
    if not BASE_SCRIPT_PATH.exists():
        raise FileNotFoundError(
            f"Base V3 minimal script not found:\n{BASE_SCRIPT_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "run_al_yolo_ablation_v3_minimal",
        str(BASE_SCRIPT_PATH),
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from: {BASE_SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)

    # Register module so relative/internal references are safer
    sys.modules["run_al_yolo_ablation_v3_minimal"] = module

    spec.loader.exec_module(module)

    return module


# =============================================================================
# [3] Override settings
# =============================================================================

def override_settings(module):
    """
    Override global config values in the imported V3 minimal module.
    """

    overrides = {
        "DRY_RUN_ONLY": DRY_RUN_ONLY,
        "STRICT_LABEL_CHECK": STRICT_LABEL_CHECK,

        "SEEDS": SEEDS,
        "STRATEGIES_TO_RUN": STRATEGIES_TO_RUN,

        "INITIAL_SEED_SIZE": INITIAL_SEED_SIZE,
        "AL_ROUNDS": AL_ROUNDS,
        "QUERY_SIZE": QUERY_SIZE,

        "EPOCHS_PER_ROUND": EPOCHS_PER_ROUND,
        "IMGSZ": IMGSZ,
        "BATCH_SIZE": BATCH_SIZE,
        "WORKERS": WORKERS,

        "YOLO_MODEL_NAME": YOLO_MODEL_NAME,

        "RUNS_ROOT": RUNS_ROOT,
        "YOLO_DATASETS_ROOT": YOLO_DATASETS_ROOT,
    }

    print("=" * 100)
    print("[DIRECTION CHECK CONFIG OVERRIDE]")
    print(f"Base script: {BASE_SCRIPT_PATH}")
    print("=" * 100)

    for key, value in overrides.items():
        if hasattr(module, key):
            setattr(module, key, value)
            print(f"[OVERRIDE] {key} = {value}")
        else:
            print(f"[WARN] Base module has no attribute: {key}")

    print("=" * 100)


# =============================================================================
# [4] Main
# =============================================================================

def main():
    module = load_base_module()
    override_settings(module)

    if not hasattr(module, "main"):
        raise AttributeError(
            "The base V3 minimal script does not have a main() function."
        )

    print("\n" + "=" * 100)
    print("[START] Active Learning Direction Check")
    print("=" * 100)
    print(f"DRY_RUN_ONLY    : {DRY_RUN_ONLY}")
    print(f"SEEDS           : {SEEDS}")
    print(f"STRATEGIES      : {STRATEGIES_TO_RUN}")
    print(f"RUNS_ROOT       : {RUNS_ROOT}")
    print(f"DATASETS_ROOT   : {YOLO_DATASETS_ROOT}")
    print("=" * 100 + "\n")

    module.main()

    print("\n" + "=" * 100)
    print("[DONE] Active Learning Direction Check")
    print("=" * 100)


if __name__ == "__main__":
    main()