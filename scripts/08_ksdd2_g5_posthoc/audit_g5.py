"""Skeptical, claim-bounded post-hoc audit of a completed frozen G5 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_G5_ROOT = (
    ROOT.parent
    / "Defect_VLM_Project_ksdd2_g5"
    / "runs"
    / "ksdd2_g5_single_unet_v1"
)
DEFAULT_G1_ROOT = (
    ROOT.parent
    / "Defect_VLM_Project_ksdd2_frozen"
    / "runs"
    / "ksdd2_patchcore_prospective_v1"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def seal(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
    )


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def query_ids(g5_root: Path, acquisition_seed: int, strategy: str) -> set[str]:
    rows = read_csv(
        g5_root
        / "frozen_inputs"
        / f"train_acq{acquisition_seed:02d}_{strategy}.csv"
    )
    return {
        row["sample_id"]
        for row in rows
        if row["source_role"] == f"query_{strategy}"
    }


def overlap_summary(g5_root: Path, strategy: str) -> dict[str, Any]:
    sets = [query_ids(g5_root, seed, strategy) for seed in range(10)]
    jaccards = [
        len(left & right) / len(left | right)
        for left, right in combinations(sets, 2)
    ]
    union = set.union(*sets)
    intersection = set.intersection(*sets)
    return {
        "strategy": strategy,
        "pairwise_jaccard_mean": mean(jaccards),
        "pairwise_jaccard_min": float(min(jaccards)),
        "query_union": len(union),
        "selected_in_all_ten": len(intersection),
    }


def annotation_burden(
    g5_root: Path,
    acquisition_labels: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in range(10):
        by_strategy: dict[str, dict[str, float]] = {}
        for strategy in ("patchcore", "random"):
            manifest = read_csv(
                g5_root
                / "frozen_inputs"
                / f"train_acq{seed:02d}_{strategy}.csv"
            )
            query = [
                acquisition_labels[row["sample_id"]]
                for row in manifest
                if row["source_role"] == f"query_{strategy}"
            ]
            full = [acquisition_labels[row["sample_id"]] for row in manifest]
            by_strategy[strategy] = {
                "query_positive_images": sum(
                    int(row["has_defect"]) for row in query
                ),
                "query_defect_pixels": sum(
                    int(row["defect_area_pixels"]) for row in query
                ),
                "query_components": sum(int(row["bbox_count"]) for row in query),
                "training_positive_images": sum(
                    int(row["has_defect"]) for row in full
                ),
                "training_defect_pixels": sum(
                    int(row["defect_area_pixels"]) for row in full
                ),
                "training_components": sum(
                    int(row["bbox_count"]) for row in full
                ),
            }
        result: dict[str, Any] = {"acquisition_seed": seed}
        for metric in by_strategy["patchcore"]:
            patchcore = by_strategy["patchcore"][metric]
            random = by_strategy["random"][metric]
            result[f"patchcore_{metric}"] = patchcore
            result[f"random_{metric}"] = random
            result[f"delta_{metric}"] = patchcore - random
            result[f"ratio_{metric}"] = (
                patchcore / random if random > 0 else float("inf")
            )
        rows.append(result)
    return rows


def training_seed_deltas(
    g5_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for result_path in sorted(
        (g5_root / "learner_runs").glob("*/run_result.json")
    ):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        key = (int(result["acquisition_seed"]), int(result["training_seed"]))
        by_key[key][result["strategy"]] = result
    rows: list[dict[str, Any]] = []
    for (acquisition_seed, training_seed), arms in sorted(by_key.items()):
        if set(arms) != {"patchcore", "random"}:
            raise RuntimeError(f"Incomplete pair: {acquisition_seed}, {training_seed}")
        patchcore = arms["patchcore"]["metrics"]
        random = arms["random"]["metrics"]
        rows.append(
            {
                "acquisition_seed": acquisition_seed,
                "training_seed": training_seed,
                "patchcore_macro_dice": patchcore[
                    "positive_image_macro_hard_dice"
                ],
                "random_macro_dice": random[
                    "positive_image_macro_hard_dice"
                ],
                "delta_macro_dice": patchcore[
                    "positive_image_macro_hard_dice"
                ]
                - random["positive_image_macro_hard_dice"],
                "delta_pixel_ap": patchcore[
                    "all_pixel_average_precision_hist4096"
                ]
                - random["all_pixel_average_precision_hist4096"],
                "patchcore_normal_fpr": patchcore[
                    "normal_image_false_positive_rate"
                ],
                "random_normal_fpr": random[
                    "normal_image_false_positive_rate"
                ],
                "delta_normal_fpr": patchcore[
                    "normal_image_false_positive_rate"
                ]
                - random["normal_image_false_positive_rate"],
            }
        )
    if len(rows) != 30:
        raise RuntimeError(f"Expected 30 paired run rows, got {len(rows)}")
    by_training_seed: dict[str, Any] = {}
    for training_seed in sorted({int(row["training_seed"]) for row in rows}):
        subset = [row for row in rows if int(row["training_seed"]) == training_seed]
        deltas = [float(row["delta_macro_dice"]) for row in subset]
        by_training_seed[str(training_seed)] = {
            "mean_macro_dice_delta": mean(deltas),
            "positive_pair_fraction": mean(
                [float(value > 0) for value in deltas]
            ),
            "minimum_delta": float(min(deltas)),
            "maximum_delta": float(max(deltas)),
        }
    return rows, by_training_seed


def size_stratified(
    g5_root: Path,
    development_labels: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    positive_ids = [
        sample_id
        for sample_id, row in development_labels.items()
        if int(row["has_defect"]) == 1
    ]
    fractions = np.asarray(
        [
            float(development_labels[sample_id]["defect_area_fraction"])
            for sample_id in positive_ids
        ]
    )
    lower, upper = np.quantile(fractions, [1 / 3, 2 / 3])

    def group(sample_id: str) -> str:
        value = float(
            development_labels[sample_id]["defect_area_fraction"]
        )
        return "small_tertile" if value <= lower else "medium_tertile" if value <= upper else "large_tertile"

    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for result_path in sorted(
        (g5_root / "learner_runs").glob("*/run_result.json")
    ):
        strategy = json.loads(
            result_path.read_text(encoding="utf-8")
        )["strategy"]
        per_image = read_csv(result_path.parent / "per_image_metrics.csv")
        for row in per_image:
            if row["has_defect"] == "1":
                values[(strategy, group(row["sample_id"]))].append(
                    float(row["hard_dice"])
                )
    rows: list[dict[str, Any]] = []
    for strategy in ("patchcore", "random"):
        for size_group in ("small_tertile", "medium_tertile", "large_tertile"):
            group_values = values[(strategy, size_group)]
            rows.append(
                {
                    "strategy": strategy,
                    "posthoc_size_group": size_group,
                    "observations": len(group_values),
                    "mean_hard_dice": mean(group_values),
                    "median_hard_dice": float(np.median(group_values)),
                    "zero_dice_fraction": mean(
                        [float(value == 0) for value in group_values]
                    ),
                    "definition": (
                        f"development positive defect-area tertiles; "
                        f"cutpoints={lower:.8f},{upper:.8f}"
                    ),
                }
            )
    return rows


def fmt(value: float) -> str:
    return f"{value:+.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g5-root", type=Path, default=DEFAULT_G5_ROOT)
    parser.add_argument("--g1-root", type=Path, default=DEFAULT_G1_ROOT)
    args = parser.parse_args()
    g5_root = args.g5_root.resolve()
    g1_root = args.g1_root.resolve()
    decision_path = g5_root / "g5_authorization_decision.json"
    if not decision_path.exists():
        raise RuntimeError("Frozen G5 decision does not exist yet")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision["run_conformance"]["completed_runs"] != 60:
        raise RuntimeError("Post-hoc audit requires 60/60 completed runs")
    output = g5_root / "posthoc_skeptical_audit"

    acquisition_labels = {
        row["sample_id"]: row
        for row in read_csv(
            g1_root / "g2_g4" / "posthoc_acquisition_labels.csv"
        )
    }
    development_labels = {
        row["sample_id"]: row
        for row in read_csv(
            g1_root / "g1" / "g1_posthoc_development_labels.csv"
        )
    }
    burden_rows = annotation_burden(g5_root, acquisition_labels)
    seed_rows, seed_summary = training_seed_deltas(g5_root)
    size_rows = size_stratified(g5_root, development_labels)
    overlap = {
        strategy: overlap_summary(g5_root, strategy)
        for strategy in ("patchcore", "random")
    }
    paired = read_csv(g5_root / "paired_seed_metrics.csv")
    positive_burden_delta = np.asarray(
        [float(row["delta_query_positive_images"]) for row in burden_rows]
    )
    dice_delta = np.asarray(
        [
            float(row["delta_positive_image_macro_hard_dice"])
            for row in paired
        ]
    )
    correlation = spearmanr(positive_burden_delta, dice_delta)
    summary = decision["summary"]
    burden_summary = {
        "query_positive_image_delta_mean": mean(
            [float(row["delta_query_positive_images"]) for row in burden_rows]
        ),
        "query_defect_pixel_ratio_mean": mean(
            [float(row["ratio_query_defect_pixels"]) for row in burden_rows]
        ),
        "query_component_ratio_mean": mean(
            [float(row["ratio_query_components"]) for row in burden_rows]
        ),
    }
    audit = {
        "status": "POSTHOC_NOT_GATE_CHANGING",
        "frozen_decision": decision["decision"],
        "official_test_accessed": decision["official_test_accessed"],
        "summary": summary,
        "annotation_burden": burden_summary,
        "training_seed_robustness": seed_summary,
        "query_overlap": overlap,
        "discovery_delta_vs_dice_delta_spearman": {
            "rho": float(correlation.statistic),
            "p_value_descriptive_only": float(correlation.pvalue),
            "n": 10,
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    burden_target = output / "annotation_burden_by_seed.csv"
    seed_target = output / "training_seed_pair_deltas.csv"
    size_target = output / "size_stratified_performance.csv"
    summary_target = output / "audit_summary.json"
    write_csv(burden_target, burden_rows)
    write_csv(seed_target, seed_rows)
    write_csv(size_target, size_rows)
    write_json(summary_target, audit)
    for target in (burden_target, seed_target, size_target, summary_target):
        seal(target)

    decision_name = decision["decision"]
    if decision_name == "LEARNING_TRANSLATION_OBSERVED":
        outcome_text = (
            "동결된 development screen에서는 learning translation이 관찰됐다. "
            "그러나 동일 image budget이 동일 annotation burden을 뜻하지 않으며, "
            "이 결과만으로 비용 효율성이나 PatchCore의 보편적 우위를 주장할 수 없다."
        )
    elif decision_name == "LEARNING_GAIN_WITH_SAFETY_FAIL_STOP":
        outcome_text = (
            "Dice/AP 이득은 관찰됐지만 정상 오탐 안전 기준을 충족하지 못했다. "
            "이는 배포 가능한 성공이 아니라 sensitivity와 false-positive burden의 "
            "trade-off이며, frozen rule에 따라 확장을 중단해야 한다."
        )
    elif decision_name == "TRANSLATION_NOT_OBSERVED_STOP":
        outcome_text = (
            "강한 discovery 이득이 이 동결 learner의 paired learning endpoint로 "
            "번역되지 않았다. 이는 PatchCore나 모든 selector가 무의미하다는 증명이 "
            "아니라, 이 조건에서 자동 전이 가정이 지지되지 않았다는 뜻이다."
        )
    else:
        outcome_text = (
            "실행 또는 식별 조건이 충족되지 않아 학습 전이에 관한 결론을 낼 수 없다."
        )

    report = f"""# KSDD2 G5 냉정·객관적 post-hoc audit

상태: **POSTHOC_NOT_GATE_CHANGING**  
Frozen decision: **{decision_name}**

## 1. 결론

{outcome_text}

## 2. Frozen primary 결과

- PatchCore macro Dice: **{summary['patchcore_macro_dice_mean']:.4f}**
- Random macro Dice: **{summary['random_macro_dice_mean']:.4f}**
- Paired Dice delta: **{fmt(summary['macro_dice_delta_mean'])}**
- 95% CI: **[{fmt(summary['macro_dice_delta_ci_low'])}, {fmt(summary['macro_dice_delta_ci_high'])}]**
- Positive acquisition-seed fraction: **{summary['positive_acquisition_seed_fraction']:.1%}**
- Pixel AP delta: **{fmt(summary['pixel_ap_delta_mean'])}**
- PatchCore normal FPR: **{summary['patchcore_normal_fpr_mean']:.4f}**
- Normal FPR delta: **{fmt(summary['normal_fpr_delta_mean'])}**

## 3. 가장 불편하지만 중요한 사실

### 동일 image budget은 동일 annotation cost가 아니다

- Query positive-image delta: 평균 **{burden_summary['query_positive_image_delta_mean']:+.1f}장**
- Query defect-pixel burden ratio: 평균 **{burden_summary['query_defect_pixel_ratio_mean']:.2f}×**
- Query component burden ratio: 평균 **{burden_summary['query_component_ratio_mean']:.2f}×**

PatchCore arm은 Random보다 훨씬 많은 positive mask와 결함 pixel을 제공받는다.
이는 AL discovery의 의도된 효과지만, mask 작성 시간이 더 길 수 있으므로
`같은 100장 = 같은 annotation 비용`이라고 해석하면 안 된다. 학습 이득이
있더라도 그것은 동일 **image-count budget**의 이득이지 동일 노동시간의
비용 효율성 증거가 아니다.

### 열 acquisition seed는 독립 생산환경 열 개가 아니다

- PatchCore query pairwise Jaccard mean:
  **{overlap['patchcore']['pairwise_jaccard_mean']:.3f}**
- PatchCore query union:
  **{overlap['patchcore']['query_union']}장**
- 열 seed 모두에서 선택:
  **{overlap['patchcore']['selected_in_all_ten']}장**

동일 pool에서 candidate set이 크게 겹치므로 10개 seed는 독립 site나 lot
일반화가 아니다. 현재 CI는 acquisition perturbation에 대한 안정성이지
production generalization의 CI가 아니다.

### Development split은 final evidence가 아니다

Development 466장은 G1에서 이미 사용됐고 G5에서도 평가됐다. 학습이나
checkpoint 선택에는 사용하지 않았지만 완전히 새로운 final test는 아니다.
따라서 PASS도 `screen-stage translation observed`까지만 허용한다.

### FPR은 부차적인 장식이 아니다

정상 FPR 기준은 Dice 이득을 무효화하기 위한 임의 페널티가 아니다. 산업
검사에서 정상 샘플의 반복 오탐은 review workload를 직접 증가시킨다.
Dice PASS와 FPR FAIL이 함께 나오면 알고리즘 성공이 아니라 운영 trade-off다.

## 4. 추가 진단의 위치

- Training-seed pair 결과는 robustness 설명용이며 primary n을 30으로
  늘리지 않는다.
- 결함 크기 tertile 분석은 사후 분석이며 Gate 판정을 변경하지 않는다.
- Discovery delta와 Dice delta의 Spearman은 n=10의 기술 통계일 뿐
  인과관계나 일반화를 주장하지 않는다.
- Threshold sweep, epoch rescue, backbone 교체는 현재 판정을 구조하기
  위한 근거로 사용할 수 없다.

## 5. 결과별 허용되는 다음 행동

- `LEARNING_TRANSLATION_OBSERVED`: 별도 protocol과 지도교수 승인 후에만
  locked official-test confirmation 검토
- `LEARNING_GAIN_WITH_SAFETY_FAIL_STOP`: 확장 중단, sensitivity–review
  burden trade-off 사례로 보고
- `TRANSLATION_NOT_OBSERVED_STOP`: rescue 금지, selection–learning
  translation 미관찰 사례로 보고
- `NOT_IDENTIFIABLE_STOP`: infrastructure 원인을 outcome과 분리해 보고

## 6. 금지되는 주장

- PatchCore의 보편적 우위
- 동일 annotation cost에서의 효율 향상
- 독립 production-pool 일반화
- 실제 annotation 시간 또는 금액 절감
- Official-test 성능
- Workflow의 predictive accuracy
"""
    report_target = output / "SKEPTICAL_OBJECTIVE_AUDIT_KO.md"
    report_target.write_text(report, encoding="utf-8")
    seal(report_target)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"report={report_target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
