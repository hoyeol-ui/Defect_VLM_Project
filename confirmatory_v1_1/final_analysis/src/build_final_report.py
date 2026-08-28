"""Build the canonical Korean final report from frozen confirmatory tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "final_analysis" / "results"
REPORT = ROOT / "final_analysis" / "CONFIRMATORY_FINAL_ANALYSIS_REPORT.md"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def pp(value: str | float) -> str:
    return f"{100.0 * float(value):+.3f} pp"


def dec(value: str | float) -> str:
    return f"{float(value):.6f}"


def primary_case(rows: list[dict[str, str]]) -> tuple[str, str]:
    means = [float(row["mean_dino_minus_random"]) for row in rows]
    lows = [float(row["ci95_low"]) for row in rows]
    highs = [float(row["ci95_high"]) for row in rows]
    if means[0] * means[1] < 0:
        return "Case D", "두 데이터셋의 방향이 달라 selection-to-learning translation은 dataset-dependent했다."
    if all(low > 0 for low in lows):
        return "Case A", "평가된 두 dataset/protocol에서 Frozen DINO의 positive paired downstream difference가 확립됐다."
    if all(high < 0 for high in highs):
        return "Case C", "평가된 두 dataset/protocol에서 Frozen DINO가 Random보다 불리한 paired difference를 보였다."
    return "Case B", "평가된 controlled protocol에서 Frozen DINO의 명확한 downstream advantage가 확립되지 않았다. 이는 동등성 증명이 아니다."


def find(rows: list[dict[str, str]], dataset: str, metric: str) -> dict[str, str]:
    return next(row for row in rows if row["dataset"] == dataset and row["metric"] == metric)


def main() -> None:
    primary = read_csv("primary_summary_2.csv")
    deltas = read_csv("primary_acquisition_deltas_20.csv")
    curves = read_csv("budget_curve_summary.csv")
    secondary = read_csv("secondary_summary.csv")
    classwise = read_csv("classwise_final_summary.csv")
    selection = read_csv("selection_composition_summary.csv")
    sensitivity = read_csv("training_seed_sensitivity.csv")
    case, claim = primary_case(primary)

    lines = [
        "# Random vs Frozen DINO Controlled Confirmatory Final Report",
        "",
        "- Protocol: `confirmatory-random-dino-v1.1`",
        "- Primary unit: acquisition seed, dataset당 `n=10`",
        "- Learner repeats: acquisition 안에서 training seed A/B 평균",
        "- Primary: six-budget locked-final mAP50-95 AULC",
        "- Raw final evaluations: `440/440`",
        "- Missing/interpolated primary point: `0`",
        "- Development substitution: `0`",
        "",
        "## 결론",
        "",
        f"**{case}. {claim}**",
        "",
        "이 결론은 DINO의 보편적 우월성·열등성이나 두 방법의 통계적 동등성을 뜻하지 않는다. 평가 범위는 GC10-DET와 NEU-DET, 약 10–35% image budget, Frozen DINO global representation diversity, YOLOv8n cold-start protocol이다.",
        "",
        "## Primary result",
        "",
        "| Dataset | DINO − Random AULC | 95% acquisition bootstrap CI | wins/losses | exact p | Holm p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            f"| {row['dataset']} | {pp(row['mean_dino_minus_random'])} | "
            f"[{pp(row['ci95_low'])}, {pp(row['ci95_high'])}] | "
            f"{row['wins']}/{row['losses']} | {float(row['exact_sign_flip_p_raw']):.6f} | {float(row['holm_adjusted_p']):.6f} |"
        )
    lines += ["", "### Acquisition-seed raw paired effects", "", "| Dataset | acquisition | seed | Random AULC | DINO AULC | DINO − Random |", "|---|---:|---:|---:|---:|---:|"]
    for row in deltas:
        lines.append(
            f"| {row['dataset']} | a{int(row['acquisition_index']):02d} | {row['acquisition_seed']} | "
            f"{float(row['random_two_training_seed_mean']):.6f} | {float(row['dino_two_training_seed_mean']):.6f} | {pp(row['dino_minus_random'])} |"
        )

    lines += ["", "## Budget curve", "", "| Dataset | Round | Labeled fraction | Random mAP50-95 | DINO mAP50-95 | paired delta (95% CI) |", "|---|---:|---:|---:|---:|---:|"]
    for row in curves:
        lines.append(
            f"| {row['dataset']} | R{row['round']} | {100*float(row['budget_fraction']):.1f}% | "
            f"{100*float(row['random_mean_map50_95']):.2f}% | {100*float(row['dino_mean_map50_95']):.2f}% | "
            f"{pp(row['mean_dino_minus_random'])} [{pp(row['delta_ci95_low'])}, {pp(row['delta_ci95_high'])}] |"
        )

    lines += ["", "## Secondary and subgroup endpoints", "", "Secondary 결과는 primary를 rescue하는 formal claim으로 사용하지 않는다.", "", "| Dataset | Endpoint | DINO − Random | 95% CI | wins/losses |", "|---|---|---:|---:|---:|"]
    for row in secondary:
        lines.append(
            f"| {row['dataset']} | `{row['metric']}` | {pp(row['mean_dino_minus_random'])} | "
            f"[{pp(row['ci95_low'])}, {pp(row['ci95_high'])}] | {row['wins']}/{row['losses']} |"
        )

    lines += ["", "### Class-wise final-budget effects", "", "| Dataset | Class | DINO − Random AP50-95 | 95% CI |", "|---|---|---:|---:|"]
    for row in classwise:
        lines.append(
            f"| {row['dataset']} | {row['class_name']} | {pp(row['mean_dino_minus_random'])} | "
            f"[{pp(row['ci95_low'])}, {pp(row['ci95_high'])}] |"
        )

    lines += ["", "## Selection-stage discovery/composition", "", "| Dataset | Round | Metric | DINO − Random | 95% CI |", "|---|---:|---|---:|---:|"]
    for row in selection:
        if int(row["round"]) == 5:
            unit = " images" if "count" in row["metric"] else ""
            lines.append(
                f"| {row['dataset']} | R5 | `{row['metric']}` | {float(row['mean_dino_minus_random']):+.4f}{unit} | "
                f"[{float(row['ci95_low']):+.4f}, {float(row['ci95_high']):+.4f}] |"
            )

    lines += ["", "## Learner variation", "", "| Dataset | Strategy | mean |A−B| AULC | median |A−B| | max |A−B| |", "|---|---|---:|---:|---:|"]
    for row in sensitivity:
        lines.append(
            f"| {row['dataset']} | {row['strategy']} | {pp(row['mean_absolute_A_minus_B_aulc'])} | "
            f"{pp(row['median_absolute_A_minus_B_aulc'])} | {pp(row['maximum_absolute_A_minus_B_aulc'])} |"
        )

    lines += [
        "",
        "## Historical evidence와의 bounded comparison",
        "",
        "- Historical NEU fixed-set `+0.016236 (5/5)`는 training stability이고, 새 acquisition 10개의 historical estimate `+0.007019`, CI `[-0.005211,+0.019678]`는 acquisition reproducibility 미확립이었다. 이번 controlled result는 어느 쪽도 새 n으로 합치지 않고 방향과 구조만 비교한다.",
        "- Historical GC10 q20은 overall `+0.017378`과 rare macro `−0.019877`이 공존했다. 이번 결과에서도 overall과 frozen rare subgroup을 반드시 함께 해석한다.",
        "- MPDD/VisA discovery, KSDD2 threshold, DeepPCB STOP은 다른 task/estimand이며 confirmatory effect에 pooling하지 않는다.",
        "",
        "## 냉정한 연구 평가 기준",
        "",
        "1. **Selector 논문으로서:** Frozen DINO 자체의 novelty나 보편적 superiority를 주장할 수 없다.",
        "2. **석사학위 연구로서:** result-independent freeze, acquisition/training seed 분리, full learning curve, subgroup, raw registry와 negative-result 보존은 강한 장점이다.",
        "3. **핵심 학술 기여:** candidate signal → discovery → composition → reproducibility → downstream learning을 동일 개념으로 취급하면 안 된다는 산업 결함 AL의 empirical evidence다.",
        "4. **해석 한계:** 두 dataset, YOLOv8n 한 family, global DINO image representation, image-count budget, training seed 2개에 한정된다.",
        "5. **금지 주장:** annotation-time 절감, 산업 현장 일반화, 완전한 untouched test, DINO의 universal superiority/equivalence.",
        "",
        "## Reproducibility artifacts",
        "",
        "- `results/raw_locked_final_metrics_conceptual_480.csv`",
        "- `results/curve_level_aulc_80.csv`",
        "- `results/primary_acquisition_deltas_20.csv`",
        "- `results/selection_composition_raw_240.csv`",
        "- `results/CONFIRMATORY_ANALYSIS_INTEGRITY_AND_SUMMARY.json`",
        "- `figures/confirmatory_primary_results.*`",
        "- `figures/confirmatory_classwise_results.*`",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(str(REPORT))


if __name__ == "__main__":
    main()

