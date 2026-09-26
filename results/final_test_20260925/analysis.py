"""Frozen five-seed test analysis. Runs only after 162/162 completeness PASS."""
from __future__ import annotations

import csv
import datetime as dt
import itertools
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from input_audit import METADATA, OUT, PHASE1B, RUNS, check, read_csv, read_json, sha

import numpy as np

METHODS = ("Random", "DINO_F518", "Entropy")
METHOD_LABELS = {"Random": "Random", "DINO_F518": "DINOv2-FF", "Entropy": "Entropy"}
SEEDS = tuple(range(2026091901, 2026091906))
CONTRASTS = (("DINO_F518", "Random"), ("DINO_F518", "Entropy"), ("Entropy", "Random"))
BOOTSTRAP = 100_000
BOOTSTRAP_SEEDS = {"GC10-DET": 2026091912, "NEU-DET": 2026091913}
ALPHA = 0.05


def write_json_x(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def write_csv_x(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)


def aulc(values: list[float], budgets: list[int]) -> float:
    check(len(values) == len(budgets) == 6 and len(set(budgets)) == 6, "Incomplete six-point nAULC")
    return float(np.trapezoid(np.asarray(values, dtype=np.float64),
                              np.asarray(budgets, dtype=np.float64)) / (budgets[-1] - budgets[0]))


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values)


def paired_stats(ds: str, label: str, a: str, b: str, by_seed: dict, metric: str):
    diffs = np.asarray([by_seed[(ds, seed, a)] - by_seed[(ds, seed, b)] for seed in SEEDS], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEEDS[ds])
    samples = diffs[rng.integers(0, len(SEEDS), size=(BOOTSTRAP, len(SEEDS)))].mean(axis=1)
    ci = np.quantile(samples, [0.025, 0.975])
    observed = float(diffs.mean())
    flipped = [float(np.mean(diffs * np.asarray(signs))) for signs in itertools.product((-1, 1), repeat=5)]
    p = sum(abs(v) >= abs(observed) - 1e-15 for v in flipped) / 32
    return {"dataset": ds, "metric": metric, "comparison": label, "method_a": a, "method_b": b,
            "mean_a": statistics.mean(by_seed[(ds, seed, a)] for seed in SEEDS),
            "sd_a": statistics.stdev(by_seed[(ds, seed, a)] for seed in SEEDS),
            "mean_b": statistics.mean(by_seed[(ds, seed, b)] for seed in SEEDS),
            "sd_b": statistics.stdev(by_seed[(ds, seed, b)] for seed in SEEDS),
            "paired_mean_difference": observed, "paired_mean_difference_pp": observed * 100,
            "paired_sd_difference": float(np.std(diffs, ddof=1)),
            "positive_seeds": int((diffs > 0).sum()), "zero_seeds": int((diffs == 0).sum()),
            "n_seeds": 5, "bootstrap_draws": BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEEDS[ds],
            "bootstrap_ci_lower": float(ci[0]), "bootstrap_ci_upper": float(ci[1]),
            "bootstrap_ci_lower_pp": float(ci[0] * 100), "bootstrap_ci_upper_pp": float(ci[1] * 100),
            "exact_two_sided_sign_flip_p": p, "exact_enumerations": 32,
            "paired_differences": [float(v) for v in diffs]}


def holm_primary(rows: list[dict]):
    primary = [r for r in rows if r["metric"] == "mAP50_95_nAULC" and r["comparison"] in
               ("DINOv2-FF - Random", "DINOv2-FF - Entropy")]
    check(len(primary) == 4, "Four primary comparisons required")
    previous = 0.0
    for i, r in enumerate(sorted(primary, key=lambda x: (x["exact_two_sided_sign_flip_p"], x["dataset"], x["comparison"]))):
        previous = min(1.0, max(previous, (4 - i) * r["exact_two_sided_sign_flip_p"]))
        r["holm_adjusted_p"] = previous
        r["holm_reject_0_05"] = previous <= ALPHA
    for r in rows:
        if "holm_adjusted_p" not in r:
            r["holm_adjusted_p"] = None
            r["holm_reject_0_05"] = None
    return primary


def figure_outputs(budget_summary: list[dict], class_diff: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    for ds in ("GC10-DET", "NEU-DET"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in METHODS:
            rows = sorted((r for r in budget_summary if r["dataset"] == ds and r["method"] == method),
                          key=lambda r: r["round"])
            x = [r["labeled_images"] for r in rows]
            y = [100 * r["mean_mAP50_95"] for r in rows]
            sd = [100 * r["sample_sd_mAP50_95"] for r in rows]
            ax.plot(x, y, marker="o", label=METHOD_LABELS[method])
            ax.fill_between(x, np.asarray(y) - sd, np.asarray(y) + sd, alpha=.15)
        ax.set(title=f"{ds} TEST learning curve (mean ± sample SD, n=5)",
               xlabel="Labeled training images", ylabel="Held-out test mAP50–95 (%)")
        ax.grid(alpha=.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f"{ds}_TEST_learning_curve.png", dpi=220)
        plt.close(fig)

        rows = [r for r in class_diff if r["dataset"] == ds]
        fig, ax = plt.subplots(figsize=(max(8, .8 * len(rows)), 5))
        ax.bar([r["class_name"] for r in rows], [r["dino_minus_random_mean_ap_pp"] for r in rows])
        ax.axhline(0, color="black", linewidth=.8)
        ax.tick_params(axis="x", labelrotation=40)
        ax.set(title=f"{ds} TEST R5 class AP: DINOv2-FF − Random",
               ylabel="Paired mean AP50–95 difference (percentage points)")
        fig.tight_layout()
        fig.savefig(figures / f"{ds}_TEST_R5_class_AP_difference.png", dpi=220)
        plt.close(fig)


def markdown_report(config, results, budget_summary, seed_naulc, comparisons, class_diff,
                    composition, worst_naulc, r5_full, dev_test, primary):
    repo = OUT.parent / "Defect_VLM_Project_final_main_results_20260922"
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    import torch, torchvision, ultralytics
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    lines = ["# Final held-out TEST evaluation — 2026-09-25", "",
             "## 1. 최종평가 실행 개요", "",
             f"- 실행 완료 시각 UTC: {now}", f"- 저장소: `{repo}`; branch `{branch}`; commit `{commit}`",
             f"- GPU: {torch.cuda.get_device_name(0)}; Python {sys.version.split()[0]}; PyTorch {torch.__version__}; torchvision {torchvision.__version__}; Ultralytics {ultralytics.__version__}",
             "- Scientific checkpoint SHA 검증: 162/162. `TECHNICAL_NEU-DET` 제외.",
             "- Test 이미지: GC10-DET 344, NEU-DET 270.",
             "- YOLOv8n; imgsz 640; batch 8; conf 0.001; NMS IoU 0.70; max_det 300; 각 `best.pt` 1회 평가.",
             "", "## 2. 사전 검증", "",
             "- 고정 split의 train/dev/test ID, RGB SHA, 파일 SHA, component 중복 0.",
             "- 원본 test 이미지·XML SHA, RGB identity, bbox, class mapping, native label parser: PASS.",
             "- GC10 DINO R5 dev dry-run: 저장된 전체 및 클래스별 AP와 최대 절대 차이 0.0; PASS.",
             "- Test completeness: 162/162 SUCCESS, missing/duplicate/unexpected/technical 0.", ""]
    for ds, heading in (("GC10-DET", "3"), ("NEU-DET", "4")):
        lines += [f"## {heading}. {ds} test 결과", "",
                  "### R0–R5 mAP50–95 (%)", "",
                  "|Round|Labeled|Random|DINOv2-FF|Entropy|", "|---:|---:|---:|---:|---:|"]
        for rd in range(6):
            rows = {r["method"]: r for r in budget_summary if r["dataset"] == ds and r["round"] == rd}
            n = rows["Random"]["labeled_images"]
            lines.append(f"|R{rd}|{n}|{100*rows['Random']['mean_mAP50_95']:.3f}|{100*rows['DINO_F518']['mean_mAP50_95']:.3f}|{100*rows['Entropy']['mean_mAP50_95']:.3f}|")
        lines += ["", "### nAULC 및 비교", "", "|Method|Mean nAULC (%)|Sample SD (%p)|",
                  "|---|---:|---:|"]
        for method in METHODS:
            vals = [r["nAULC"] for r in seed_naulc if r["dataset"] == ds and r["method"] == method]
            lines.append(f"|{METHOD_LABELS[method]}|{100*statistics.mean(vals):.3f}|{100*statistics.stdev(vals):.3f}|")
        lines += ["", "|Comparison|Paired mean (%p)|95% bootstrap CI (%p)|Positive seeds|Exact p|Holm p|",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in comparisons:
            if r["dataset"] != ds or r["metric"] != "mAP50_95_nAULC":
                continue
            hp = "—" if r["holm_adjusted_p"] is None else f"{r['holm_adjusted_p']:.5f}"
            lines.append(f"|{r['comparison']}|{r['paired_mean_difference_pp']:+.3f}|[{r['bootstrap_ci_lower_pp']:+.3f}, {r['bootstrap_ci_upper_pp']:+.3f}]|{r['positive_seeds']}/5|{r['exact_two_sided_sign_flip_p']:.5f}|{hp}|")
        full = next(r for r in r5_full if r["dataset"] == ds and r["method"] == "DINO_F518")
        lines += ["", f"Full-data reference test mAP50–95: **{100*full['full_data_mAP50_95']:.3f}%**. DINO R5 평균과의 절대 차이: **{full['r5_minus_full_pp']:+.3f}%p** (기술적 참조).", ""]
    lines += ["## 5. 통계 결과", "",
              "표의 CI는 seed 5개의 대응 차이를 100,000회 percentile bootstrap으로 재표본 추출한 값이다. Exact two-sided sign-flip은 32개 부호 배치를 열거했다. 네 primary comparison에만 Holm 보정을 적용했다. n=5에서 최소 가능한 양측 p는 0.0625다.", "",
              "## 6. 클래스별 결과", "",
              "아래 값은 R5에서 seed 5개의 평균 AP50–95이며 단위는 %. Selected composition은 봉인된 R5 누적 선택 이미지 중 해당 클래스를 포함한 이미지 수의 평균이다.", "",
              "|Dataset|Class|Random AP|DINO AP|Entropy AP|DINO−Random (%p)|Random selected|DINO selected|Entropy selected|",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in class_diff:
        counts = {m: next(x["mean_selected_images"] for x in composition if x["dataset"] == r["dataset"] and x["class_name"] == r["class_name"] and x["method"] == m) for m in METHODS}
        lines.append(f"|{r['dataset']}|{r['class_name']}|{100*r['random_mean_ap']:.3f}|{100*r['dino_mean_ap']:.3f}|{100*r['entropy_mean_ap']:.3f}|{r['dino_minus_random_mean_ap_pp']:+.3f}|{counts['Random']:.1f}|{counts['DINO_F518']:.1f}|{counts['Entropy']:.1f}|")
    lines += ["", "Worst-class nAULC (%), 매 round의 최저 클래스 AP로 구성:", "",
              "|Dataset|Random|DINOv2-FF|Entropy|", "|---|---:|---:|---:|"]
    for ds in ("GC10-DET", "NEU-DET"):
        vals = {m: 100 * statistics.mean(r["worst_class_nAULC"] for r in worst_naulc if r["dataset"] == ds and r["method"] == m) for m in METHODS}
        lines.append(f"|{ds}|{vals['Random']:.3f}|{vals['DINO_F518']:.3f}|{vals['Entropy']:.3f}|")
    lines += ["", "## 7. Development vs Test", "",
              "|Dataset|Comparison|Dev difference (%p)|Test difference (%p)|Direction|",
              "|---|---|---:|---:|---|"]
    for r in dev_test:
        if r["comparison"] not in ("DINOv2-FF - Random", "DINOv2-FF - Entropy"):
            continue
        lines.append(f"|{r['dataset']}|{r['comparison']}|{r['dev_paired_mean_difference_pp']:+.3f}|{r['test_paired_mean_difference_pp']:+.3f}|{r['direction_relation']}|")
    lines += ["", "## 8. 논문에서 수정해야 할 핵심 수치", "",
              "국문초록·영문초록·3장 평가 설명·4장 결과·5장 논의·6장 결론에서 기존 ‘최종평가 미실시’ 문구와 development 전용 수치를 검토한다. 교체에 사용할 정확한 test 수치는 `statistics/` CSV와 `raw/results.csv`에 보존했다. 기존 문장은 여기서 임의로 다시 쓰지 않는다.",
              "", "## 9. 최종평가 결론", ""]
    for ds in ("GC10-DET", "NEU-DET"):
        r1 = next(r for r in comparisons if r["dataset"] == ds and r["comparison"] == "DINOv2-FF - Random" and r["metric"] == "mAP50_95_nAULC")
        r2 = next(r for r in comparisons if r["dataset"] == ds and r["comparison"] == "DINOv2-FF - Entropy" and r["metric"] == "mAP50_95_nAULC")
        relation = next(r["direction_relation"] for r in dev_test if r["dataset"] == ds and r["comparison"] == "DINOv2-FF - Random")
        heterogeneity = any(r["dataset"] == ds and r["dino_minus_random_mean_ap_pp"] < 0 for r in class_diff) and any(r["dataset"] == ds and r["dino_minus_random_mean_ap_pp"] > 0 for r in class_diff)
        lines += [f"- **{ds}:** DINOv2-FF의 test 평균 nAULC 차이는 Random 대비 {r1['paired_mean_difference_pp']:+.3f}%p, Entropy 대비 {r2['paired_mean_difference_pp']:+.3f}%p. Development 대비 Random 방향: {relation}. 클래스별 양·음 차이 공존: {'예' if heterogeneity else '아니오'}. Worst-class 값은 위 표와 원장 참조. n=5와 Holm 결과를 넘는 우월성 주장은 하지 않는다."]
    lines += ["", "## 10. 생성 파일 목록", "",
              "- `preflight/`: test 데이터·checkpoint 검증, dev dry-run, native label 검사, 고정 protocol, completeness",
              "- `raw/runs/<run_id>/`: 모델별 config, raw metrics JSON, console·Ultralytics 로그 및 원래 validator 출력",
              "- `raw/results.csv`, `raw/class_ap.csv`, `raw/results.json`: 논문 수치의 machine-readable 원장",
              "- `statistics/`: budget, seed별 nAULC, 대응 비교·bootstrap·Holm, class AP·composition, worst-class, full-data, dev/test 비교",
              "- `figures/`: 제목과 파일명에 TEST가 표시된 PNG 4개",
              "- `FINAL_TEST_REPORT_20260925.md`: 사람이 읽는 요약", ""]
    with (OUT / "FINAL_TEST_REPORT_20260925.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    complete = read_json(OUT / "preflight/test_completeness.json")
    check(complete["status"] == "PASS" and complete["successful"] == 162, "Completeness gate failed")
    lock = read_json(OUT / "preflight/protocol_lock.json")
    check(sha(OUT / "analysis.py") == lock["analysis_script_sha256"], "Analysis code changed after test opened")
    config = read_json(METADATA / "config/protocol.json")
    runs = read_json(METADATA / "config/cells.json")
    results = {r["run_id"]: read_json(OUT / "raw/runs" / r["run_id"] / "result.json") for r in runs}
    check(len(results) == 162, "Missing raw results")
    write_json_x(OUT / "raw/results.json", [results[k] for k in lock["evaluation_order"]])
    raw_rows, raw_classes = [], []
    for run_id in lock["evaluation_order"]:
        r = results[run_id]
        raw_rows.append({k: r[k] for k in ("dataset", "run_id", "acquisition_seed", "method", "round",
                         "cumulative_labeled_images", "cumulative_labeled_ratio", "checkpoint_path",
                         "checkpoint_sha256", "evaluation_dataset", "runtime_seconds", "status", "mAP50_95", "mAP50")})
        for name, ap in r["per_class_AP50_95"].items():
            raw_classes.append({"dataset": r["dataset"], "run_id": run_id, "class_name": name,
                                "AP50_95": ap})
    write_csv_x(OUT / "raw/results.csv", raw_rows, list(raw_rows[0]))
    write_csv_x(OUT / "raw/class_ap.csv", raw_classes, list(raw_classes[0]))

    curves, worst_curves, seed_naulc, worst_naulc, budget_summary = [], [], [], [], []
    test_naulc, worst_by_seed, dev_naulc = {}, {}, {}
    for ds in ("GC10-DET", "NEU-DET"):
        budgets = config["datasets"][ds]["counts"]
        tag = "GC10" if ds == "GC10-DET" else "NEU"
        for seed in SEEDS:
            for method in METHODS:
                values, worst_values, dev_values = [], [], []
                for rd, budget in enumerate(budgets):
                    run_id = f"{tag}_v2_a{seed}_SHARED_R0" if rd == 0 else f"{tag}_v2_a{seed}_{method}_R{rd}"
                    r = results[run_id]
                    check(r["cumulative_labeled_images"] == budget, f"Budget mismatch: {run_id}")
                    ap = r["per_class_AP50_95"]
                    check(set(ap) == set(config["classes"][ds]), f"Class AP missing: {run_id}")
                    minimum = min(ap.values())
                    min_classes = [k for k, v in ap.items() if v == minimum]
                    values.append(r["mAP50_95"])
                    worst_values.append(minimum)
                    saved = read_json(RUNS / run_id / "sealed_dev_metrics.json")
                    dev_values.append(saved["overall"]["metrics/mAP50-95(B)"])
                    curves.append({"dataset": ds, "seed": seed, "method": method, "round": rd,
                                   "run_id": run_id, "labeled_images": budget,
                                   "labeled_ratio": budget / config["datasets"][ds]["train"],
                                   "test_mAP50_95": r["mAP50_95"], "test_mAP50": r["mAP50"]})
                    worst_curves.append({"dataset": ds, "seed": seed, "method": method, "round": rd,
                                         "run_id": run_id, "labeled_images": budget,
                                         "worst_class_AP50_95": minimum,
                                         "worst_class_names": "|".join(min_classes)})
                value = aulc(values, budgets)
                worst = aulc(worst_values, budgets)
                dev = aulc(dev_values, budgets)
                test_naulc[(ds, seed, method)] = value
                worst_by_seed[(ds, seed, method)] = worst
                dev_naulc[(ds, seed, method)] = dev
                seed_naulc.append({"dataset": ds, "seed": seed, "method": method, "nAULC": value,
                                   "nAULC_percent": 100 * value, "development_nAULC": dev})
                worst_naulc.append({"dataset": ds, "seed": seed, "method": method,
                                    "worst_class_nAULC": worst, "worst_class_nAULC_percent": worst * 100})
        for method in METHODS:
            for rd, budget in enumerate(budgets):
                rows = [r for r in curves if r["dataset"] == ds and r["method"] == method and r["round"] == rd]
                vals = [r["test_mAP50_95"] for r in rows]
                budget_summary.append({"dataset": ds, "method": method, "round": rd,
                                       "labeled_images": budget, "mean_mAP50_95": statistics.mean(vals),
                                       "sample_sd_mAP50_95": statistics.stdev(vals), "n_seeds": 5})

    comparisons, worst_comparisons, dev_test = [], [], []
    for ds in ("GC10-DET", "NEU-DET"):
        for a, b in CONTRASTS:
            label = f"{METHOD_LABELS[a]} - {METHOD_LABELS[b]}"
            test = paired_stats(ds, label, a, b, test_naulc, "mAP50_95_nAULC")
            comparisons.append(test)
            worst_comparisons.append(paired_stats(ds, label, a, b, worst_by_seed, "worst_class_nAULC"))
            dev = statistics.mean(dev_naulc[(ds, seed, a)] - dev_naulc[(ds, seed, b)] for seed in SEEDS)
            observed = test["paired_mean_difference"]
            relation = "방향이 바뀜" if (dev > 0 > observed or dev < 0 < observed) else "방향은 같으나 크기가 달라짐" if abs(dev-observed) > 1e-12 else "재현됨"
            dev_test.append({"dataset": ds, "comparison": label,
                             "dev_paired_mean_difference_pp": dev * 100,
                             "test_paired_mean_difference_pp": observed * 100,
                             "direction_relation": relation})
    primary = holm_primary(comparisons)

    class_diff, class_seed, composition_seed, composition = [], [], [], []
    for ds in ("GC10-DET", "NEU-DET"):
        canonical = {r["sample_id"]: r for r in read_csv(PHASE1B / ds / "CANONICAL_INDEX.csv")}
        classes = config["classes"][ds]
        tag = "GC10" if ds == "GC10-DET" else "NEU"
        for seed in SEEDS:
            for method in METHODS:
                run_id = f"{tag}_v2_a{seed}_{method}_R5"
                r = results[run_id]
                selected = read_json(RUNS / run_id / "selection_manifest.json")["cumulative_ids"]
                check(len(selected) == len(set(selected)) == config["datasets"][ds]["counts"][-1],
                      f"R5 selection count mismatch: {run_id}")
                image_counts = Counter()
                box_counts = Counter()
                for id in selected:
                    check(id in canonical, f"Selected ID absent from canonical data: {id}")
                    boxes = json.loads(canonical[id]["boxes_json"])
                    for name in {box[0] for box in boxes}:
                        image_counts[name] += 1
                    for box in dict.fromkeys(tuple(x) for x in boxes):
                        box_counts[box[0]] += 1
                for name in classes:
                    class_seed.append({"dataset": ds, "seed": seed, "method": method,
                                       "class_name": name, "R5_AP50_95": r["per_class_AP50_95"][name]})
                    composition_seed.append({"dataset": ds, "seed": seed, "method": method,
                                             "class_name": name, "selected_images": image_counts[name],
                                             "selected_unique_boxes": box_counts[name]})
        for name in classes:
            means = {method: statistics.mean(r["R5_AP50_95"] for r in class_seed if r["dataset"] == ds and r["method"] == method and r["class_name"] == name) for method in METHODS}
            paired = [results[f"{tag}_v2_a{seed}_DINO_F518_R5"]["per_class_AP50_95"][name] -
                      results[f"{tag}_v2_a{seed}_Random_R5"]["per_class_AP50_95"][name] for seed in SEEDS]
            class_diff.append({"dataset": ds, "class_name": name,
                               "random_mean_ap": means["Random"], "dino_mean_ap": means["DINO_F518"],
                               "entropy_mean_ap": means["Entropy"],
                               "dino_minus_random_mean_ap_pp": 100 * statistics.mean(paired)})
            for method in METHODS:
                matching = [r for r in composition_seed if r["dataset"] == ds and r["method"] == method and r["class_name"] == name]
                composition.append({"dataset": ds, "method": method, "class_name": name,
                                    "mean_selected_images": statistics.mean(r["selected_images"] for r in matching),
                                    "mean_selected_unique_boxes": statistics.mean(r["selected_unique_boxes"] for r in matching),
                                    "n_seeds": 5})

    r5_full = []
    for ds in ("GC10-DET", "NEU-DET"):
        tag = "GC10" if ds == "GC10-DET" else "NEU"
        full = results[f"{tag}_v2_FULL"]["mAP50_95"]
        for method in METHODS:
            r5 = statistics.mean(results[f"{tag}_v2_a{seed}_{method}_R5"]["mAP50_95"] for seed in SEEDS)
            r5_full.append({"dataset": ds, "method": method, "r5_mean_mAP50_95": r5,
                            "full_data_mAP50_95": full, "r5_minus_full_pp": 100 * (r5 - full)})

    tables = {
        "curves.csv": curves, "budget_summary.csv": budget_summary,
        "seed_nAULC.csv": seed_naulc, "paired_comparisons.csv": comparisons,
        "worst_class_curves.csv": worst_curves, "worst_class_nAULC.csv": worst_naulc,
        "worst_class_comparisons.csv": worst_comparisons, "R5_class_AP_by_seed.csv": class_seed,
        "R5_class_AP_difference.csv": class_diff, "selected_class_composition_by_seed.csv": composition_seed,
        "selected_class_composition_summary.csv": composition, "R5_vs_full_data.csv": r5_full,
        "development_vs_test.csv": dev_test}
    for name, rows in tables.items():
        check(rows, f"Empty analysis table: {name}")
        write_csv_x(OUT / "statistics" / name, rows, list(rows[0]))
    write_json_x(OUT / "statistics/paired_comparisons.json", comparisons)
    write_json_x(OUT / "statistics/seed_nAULC.json", seed_naulc)
    write_json_x(OUT / "statistics/classwise_and_worst.json", {
        "R5_class_AP_difference": class_diff,
        "selected_class_composition_summary": composition,
        "worst_class_nAULC": worst_naulc,
        "R5_vs_full_data": r5_full})
    write_json_x(OUT / "statistics/statistical_protocol.json", {
        "unit": "acquisition seed", "n": 5, "nAULC": "trapezoid over actual six image counts / (R5-R0)",
        "bootstrap_draws": BOOTSTRAP, "bootstrap_type": "paired percentile 95%",
        "bootstrap_seeds": BOOTSTRAP_SEEDS, "exact_sign_flip_enumerations": 32,
        "exact_two_sided_rule": "abs(flipped_mean) >= abs(observed_mean) - 1e-15",
        "holm_family": [f"{ds}: {a} - {b}" for ds in ("GC10-DET", "NEU-DET")
                        for a, b in (("DINO_F518", "Random"), ("DINO_F518", "Entropy"))],
        "class_composition": "R5 cumulative selected image count by class; exact duplicate boxes collapsed"})
    figure_outputs(budget_summary, class_diff)
    markdown_report(config, results, budget_summary, seed_naulc, comparisons, class_diff,
                    composition, worst_naulc, r5_full, dev_test, primary)
    write_json_x(OUT / "statistics/analysis_complete.json", {"status": "PASS", "raw_results": 162,
                 "curve_rows": len(curves), "seed_nAULC_rows": len(seed_naulc),
                 "primary_comparisons": len(primary), "bootstrap_draws_each": BOOTSTRAP,
                 "figure_png": 4, "report": str(OUT / "FINAL_TEST_REPORT_20260925.md")})
    print("ANALYSIS COMPLETE: raw CSV/JSON, statistics, figures, Markdown report", flush=True)


if __name__ == "__main__":
    main()
