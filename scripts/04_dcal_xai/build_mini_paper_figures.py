from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"
OUT = ROOT / "runs" / "mini_paper_package_20260718"


NAVY = "#17324D"
BLUE = "#376F95"
GREEN = "#3F7D57"
RED = "#B84A4A"
AMBER = "#B77B24"
GRAY = "#6B7280"
PALE = "#F4F7FA"
PALE_BLUE = "#EAF1F7"
PALE_GREEN = "#E9F2EC"
PALE_RED = "#F7EDED"
PALE_AMBER = "#F8F1E5"
PALE_GRAY = "#EEF1F4"
WHITE = "#FFFFFF"


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for candidate in (Path(r"C:\Windows\Fonts\malgun.ttf"), Path(r"C:\Windows\Fonts\malgunsl.ttf")):
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(candidate)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["font.weight"] = 400
    return plt


def rounded_box(ax, xy: tuple[float, float], width: float, height: float, *, face: str = WHITE,
                edge: str = NAVY, linewidth: float = 1.4, radius: float = 0.12,
                linestyle: str = "-"):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle=f"round,pad=0.10,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=linewidth, linestyle=linestyle,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start: tuple[float, float], end: tuple[float, float], *, color: str = GRAY,
          style: str = "-", width: float = 1.4, head: str = "-|>") -> None:
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle=head, color=color, lw=width, linestyle=style, shrinkA=0, shrinkB=0),
    )


def save_figure(fig, stem: str, *, dpi: int = 240, svg: bool = True, pdf: bool = True) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    png = FIGURES / f"{stem}.png"
    # Keep the declared canvas size so the architecture export is exactly
    # 3840 x 2160 at 20 x 11.25 inches and 192 dpi.  ``bbox_inches='tight'``
    # silently cropped that publication canvas and reduced the PNG resolution.
    fig.savefig(png, dpi=dpi, facecolor=WHITE)
    outputs.append(png)
    if svg:
        path = FIGURES / f"{stem}.svg"
        fig.savefig(path, format="svg", facecolor=WHITE)
        outputs.append(path)
    if pdf:
        path = FIGURES / f"{stem}.pdf"
        fig.savefig(path, format="pdf", facecolor=WHITE)
        outputs.append(path)
    return outputs


def architecture_figure(plt) -> list[Path]:
    fig, ax = plt.subplots(figsize=(20, 11.25))
    ax.set_xlim(0, 20); ax.set_ylim(0, 11.25); ax.axis("off")

    ax.text(0.35, 10.86, "산업 결함 Active Learning 후보 신호의 단계적 타당성 평가",
            fontsize=23, color=NAVY, va="top")
    ax.text(0.35, 10.46, "Validity-gated empirical evaluation and cost-containment workflow",
            fontsize=12.2, color=GRAY, va="top")

    # A. Initial hypothesis.
    rounded_box(ax, (0.35, 7.50), 3.55, 2.35, face=PALE_RED, edge=RED, linestyle="--")
    ax.text(0.58, 9.55, "A  초기 가설", fontsize=13.4, color=RED, va="top")
    ax.text(0.62, 9.05, "VLM Consistency  ·  DINO Diversity\nDetector Uncertainty",
            fontsize=10.2, color=NAVY, va="top", linespacing=1.35)
    ax.text(2.12, 8.47, "↓", fontsize=13, color=GRAY, va="top", ha="center")
    ax.text(2.12, 8.18, "유용한 선택 → detector 성능 → label 비용 절감",
            fontsize=8.2, color=NAVY, va="top", ha="center")
    ax.text(0.60, 7.82, "점선 = 기각된 superiority 경로", fontsize=8.8, color=GRAY)

    # B. Six-stage gate pipeline.
    ax.text(4.35, 9.92, "B  Validity Gate Pipeline", fontsize=13.4, color=BLUE, va="bottom")
    gate_x = [4.35, 6.84, 9.33, 11.82, 14.31, 16.80]
    gate_titles = [
        ("G1", "Signal validity", "grounding\nerror ranking\ntransform confound"),
        ("G2", "Target discovery", "rare/anomaly yield\nenrichment@budget"),
        ("G3", "Composition safety", "coverage · entropy\nHHI · rare safety"),
        ("G4", "Acquisition reproducibility", "new acquisition seed\nfixed vs new set"),
        ("G5", "Learning utility", "mAP50-95 · recall\nrare AP · AULC"),
        ("G6", "Operational validity", "top-budget enrichment\nhuman/deployment value"),
    ]
    for idx, (x, (gid, title, details)) in enumerate(zip(gate_x, gate_titles)):
        face = PALE_BLUE if idx < 4 else (PALE_GREEN if idx == 4 else PALE_AMBER)
        edge = BLUE if idx < 4 else (GREEN if idx == 4 else AMBER)
        rounded_box(ax, (x, 7.55), 2.12, 2.15, face=face, edge=edge)
        ax.text(x + 0.18, 9.44, gid, fontsize=10.0, color=edge, va="top")
        ax.text(x + 0.18, 9.11, title, fontsize=10.4, color=NAVY, va="top")
        ax.text(x + 0.18, 8.55, details, fontsize=8.5, color=GRAY, va="top", linespacing=1.28)
        if idx < len(gate_x) - 1:
            arrow(ax, (x + 2.18, 8.62), (gate_x[idx + 1] - 0.08, 8.62), color=GRAY, width=1.2)

    rounded_box(ax, (4.38, 6.62), 14.58, 0.64, face=PALE_GREEN, edge=GREEN, linewidth=1.5)
    ax.text(11.67, 6.94,
            "Selection PASS는 성공 예측이 아니라, learning utility를 측정할 수 있는 1회 bounded detector screen 권한이다.",
            fontsize=12.2, color=NAVY, ha="center", va="center")
    ax.text(4.43, 6.32, "PASS  → 다음 검증 단계", fontsize=9.3, color=GREEN)
    ax.text(7.55, 6.32, "FAIL / NOT IDENTIFIABLE  → STOP · 확장 금지 · claim 축소", fontsize=9.3, color=RED)
    ax.text(15.25, 6.32, "Downstream FAIL  → backbone 확장 금지 · final 잠금", fontsize=9.3, color=AMBER)

    # C. Dataset mechanism cards.
    ax.text(0.35, 5.92, "C  데이터셋별 mechanism 사례", fontsize=13.4, color=BLUE, va="bottom")
    cards = [
        (0.35, "GC10-DET", "+2.720 rare images", "proxy diversity ↑", "q20 overall mAP 일부 ↑\nrare AP ↓ · K40 utility FAIL", "Diversification without\nreliable learning utility", GREEN),
        (4.18, "MPDD", "+6.245 anomalies", "capture-day ↑\nsource confound 67.3%", "downstream not tested", "Source-confounded\ndiversification", AMBER),
        (8.01, "VisA", "+14.480 anomalies", "category -4.110\nHHI +0.286025", "downstream not authorized", "Category collapse", RED),
    ]
    for x, name, discovery, composition, downstream, mechanism, color in cards:
        rounded_box(ax, (x, 1.55), 3.45, 4.15, face=WHITE, edge=color)
        ax.text(x + 0.20, 5.43, name, fontsize=12.2, color=NAVY, va="top")
        ax.text(x + 0.20, 4.92, "Discovery", fontsize=8.8, color=GREEN, va="top")
        ax.text(x + 0.20, 4.62, discovery, fontsize=10.2, color=NAVY, va="top")
        ax.text(x + 0.20, 4.10, "Composition", fontsize=8.8, color=AMBER, va="top")
        ax.text(x + 0.20, 3.80, composition, fontsize=9.5, color=NAVY, va="top", linespacing=1.25)
        ax.text(x + 0.20, 3.00, "Downstream", fontsize=8.8, color=RED, va="top")
        ax.text(x + 0.20, 2.70, downstream, fontsize=9.2, color=NAVY, va="top", linespacing=1.22)
        ax.text(x + 0.20, 1.99, mechanism, fontsize=8.8, color=color, va="top", linespacing=1.20)

    # D. Broken translations.
    ax.text(11.85, 5.92, "D  자동으로 성립하지 않은 연결", fontsize=13.4, color=RED, va="bottom")
    rounded_box(ax, (11.85, 1.55), 3.85, 4.15, face=PALE_RED, edge=RED)
    broken = [
        "VLM consistency  ≠  grounding",
        "Global coverage  ≠  local utility",
        "Discovery gain  ≠  composition safety",
        "Train-seed stability  ≠\nacquisition generalization",
        "Selection PASS  ≠  detector success",
        "Error AUROC  ≠  budget enrichment",
        "Metadata  ≠  independent production pool",
    ]
    y_positions = [5.28, 4.78, 4.28, 3.78, 3.12, 2.62, 2.12]
    for y, text in zip(y_positions, broken):
        ax.text(12.06, y + 0.03, "×", fontsize=10.5, color=RED, va="top")
        ax.text(12.36, y, text, fontsize=8.2, color=NAVY, va="top", linespacing=1.15)

    # E. Scientific and operational value.
    ax.text(16.00, 5.92, "E  최종 산출 가치", fontsize=13.4, color=GREEN, va="bottom")
    rounded_box(ax, (16.00, 1.55), 3.60, 4.15, face=PALE_GREEN, edge=GREEN)
    ax.text(16.20, 5.30, "Scientific output", fontsize=10.2, color=GREEN, va="top")
    ax.text(16.20, 4.92, "8개 failure mechanism\ndiscovery-safety-utility separation\nclaim-boundary enforcement",
            fontsize=8.9, color=NAVY, va="top", linespacing=1.27)
    ax.text(16.20, 3.80, "Cost containment", fontsize=10.2, color=GREEN, va="top")
    ax.text(16.20, 3.43, "D2R 15 + K40 30\n≥45 planned model runs stopped\nlocked final actual use = 0",
            fontsize=8.9, color=NAVY, va="top", linespacing=1.27)
    ax.text(16.20, 2.35, "RETROSPECTIVE EMPIRICAL EVALUATION\n+ COST-CONTAINMENT WORKFLOW",
            fontsize=8.9, color=GREEN, va="top", linespacing=1.22)
    ax.text(16.20, 1.82, "PREDICTIVE POLICY: NOT IDENTIFIABLE", fontsize=8.4, color=RED, va="top")

    rounded_box(ax, (0.35, 0.45), 19.25, 0.62, face=PALE_GRAY, edge=GRAY, linewidth=1.0)
    ax.text(9.98, 0.76,
            "후보 신호의 부분적 성공을 detector utility로 곧바로 해석하지 않고, 각 번역 단계를 검증한 뒤 다음 고비용 단계의 수행 권한을 제한한다.",
            fontsize=10.6, color=NAVY, ha="center", va="center")
    outputs = save_figure(fig, "full_page_validity_gated_architecture", dpi=192, svg=True, pdf=True)
    plt.close(fig)
    return outputs


def flowchart_figure(plt) -> list[Path]:
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(10.5, 14.5))
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 14.5); ax.axis("off")
    ax.text(0.45, 14.05, "Algorithm 1. Validity-Gated Authorization", fontsize=20, color=NAVY, va="top")
    ax.text(0.45, 13.62, "고비용 다음 단계를 수행할 근거를 통제하며, downstream 성공을 예측하지 않는다.",
            fontsize=10.4, color=GRAY, va="top")

    center_x, box_w, box_h = 5.25, 5.70, 0.82

    def step(y: float, title: str, detail: str, color: str = BLUE, face: str = PALE_BLUE) -> None:
        rounded_box(ax, (center_x - box_w/2, y), box_w, box_h, face=face, edge=color)
        ax.text(center_x, y + 0.56, title, fontsize=11.2, color=NAVY, ha="center", va="center")
        ax.text(center_x, y + 0.23, detail, fontsize=8.5, color=GRAY, ha="center", va="center")

    def diamond(y: float, label: str) -> None:
        pts = [(center_x, y + 0.55), (center_x + 1.18, y), (center_x, y - 0.55), (center_x - 1.18, y)]
        ax.add_patch(Polygon(pts, closed=True, facecolor=PALE_AMBER, edgecolor=AMBER, linewidth=1.3))
        ax.text(center_x, y, label, fontsize=9.2, color=NAVY, ha="center", va="center")

    step(12.35, "0. Protocol freeze", "가설 · endpoint · threshold · stopping rule · provenance")
    arrow(ax, (center_x, 12.30), (center_x, 11.86))
    step(10.95, "G1. Signal validity audit", "grounding · error ranking · transform confound")
    arrow(ax, (center_x, 10.90), (center_x, 10.35))
    diamond(9.72, "PASS?")
    rounded_box(ax, (0.42, 9.16), 2.35, 1.05, face=PALE_RED, edge=RED)
    ax.text(1.60, 9.83, "FAIL / NA", fontsize=9.4, color=RED, ha="center")
    ax.text(1.60, 9.50, "STOP\nexpensive path 금지", fontsize=8.7, color=NAVY, ha="center", va="top")
    arrow(ax, (4.02, 9.72), (2.82, 9.72), color=RED, style="--")
    ax.text(3.32, 9.88, "아니오", fontsize=8.0, color=RED)
    arrow(ax, (center_x, 9.15), (center_x, 8.78), color=GREEN)
    ax.text(5.45, 9.03, "예", fontsize=8.0, color=GREEN)

    step(7.92, "G2-G3. Selection-only discovery + composition safety", "target yield와 category/source/session/rare safety를 함께 평가")
    arrow(ax, (center_x, 7.86), (center_x, 7.32))
    diamond(6.69, "Discovery + safety PASS?")
    rounded_box(ax, (0.42, 6.03), 2.35, 1.25, face=PALE_AMBER, edge=AMBER)
    ax.text(1.60, 6.90, "Safety FAIL", fontsize=9.4, color=AMBER, ha="center")
    ax.text(1.60, 6.58, "STOP 또는\ndiscovery-only claim", fontsize=8.5, color=NAVY, ha="center", va="top")
    arrow(ax, (4.02, 6.69), (2.82, 6.69), color=AMBER, style="--")
    arrow(ax, (center_x, 6.14), (center_x, 5.74), color=GREEN)

    step(4.88, "G4. Acquisition-set confirmation", "새 acquisition seed · fixed set vs new selection")
    arrow(ax, (center_x, 4.82), (center_x, 4.28))
    diamond(3.67, "Selection gates PASS?")
    rounded_box(ax, (7.40, 3.06), 2.65, 1.20, face=PALE_GREEN, edge=GREEN)
    ax.text(8.72, 3.88, "BOUNDED_SCREEN", fontsize=9.4, color=GREEN, ha="center")
    ax.text(8.72, 3.55, "정확히 1회 learner screen\n성공 보장 아님", fontsize=8.4, color=NAVY, ha="center", va="top")
    arrow(ax, (6.44, 3.67), (7.34, 3.67), color=GREEN)
    rounded_box(ax, (0.42, 3.07), 2.35, 1.18, face=PALE_RED, edge=RED)
    ax.text(1.60, 3.85, "FAIL / NA", fontsize=9.4, color=RED, ha="center")
    ax.text(1.60, 3.52, "STOP", fontsize=8.7, color=NAVY, ha="center")
    arrow(ax, (4.02, 3.67), (2.82, 3.67), color=RED, style="--")
    step(1.92, "G5. Downstream learning utility", "mAP50-95 · recall · rare AP · AULC", color=GREEN, face=PALE_GREEN)
    arrow(ax, (8.72, 3.02), (7.55, 2.80), color=GREEN)
    rounded_box(ax, (0.42, 0.48), 3.05, 1.08, face=PALE_RED, edge=RED)
    ax.text(1.94, 1.21, "FAIL", fontsize=9.4, color=RED, ha="center")
    ax.text(1.94, 0.88, "STOP · 확장 학습 및 final 금지", fontsize=8.3, color=NAVY, ha="center")
    arrow(ax, (4.02, 1.92), (3.35, 1.58), color=RED, style="--")
    rounded_box(ax, (6.65, 0.42), 3.40, 1.30, face=PALE_AMBER, edge=AMBER)
    ax.text(8.35, 1.43, "G6. Operational validity", fontsize=9.4, color=AMBER, ha="center")
    ax.text(8.35, 1.10, "top-budget enrichment · human/deployment relevance", fontsize=7.5, color=NAVY, ha="center")
    ax.text(8.35, 0.72, "PASS 시에만 locked final authorization", fontsize=8.1, color=GREEN, ha="center")
    arrow(ax, (6.48, 1.92), (6.78, 1.70), color=GREEN)

    ax.text(5.25, 0.12, "모든 단계: consumed/avoided cost · provenance · CLAIM_SCOPE 기록",
            fontsize=9.2, color=GRAY, ha="center")
    outputs = save_figure(fig, "validity_gated_algorithm_flowchart", dpi=220, svg=True, pdf=True)
    plt.close(fig)
    return outputs


def read_timeline() -> list[dict[str, str]]:
    path = DOCS / "framework_branch_timeline_20260718.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def paper_temporal_figure(plt) -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 6.5); ax.axis("off")
    ax.text(0.35, 6.08, "Temporal audit의 식별 경계", fontsize=18, color=NAVY)
    panels = [
        (0.40, "요청된 predictive metrics", ["early-stop recall: NA", "false-advance rate: NA", "correct-stop precision: NA"], RED, PALE_RED),
        (4.43, "확인 가능한 process evidence", ["explicit protocol 11/15", "script-only chronology 4/15", "generic policy precommit 0/6"], GREEN, PALE_GREEN),
        (8.46, "판정", ["C. RETROSPECTIVE_AUDIT_ONLY", "authorization + cost containment", "predictive policy 아님"], AMBER, PALE_AMBER),
    ]
    for x, title, lines, edge, face in panels:
        rounded_box(ax, (x, 1.20), 3.65, 4.35, face=face, edge=edge)
        ax.text(x + 0.22, 5.15, title, fontsize=11.3, color=edge, va="top")
        for idx, line in enumerate(lines):
            ax.text(x + 0.25, 4.45 - idx*0.72, "• " + line, fontsize=10.2, color=NAVY, va="top")
    ax.text(0.42, 0.52, "NA는 기준 미달이 아니라, generic policy의 사전동결과 STOP branch의 counterfactual truth가 없어 계산이 성립하지 않음을 뜻한다.",
            fontsize=9.5, color=GRAY)
    path = FIGURES / "paper_framework_temporal_validation.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=WHITE); plt.close(fig)
    return path


def paper_cost_figure(plt) -> Path:
    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    labels, values = ["D2R detector confirmation", "K40 YOLOv8s expansion"], [15, 30]
    bars = ax.barh(labels, values, color=[BLUE, GREEN], height=0.50)
    for bar, value in zip(bars, values):
        ax.text(value + 0.6, bar.get_y() + bar.get_height()/2, f"{value} models", va="center", fontsize=11, color=NAVY)
    ax.set_xlim(0, 35); ax.grid(axis="x", color="#D9E0E7", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("실행하지 않은, 명시적으로 계획된 detector model runs", color=NAVY)
    ax.set_title("Documented cost containment: lower bound = 45 model runs", loc="left", fontsize=17, color=NAVY, pad=15)
    ax.text(0.0, -0.18, "실제 미실행 run만 합산 · GPU-hours/금액 미추정 · locked final actual use = 0 · counterfactual avoided uses = NA",
            transform=ax.transAxes, fontsize=9.5, color=GRAY)
    path = FIGURES / "paper_framework_cost_avoidance.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=WHITE); plt.close(fig)
    return path


def paper_timeline_figure(plt) -> Path:
    from datetime import datetime
    from matplotlib.dates import DateFormatter, DayLocator

    rows = sorted(read_timeline(), key=lambda x: x["outcome_time"])
    times = [datetime.strptime(item["outcome_time"], "%Y-%m-%d %H:%M:%S") for item in rows]
    fig, ax = plt.subplots(figsize=(13.0, 8.2))
    ys = list(range(len(rows)))
    for y, time, item in zip(ys, times, rows):
        passed = "PASS" in item["local_gate"]
        explicit = item["protocol_evidence_status"] == "EXPLICIT_PROTOCOL_BEFORE_OUTCOME"
        color, marker = (GREEN, "D") if passed else (RED, "o")
        ax.hlines(y, min(times), time, color="#D9E0E7", linewidth=1.0)
        ax.scatter(time, y, s=60, marker=marker, facecolor=color if explicit else WHITE,
                   edgecolor=color, linewidth=1.5, zorder=3)
    ax.axhspan(8.5, 14.5, color=BLUE, alpha=0.06)
    ax.text(max(times), 8.65, "후기 35%: generic policy pre-frozen 아님", ha="right", va="bottom", fontsize=9, color=BLUE)
    ax.set_yticks(ys, [f'{item["branch_id"]} {item["branch"]}' for item in rows], fontsize=8.4)
    ax.invert_yaxis(); ax.xaxis.set_major_locator(DayLocator()); ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
    ax.grid(axis="x", color="#D9E0E7", linewidth=0.8); ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("Recorded outcome chronology (local filesystem)", color=NAVY)
    ax.set_title("15개 연구 branch의 실제 chronology와 authorization 기록", loc="left", fontsize=17, color=NAVY, pad=15)
    ax.text(0.0, -0.10, "● FAIL/STOP   ◆ PASS 후 bounded screen   빈 표식: script-only chronology   음영: temporal holdout으로 해석할 수 없는 후기 35%",
            transform=ax.transAxes, fontsize=9.0, color=GRAY)
    path = FIGURES / "paper_framework_advance_stop_timeline.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=WHITE); plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build publication figures only; no training or inference.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    timeline = DOCS / "framework_branch_timeline_20260718.csv"
    if not timeline.exists():
        raise FileNotFoundError(timeline)
    config: dict[str, Any] = {
        "training_performed": False,
        "inference_performed": False,
        "vlm_calls_performed": False,
        "embedding_extraction_performed": False,
        "selector_implementation_performed": False,
        "fn_screen_performed": False,
        "final_test_used": False,
    }
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_OK", **config}, ensure_ascii=False, indent=2))
        return
    plt = configure_matplotlib()
    outputs = []
    outputs.extend(architecture_figure(plt))
    outputs.extend(flowchart_figure(plt))
    outputs.extend([paper_temporal_figure(plt), paper_cost_figure(plt), paper_timeline_figure(plt)])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figure_build_config.json").write_text(
        json.dumps({"outputs": [str(path.relative_to(ROOT)).replace("\\", "/") for path in outputs], **config}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"status": "DONE", "figures": len(outputs), "outputs": [str(path) for path in outputs], **config}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
