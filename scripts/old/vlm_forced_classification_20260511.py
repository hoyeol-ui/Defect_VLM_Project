# vlm_forced_classification_20260511.py (경로 반영 최종본)
# 목적: VLM 강제 분류 정답/오답 → s_inst 상관관계 분석

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 0. 경로 설정
# ─────────────────────────────────────────
BASE_DIR  = "/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project"
LOG_DIR   = f"{BASE_DIR}/logs/consistency_multi_dataset_20260511"
SAVE_DIR  = f"{LOG_DIR}/forced_cls_results"
os.makedirs(SAVE_DIR, exist_ok=True)

FILES = {
    "NEU_DET":  f"{LOG_DIR}/NEU_DET_20260511_170927/summary/results_NEU_DET_20260511.csv",
    "KOLEKTOR": f"{LOG_DIR}/KOLEKTOR_20260511_170927/summary/results_KOLEKTOR_20260511.csv",
    "MVTEC":    f"{LOG_DIR}/MVTEC_20260511_170927/summary/results_MVTEC_20260511.csv",
}

# ─────────────────────────────────────────
# 1. 설정
# ─────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE     = "mps"   # Apple Silicon → "mps" | CUDA → "cuda" | CPU → "cpu"

CHOICE_MAP = {
    "NEU_DET":  ["crazing", "inclusion", "patches", "pitted_surface",
                 "rolled-in_scale", "scratches"],
    "KOLEKTOR": ["defect", "ok"],
    "MVTEC":    ["bent", "broken", "color", "contamination", "crack", "cut",
                 "flip", "hole", "good", "liquid", "misplace", "missing",
                 "print", "scratch", "squeeze", "thread"],
}

# ─────────────────────────────────────────
# 2. 모델 로드
# ─────────────────────────────────────────
print("모델 로드 중...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("모델 로드 완료\n")

# ─────────────────────────────────────────
# 3. 강제 분류 함수 (PIL 직접 로딩)
# ─────────────────────────────────────────
def forced_classify(image_path: str, choices: list[str]) -> str:
    """
    PIL로 이미지를 직접 읽어 VLM에 강제 분류 질문.
    반환값: 선택지 중 매칭된 클래스명 (소문자)
    """
    # 파일 존재 확인
    if not os.path.exists(image_path):
        print(f"  [SKIP] 파일 없음: {image_path}")
        return "file_not_found"

    pil_image = Image.open(image_path).convert("RGB")
    choice_str = ", ".join(choices)

    prompt_text = (
        f"Look at this industrial surface image carefully.\n"
        f"Which ONE of the following defect types best describes what you see?\n"
        f"Choices: {choice_str}\n"
        f"Reply with ONLY the single best matching word from the choices above. "
        f"Do not explain. Do not add punctuation."
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},  # PIL 직접 전달
            {"type": "text",  "text": prompt_text},
        ]
    }]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,   # greedy decoding → 재현성 확보
        )

    generated = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0].strip().lower()

    # 선택지 매칭 (부분 포함 허용)
    for c in choices:
        if c.lower() in generated:
            return c.lower()

    # 매칭 실패 → raw 응답 반환 (오답 처리됨)
    return generated

# ─────────────────────────────────────────
# 4. MVTEC class_name → defect type 추출
#    예: "capsule_crack" → "crack"
#        "metal_nut_good" → "good"
# ─────────────────────────────────────────
def extract_mvtec_gt(class_name: str, choices: list[str]) -> str:
    cn = class_name.lower().strip()
    # good 처리
    if cn == "good" or cn.endswith("_good"):
        return "good"
    # choices 역순 정렬 (긴 키워드 먼저 매칭 → "pitted_surface" 같은 경우 대비)
    for c in sorted(choices, key=len, reverse=True):
        if cn.endswith(f"_{c}") or cn == c:
            return c
    # fallback: 마지막 언더스코어 뒤 토큰
    return cn.split("_")[-1]

# ─────────────────────────────────────────
# 5. KOLEKTOR GT 처리
#    class_name이 이미 "defect" / "ok" 이므로 그대로 사용
# ─────────────────────────────────────────
def extract_kolektor_gt(class_name: str) -> str:
    cn = class_name.lower().strip()
    if "defect" in cn or "fault" in cn or "crack" in cn:
        return "defect"
    if cn == "ok" or "good" in cn or "normal" in cn:
        return "ok"
    return cn  # fallback

# ─────────────────────────────────────────
# 6. 메인 실행
# ─────────────────────────────────────────
all_results = []

for ds_name, filepath in FILES.items():
    if not os.path.exists(filepath):
        print(f"[ERROR] CSV 없음: {filepath}")
        continue

    df = pd.read_csv(filepath)
    choices = CHOICE_MAP[ds_name]

    # image_index + image_path 기준 중복 제거 (rephrase/viewpoint 각 1행)
    df_unique = df.drop_duplicates(subset=["image_index", "image_path"]).copy()
    total = len(df_unique)
    print(f"[{ds_name}] 강제 분류 시작: {total}개 이미지")

    records = []
    for idx, (_, row) in enumerate(df_unique.iterrows(), 1):
        image_path = str(row["image_path"])
        class_name = str(row["class_name"]).lower().strip()

        # GT 정규화
        if ds_name == "MVTEC":
            gt_label = extract_mvtec_gt(class_name, choices)
        elif ds_name == "KOLEKTOR":
            gt_label = extract_kolektor_gt(class_name)
        else:
            gt_label = class_name  # NEU_DET: 그대로 사용

        # VLM 강제 분류
        pred_label = forced_classify(image_path, choices)

        # 정답/오답 판정
        if pred_label == "file_not_found":
            is_correct = np.nan
        else:
            is_correct = int(
                gt_label == pred_label or
                gt_label in pred_label or
                pred_label in gt_label
            )
        is_error = 1 - is_correct if not np.isnan(is_correct) else np.nan

        # s_inst 계산: 해당 이미지의 rephrase+viewpoint SBERT 평균
        img_rows   = df[df["image_path"] == image_path]
        sbert_mean = img_rows["sbert_mean_score"].mean()
        s_inst     = 1.0 - sbert_mean

        records.append({
            "dataset":    ds_name,
            "image_path": image_path,
            "class_name": class_name,
            "gt_label":   gt_label,
            "pred_label": pred_label,
            "is_correct": is_correct,
            "vlm_error":  is_error,
            "s_inst":     round(s_inst, 6),
            "sbert_mean": round(sbert_mean, 6),
        })

        # 진행 상황 (매 10건 또는 마지막)
        if idx % 10 == 0 or idx == total:
            done = [r for r in records if not np.isnan(r["is_correct"])]
            acc_so_far = np.mean([r["is_correct"] for r in done]) if done else 0
            print(f"  {idx}/{total}  |  현재 정확도: {acc_so_far:.3f}  |  pred 예시: '{records[-1]['pred_label']}'")

    ds_df = pd.DataFrame(records)
    all_results.append(ds_df)

    valid = ds_df.dropna(subset=["is_correct"])
    acc = valid["is_correct"].mean()
    err = valid["vlm_error"].mean()
    print(f"  → [{ds_name}] 정확도: {acc:.4f} / 에러율: {err:.4f}\n")

    # 데이터셋별 중간 저장
    ds_save = f"{SAVE_DIR}/forced_cls_{ds_name}_20260511.csv"
    ds_df.to_csv(ds_save, index=False)
    print(f"  중간 저장: {ds_save}")

# 전체 합치기
combined = pd.concat(all_results, ignore_index=True)
combined_path = f"{SAVE_DIR}/forced_cls_combined_20260511.csv"
combined.to_csv(combined_path, index=False)
print(f"\n전체 결과 저장: {combined_path}")

# ─────────────────────────────────────────
# 7. 상관관계 분석
# ─────────────────────────────────────────
print("\n" + "="*60)
print("[ 강제 분류 기반 상관관계 분석 결과 ]")
print("="*60)

summary_rows = []
dataset_colors = {"NEU_DET": "#4C72B0", "KOLEKTOR": "#DD8452", "MVTEC": "#55A868"}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("s_inst vs. VLM Forced Classification Error\n(Spearman Correlation)",
             fontsize=13, fontweight='bold')

for ax, ds_name in zip(axes, ["NEU_DET", "KOLEKTOR", "MVTEC"]):
    sub = combined[
        (combined["dataset"] == ds_name) &
        combined["vlm_error"].notna() &
        combined["s_inst"].notna()
    ].copy()
    color = dataset_colors[ds_name]

    if len(sub) < 5:
        ax.set_title(f"{ds_name}\n데이터 부족")
        continue

    r, p = stats.spearmanr(sub["s_inst"], sub["vlm_error"])

    # 고/저 불안정 그룹
    q33 = sub["s_inst"].quantile(0.33)
    q67 = sub["s_inst"].quantile(0.67)
    low_g  = sub[sub["s_inst"] <= q33]["vlm_error"]
    high_g = sub[sub["s_inst"] >= q67]["vlm_error"]
    low_err  = low_g.mean()
    high_err = high_g.mean()

    # Mann-Whitney U
    if len(high_g) > 0 and len(low_g) > 0:
        _, mw_p = stats.mannwhitneyu(high_g, low_g, alternative='greater')
    else:
        mw_p = np.nan

    # 판정
    if r > 0.3 and p < 0.05:
        verdict = "✅ 유효 신호 → AL 전진 가능"
    elif r > 0.1 and p < 0.05:
        verdict = "⚠️  약한 신호"
    else:
        verdict = "❌ 신호 불충분"

    print(f"\n{ds_name}")
    print(f"  n={len(sub)}, 전체 에러율={sub['vlm_error'].mean():.4f}")
    print(f"  Spearman r={r:.4f}, p={p:.4f}")
    print(f"  고불안정 에러율={high_err:.4f} / 저불안정 에러율={low_err:.4f}")
    print(f"  Mann-Whitney p={mw_p:.4f}")
    print(f"  판정: {verdict}")

    summary_rows.append({
        "dataset": ds_name, "n": len(sub),
        "overall_error": round(sub["vlm_error"].mean(), 4),
        "spearman_r": round(r, 4), "spearman_p": round(p, 4),
        "high_inst_error": round(high_err, 4),
        "low_inst_error":  round(low_err, 4),
        "mw_p": round(mw_p, 4) if not np.isnan(mw_p) else None,
        "verdict": verdict,
    })

    # 산점도 (binary → jitter로 겹침 방지)
    np.random.seed(42)
    jitter = np.random.uniform(-0.02, 0.02, size=len(sub))
    ax.scatter(sub["s_inst"], sub["vlm_error"] + jitter,
               alpha=0.55, color=color, edgecolors="white", s=55, linewidths=0.5)

    # 추세선
    z = np.polyfit(sub["s_inst"], sub["vlm_error"], 1)
    xline = np.linspace(sub["s_inst"].min(), sub["s_inst"].max(), 100)
    ax.plot(xline, np.poly1d(z)(xline), "k--", linewidth=1.5, alpha=0.7)

    # 평균선 (고/저 그룹)
    ax.axhline(high_err, color="darkred",  linestyle=":", linewidth=1.2,
               label=f"High s_inst err={high_err:.2f}")
    ax.axhline(low_err,  color="steelblue", linestyle=":", linewidth=1.2,
               label=f"Low s_inst err={low_err:.2f}")

    ax.set_title(f"{ds_name}\nSpearman r={r:.3f}, p={p:.3f}", fontsize=11)
    ax.set_xlabel("s_inst  (1 − SBERT score)", fontsize=9)
    ax.set_ylabel("VLM Error  (0=정답, 1=오답)", fontsize=9)
    ax.set_ylim(-0.2, 1.2)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)

    border_color = "green" if (r > 0.1 and p < 0.05) else "red"
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(2)

plt.tight_layout()
fig_path = f"{SAVE_DIR}/figure8_forced_cls_correlation_20260511.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nFigure 8 저장: {fig_path}")

# 요약 저장
summary_path = f"{SAVE_DIR}/forced_cls_summary_20260511.csv"
pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
print(f"요약 저장: {summary_path}")
