# vlm_error_correlation_20260511.py
# 목적: SBERT 불안정성 점수(s_inst)와 VLM 분류 에러율의 상관관계 분석

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 0. 파일 경로 설정 (본인 경로로 수정)
# ─────────────────────────────────────────
DATA_DIR = "./data"  # results CSV 파일들이 있는 폴더

FILES = {
    "NEU_DET":  f"/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project/logs/consistency_multi_dataset_20260511/NEU_DET_20260511_170927/summary/results_NEU_DET_20260511.csv",
    "KOLEKTOR": f"/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project/logs/consistency_multi_dataset_20260511/KOLEKTOR_20260511_170927/summary/results_KOLEKTOR_20260511.csv",
    "MVTEC":    f"/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project/logs/consistency_multi_dataset_20260511/MVTEC_20260511_170927/summary/results_MVTEC_20260511.csv",
}

# ─────────────────────────────────────────
# 1. GT 클래스 키워드 매핑
#    → 각 class_name에 대해 응답 텍스트에서 찾을 키워드 목록
# ─────────────────────────────────────────
KEYWORD_MAP = {
    # NEU_DET
    "crazing":        ["crazing", "craz", "crack network", "micro crack", "craze"],
    "inclusion":      ["inclusion", "inclus", "embedded particle", "foreign material"],
    "patches":        ["patch", "patches", "stain", "discolor", "spot", "blotch"],
    "pitted_surface": ["pit", "pitted", "pitting", "crater", "cavity", "hole", "indentation"],
    "rolled-in_scale":["scale", "rolled", "flake", "lamination", "delamination"],
    "scratches":      ["scratch", "scratche", "groove", "linear defect", "score mark"],
    # KOLEKTOR
    "defect":         ["defect", "crack", "fault", "flaw", "anomaly", "damage", "scratch", "pit"],
    "ok":             ["no defect", "normal", "clean", "uniform", "no visible", "intact", "no crack"],
    # MVTEC (카테고리별 공통 결함어 + 각 타입별)
    "good":           ["no defect", "normal", "clean", "uniform", "no visible", "intact"],
    "bent":           ["bent", "bend", "deform", "warp", "buckl"],
    "broken":         ["broken", "break", "fracture", "crack", "chip"],
    "color":          ["color", "colour", "discolor", "stain", "contaminat"],
    "contamination":  ["contaminat", "dirt", "foreign", "particle", "deposit"],
    "crack":          ["crack", "fracture", "split", "fissure"],
    "cut":            ["cut", "slit", "incision", "scratch", "gouge"],
    "flip":           ["flip", "flipped", "inverted", "rotated", "orient"],
    "hole":           ["hole", "pit", "void", "cavity", "missing"],
    "liquid":         ["liquid", "fluid", "wet", "stain", "blotch"],
    "misplace":       ["misplace", "misalign", "offset", "shift", "wrong position"],
    "missing":        ["missing", "absent", "lack", "incomplete"],
    "print":          ["print", "mark", "imprint", "label"],
    "scratch":        ["scratch", "groove", "linear", "score"],
    "squeeze":        ["squeeze", "compress", "deform", "distort"],
    "thread":         ["thread", "screw thread", "groove"],
}

def check_keyword_in_responses(row, keywords):
    """5개 응답 중 키워드 포함 비율 반환 (0.0 ~ 1.0)"""
    responses = [str(row.get(f'response_{i}', '') or '').lower() for i in range(1, 6)]
    hits = sum(1 for r in responses if any(kw.lower() in r for kw in keywords))
    return hits / 5.0

# ─────────────────────────────────────────
# 2. 데이터 로드 및 에러율 계산
# ─────────────────────────────────────────
def load_and_compute(dataset_name, filepath):
    df = pd.read_csv(filepath)
    print(f"\n[{dataset_name}] 로드 완료: {len(df)}행")

    # class_name 기준 키워드 매핑
    # MVTEC은 category 컬럼도 있을 경우 활용
    def get_keywords(row):
        cn = str(row['class_name']).lower()
        # MVTEC subcategory (예: capsule_crack → crack)
        for key in KEYWORD_MAP:
            if key in cn:
                return KEYWORD_MAP[key]
        # fallback: 클래스명 자체를 키워드로
        return [cn]

    df['keywords'] = df.apply(get_keywords, axis=1)
    df['mention_rate'] = df.apply(
        lambda r: check_keyword_in_responses(r, r['keywords']), axis=1
    )
    # 에러율: GT 언급 못한 비율 (높을수록 VLM이 해당 이미지를 못 맞춤)
    df['vlm_error_rate'] = 1.0 - df['mention_rate']

    # s_inst proxy: 1 - sbert_mean_score (높을수록 불안정)
    df['s_inst'] = 1.0 - df['sbert_mean_score']

    df['dataset'] = dataset_name
    return df

all_dfs = []
for name, path in FILES.items():
    try:
        df = load_and_compute(name, path)
        all_dfs.append(df)
    except FileNotFoundError:
        print(f"[경고] 파일 없음: {path}")

combined = pd.concat(all_dfs, ignore_index=True)

# ─────────────────────────────────────────
# 3. 통계 분석 (Spearman + Mann-Whitney U)
# ─────────────────────────────────────────
print("\n" + "="*60)
print("[ 상관관계 분석 결과 ]")
print("="*60)

results_table = []

for ds in combined['dataset'].unique():
    sub = combined[combined['dataset'] == ds].dropna(subset=['s_inst', 'vlm_error_rate'])

    # Spearman 상관
    r, p = stats.spearmanr(sub['s_inst'], sub['vlm_error_rate'])

    # Mann-Whitney U: 상위 33%(고불안정) vs 하위 33%(저불안정) 에러율 비교
    q33 = sub['s_inst'].quantile(0.33)
    q67 = sub['s_inst'].quantile(0.67)
    low_grp  = sub[sub['s_inst'] <= q33]['vlm_error_rate']
    high_grp = sub[sub['s_inst'] >= q67]['vlm_error_rate']
    mw_stat, mw_p = stats.mannwhitneyu(high_grp, low_grp, alternative='greater')

    # 판정
    if r > 0.3 and p < 0.05:
        verdict = "✅ 유효 신호 (AL 전진 가능)"
    elif r > 0.1 and p < 0.05:
        verdict = "⚠️  약한 신호 (프롬프트 개선 후 재검증)"
    else:
        verdict = "❌ 신호 불충분"

    print(f"\n데이터셋: {ds}")
    print(f"  샘플 수          : {len(sub)}")
    print(f"  Spearman r       : {r:.4f}  (p={p:.4f})")
    print(f"  고불안정 에러율  : {high_grp.mean():.4f}  저불안정 에러율: {low_grp.mean():.4f}")
    print(f"  Mann-Whitney p   : {mw_p:.4f}")
    print(f"  판정             : {verdict}")

    results_table.append({
        'dataset': ds, 'n': len(sub),
        'spearman_r': round(r, 4), 'spearman_p': round(p, 4),
        'high_inst_error': round(high_grp.mean(), 4),
        'low_inst_error': round(low_grp.mean(), 4),
        'mw_p': round(mw_p, 4), 'verdict': verdict
    })

# ─────────────────────────────────────────
# 4. 시각화
# ─────────────────────────────────────────
dataset_colors = {"NEU_DET": "#4C72B0", "KOLEKTOR": "#DD8452", "MVTEC": "#55A868"}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("VLM Instability (s_inst) vs. Classification Error Rate", fontsize=14, fontweight='bold')

for ax, (ds, color) in zip(axes, dataset_colors.items()):
    sub = combined[combined['dataset'] == ds].dropna(subset=['s_inst', 'vlm_error_rate'])
    r, p = stats.spearmanr(sub['s_inst'], sub['vlm_error_rate'])

    ax.scatter(sub['s_inst'], sub['vlm_error_rate'],
               alpha=0.6, color=color, edgecolors='white', s=60)

    # 추세선
    z = np.polyfit(sub['s_inst'], sub['vlm_error_rate'], 1)
    xline = np.linspace(sub['s_inst'].min(), sub['s_inst'].max(), 100)
    ax.plot(xline, np.poly1d(z)(xline), color='black', linewidth=1.5, linestyle='--', alpha=0.7)

    ax.set_title(f"{ds}\nSpearman r={r:.3f}, p={p:.3f}", fontsize=11)
    ax.set_xlabel("s_inst (1 - SBERT score)", fontsize=9)
    ax.set_ylabel("VLM Error Rate", fontsize=9)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.4)

    # p값에 따라 테두리 색상 표시
    border_color = 'green' if p < 0.05 else 'red'
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(2)

plt.tight_layout()
plt.savefig("./figure7_sinst_vs_error.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure 7 저장 완료: figure7_sinst_vs_error.png")

# ─────────────────────────────────────────
# 5. 결과 요약 CSV 저장
# ─────────────────────────────────────────
pd.DataFrame(results_table).to_csv("./correlation_summary.csv", index=False)
print("결과 테이블 저장: correlation_summary.csv")
