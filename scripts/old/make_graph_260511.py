import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# ── [중요] 실제 경로 설정 ──────────────────────────────────
LOG_BASE = Path("/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project/logs/consistency_multi_dataset_20260511")

PATH_NEU = LOG_BASE / "NEU_DET_20260511_170927/summary/results_NEU_DET_20260511.csv"
PATH_MVT = LOG_BASE / "MVTEC_20260511_170927/summary/results_MVTEC_20260511.csv"
PATH_KOL = LOG_BASE / "KOLEKTOR_20260511_170927/summary/results_KOLEKTOR_20260511.csv"

# ── 공통 스타일 설정 ──────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

BLUE, ORANGE, GREEN = '#2E74B5', '#E36C09', '#375623'

# ── 데이터 로드 ──────────────────────────────────────────
df_neu = pd.read_csv(PATH_NEU)
df_kol = pd.read_csv(PATH_KOL)
df_mvt = pd.read_csv(PATH_MVT)


def get_stats(df):
    r_mean = df[df['prompt_group'] == 'rephrase']['sbert_mean_score'].mean()
    r_std = df[df['prompt_group'] == 'rephrase']['sbert_mean_score'].std()
    v_mean = df[df['prompt_group'] == 'viewpoint']['sbert_mean_score'].mean()
    v_std = df[df['prompt_group'] == 'viewpoint']['sbert_mean_score'].std()
    return r_mean, v_mean, r_std, v_std


# ════════════════════════════════════════════════════════════
# Figure 2: Grouped Bar — 데이터 기반 동적 생성
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

datasets = ['NEU_DET', 'KOLEKTOR', 'MVTEC']
stats = [get_stats(df_neu), get_stats(df_kol), get_stats(df_mvt)]
rephrase = [s[0] for s in stats]
viewpoint = [s[1] for s in stats]
rep_std = [s[2] for s in stats]
vp_std = [s[3] for s in stats]

x = np.arange(len(datasets))
w = 0.32

ax.bar(x - w / 2, rephrase, w, yerr=rep_std, capsize=5, color=BLUE, alpha=0.85, label='Rephrase (s_wp)')
ax.bar(x + w / 2, viewpoint, w, yerr=vp_std, capsize=5, color=ORANGE, alpha=0.85, label='Viewpoint (s_av)')

for i, (r, v) in enumerate(zip(rephrase, viewpoint)):
    gap = r - v
    ax.annotate('', xy=(x[i] + w / 2, v + 0.02), xytext=(x[i] - w / 2, r - 0.02),
                arrowprops=dict(arrowstyle='<->', color='gray'))
    ax.text(x[i] + 0.02, (r + v) / 2, f'Δ={gap:.3f}', fontsize=9, color='#444', va='center')

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontweight='bold')
ax.set_ylabel('SBERT Mean Score')
ax.set_ylim(0.4, 1.0)
ax.set_title('Figure 2. Rephrase vs. Viewpoint SBERT Consistency', pad=12)
ax.legend(loc='lower right')
plt.savefig('./figure2_rephrase_vs_viewpoint.png')
print("Figure 2 저장 완료")

# ════════════════════════════════════════════════════════════
# Figure 3: Heatmap — NEU_DET 클래스별 분석
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
neu_cls_stats = df_neu.groupby(['class_name', 'prompt_group'])['sbert_mean_score'].mean().unstack()
neu_cls_stats['Gap'] = neu_cls_stats['rephrase'] - neu_cls_stats['viewpoint']

sns.heatmap(neu_cls_stats[['rephrase', 'viewpoint']], ax=axes[0], annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5,
            vmax=0.95)
axes[0].set_title('SBERT Score (NEU_DET)')

axes[1].barh(neu_cls_stats.index, neu_cls_stats['Gap'], color=plt.cm.Reds(neu_cls_stats['Gap'] / 0.3))
axes[1].set_title('Instability Gap by Class')
plt.tight_layout()
plt.savefig('./figure3_neu_class_heatmap.png')
print("Figure 3 저장 완료")


# ════════════════════════════════════════════════════════════
# Figure 4: Scatter Plot — 샘플별 분포
# ════════════════════════════════════════════════════════════
def get_scatter_data(df):
    rep = df[df['prompt_group'] == 'rephrase'][['image_path', 'sbert_mean_score']].rename(
        columns={'sbert_mean_score': 'rep_score'})
    vp = df[df['prompt_group'] == 'viewpoint'][['image_path', 'sbert_mean_score']].rename(
        columns={'sbert_mean_score': 'vp_score'})
    return pd.merge(rep, vp, on='image_path')


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, df, name, col in zip(axes, [df_neu, df_kol, df_mvt], datasets, [BLUE, ORANGE, GREEN]):
    data = get_scatter_data(df)
    ax.scatter(data['rep_score'], data['vp_score'], color=col, alpha=0.6, edgecolors='w')
    ax.plot([0.4, 1], [0.4, 1], 'k--', alpha=0.3)

    # [수정된 부분] 숫자 컬럼(rep_score, vp_score)만 사용하여 상관계수 계산
    corr = data[['rep_score', 'vp_score']].corr().iloc[0, 1]

    ax.set_title(f'{name} (r={corr:.3f})')
    ax.set_xlim(0.4, 1.0);
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel('Rephrase');
    ax.set_ylabel('Viewpoint')

plt.tight_layout()
plt.savefig('./figure4_scatter_rep_vs_vp.png')
print("Figure 4 저장 완료")

# ════════════════════════════════════════════════════════════
# Figure 5: MVTEC Category Gap & s_inst
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
mvt_cat_stats = df_mvt.groupby(['category', 'prompt_group'])['sbert_mean_score'].mean().unstack()
mvt_cat_stats['Gap'] = mvt_cat_stats['rephrase'] - mvt_cat_stats['viewpoint']

axes[0].barh(mvt_cat_stats.index, mvt_cat_stats['Gap'], color='salmon')
axes[0].set_title('MVTEC Category-level Gap')

# s_inst 계산: α=0.4, β=0.6
s_inst_scores = []
for d in stats:
    s_inst = 0.4 * (1 - d[0]) + 0.6 * (1 - d[1])
    s_inst_scores.append(s_inst)

axes[1].bar(datasets, s_inst_scores, color=[BLUE, ORANGE, GREEN], width=0.5)
axes[1].set_title('Composite Instability Score (s_inst)')
for i, v in enumerate(s_inst_scores):
    axes[1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('./figure5_gap_sinst.png')
print("Figure 5 저장 완료")

# ════════════════════════════════════════════════════════════
# Figure 6: Boxplot 통합
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5.5))
combined_df = pd.concat([df_neu.assign(ds='NEU_DET'), df_kol.assign(ds='KOLEKTOR'), df_mvt.assign(ds='MVTEC')])
sns.boxplot(data=combined_df, x='ds', y='sbert_mean_score', hue='prompt_group',
            palette={'rephrase': BLUE, 'viewpoint': ORANGE}, ax=ax)
ax.set_title('SBERT Score Distribution by Dataset')
plt.savefig('./figure6_boxplot_distribution.png')
print("Figure 6 저장 완료")

print("\n🎉 모든 그래프가 현재 경로에 저장되었습니다.")