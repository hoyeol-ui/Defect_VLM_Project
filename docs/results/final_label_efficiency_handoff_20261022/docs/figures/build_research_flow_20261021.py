"""Paper/report diagram; protocol only, no invented performance values."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib import font_manager
P=Path(__file__).resolve().parent
font=font_manager.FontProperties(fname='C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family':font.get_name(),'svg.fonttype':'none','axes.unicode_minus':False})
fig,ax=plt.subplots(figsize=(15.5,9.5));fig.patch.set_facecolor('#ffffff');ax.set(xlim=(0,15.5),ylim=(0,9.5));ax.axis('off')
navy='#17324d';blue='#1f6eab';orange='#b56317';muted='#53687d'

def box(x,y,w,h,label,fill='#edf5fc',edge='#b5d1e8',color=navy,size=13,weight='normal'):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.12',linewidth=1.25,facecolor=fill,edgecolor=edge)
    ax.add_patch(patch);ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,color=color,weight=weight,linespacing=1.65)
def arrow(a,b,color=muted):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,linewidth=1.45,color=color,connectionstyle='arc3'))

ax.text(.35,9.13,'제한된 라벨에서의 결함 탐지 학습 효율',fontsize=23,color=navy,weight='bold',va='center')
ax.text(.36,8.70,'GC10-DET · NEU-DET  |  표본 선택 전략과 전체 학습 기준 비교',fontsize=12.5,color=muted)
box(.4,7.65,14.7,.68,'데이터 구성     GC10: 학습 1,604 · 개발 343 · 테스트 344장     /     NEU: 학습 1,258 · 개발 270 · 테스트 270장',fill='#f3f6f9',edge='#d7e0e8',size=12)
ax.text(.56,7.25,'능동학습',color=blue,fontsize=13.5,weight='bold')
ax.text(11.10,7.25,'전체 학습 기준',color=orange,fontsize=13.5,weight='bold')
arrow((5.1,7.62),(5.1,6.92),blue);arrow((12.85,7.62),(12.85,6.92),orange)
box(.55,6.25,9.15,.65,'공유 초기 라벨 10%  →  검출기 학습',fill='#edf5fc',edge='#b5d1e8',size=14,weight='bold')
box(10.65,5.62,4.35,1.27,'전체 학습 풀 100%\nGC10 1,604장 · NEU 1,258장',fill='#fff3e6',edge='#e9cba8',color=orange,size=13,weight='bold')
centers=[2.,5.1,8.2]
for c in centers:arrow((5.1,6.20),(c,5.77),blue)
box(.55,4.92,2.9,.78,'Random\n무작위 선택',size=13)
box(3.65,4.92,2.9,.78,'Frozen DINOv2\n다양성 선택',size=13)
box(6.75,4.92,2.9,.78,'Entropy\n검출기 불확실성 선택',size=12)
for c in centers:arrow((c,4.88),(c,4.45),blue)
box(.55,3.62,9.15,.80,'15% → 20% → 25% → 30% → 35%\n각 예산에서 선택과 학습을 반복',fill='#edf5fc',edge='#b5d1e8',size=13)
arrow((5.1,3.58),(5.1,3.05),blue);arrow((12.83,5.57),(12.83,3.05),orange)
box(.55,2.31,14.45,.71,'YOLO26s 학습     SGD + Cosine  ·  100 epoch  ·  입력 640  ·  batch 8',fill=navy,edge=navy,color='white',size=14,weight='bold')
arrow((7.77,2.27),(7.77,1.95))
box(.55,1.35,14.45,.56,'개발 데이터로 best 선정  →  전체 모델·설정 확정  →  별도 최종 평가',fill='#f3f6f9',edge='#d7e0e8',size=12.5)
for c in [2.9,7.75,12.6]:arrow((c,1.31),(c,1.02))
box(.55,.40,4.65,.57,'학습곡선 · nAULC',fill='#ecf6f1',edge='#c4ddce',size=13,weight='bold')
box(5.43,.40,4.65,.57,'3개 대응 반복의 평균 ± 표준편차',fill='#ecf6f1',edge='#c4ddce',size=12,weight='bold')
box(10.32,.40,4.68,.57,'35% 라벨의 100% 기준 대비 도달률',fill='#ecf6f1',edge='#c4ddce',size=12,weight='bold')
fig.text(.052,.035,'라벨 예산의 분모는 학습 풀이다. 100% 학습에 개발·테스트 이미지를 포함하지 않는다. nAULC는 10~35% 구간에서 계산한다.',fontsize=10,color=muted)
fig.subplots_adjust(left=.025,right=.99,top=.99,bottom=.065)
for ext in ['svg','png']:
    fig.savefig(P/f'20261021_research_flow.{ext}',dpi=180,facecolor='white')
plt.close(fig)
