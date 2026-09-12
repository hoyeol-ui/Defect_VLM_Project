"""Static scientific figures from local audited CSVs; no model imports."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R=Path(__file__).resolve().parent;T=R/'tables';F=R/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':16,'axes.labelsize':13,'legend.fontsize':11,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
C={'Random':'#657786','DINO':'#1565B8','Entropy':'#D65A22','Full':'#292D35'}
DS=['GC10-DET','NEU-DET'];M=['Random','DINO','Entropy'];E='Evidence: Development validation | Acquisition seed n=10; learner A/B averaged | Locked-final: not used'
def read(n):return list(csv.DictReader((T/(n+'.csv')).open(encoding='utf-8-sig')))
def subset(rows,**kw):return [r for r in rows if all(r[k]==str(v) for k,v in kw.items())]
def save(fig,name,footer=E):
    fig.text(.5,.022,footer,ha='center',va='bottom',fontsize=10,color='#454b55')
    for ext in ['png','svg','pdf']:fig.savefig(F/(name+'.'+ext),dpi=260)
    plt.close(fig)
budget=read('method_budget_summary');pairs=read('seed_budget_AB_means');effects=read('paired_nAULC_seed_effects');summ=read('paired_nAULC_comparison');full=read('full_data_reference_summary')
# 1: workflow, selection lineage separate from cold-start detector training.
fig,ax=plt.subplots(figsize=(15,7.5));fig.subplots_adjust(left=.02,right=.98,top=.88,bottom=.12);ax.set(xlim=(0,15),ylim=(0,6));ax.axis('off')
fig.suptitle('Study workflow: three selection signals, one development endpoint',fontsize=20,x=.5,y=.96)
def box(x,y,w,h,text,color='#EDF2F7'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.1,rounding_size=0.12',facecolor=color,edgecolor='#c4cdd8'))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=12)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.6,'color':'#556477'})
box(.15,2.1,2.5,1.6,'Sealed train pool\nGC10 1,605 / NEU 1,259\n10 acquisition seeds\nCommon R0: 10%')
for y,m,txt in [(4.3,'Random','Uniform selection'),(2.6,'DINO','Frozen DINOv2-small\nCosine farthest-first'),(.9,'Entropy','Detection-confidence\nbinary entropy SUM')]:
    box(3.4,y,3.1,1.2,m+'\n'+txt,{'Random':'#edf0f3','DINO':'#eaf2fc','Entropy':'#fff0e8'}[m]);arrow((2.65,2.9),(3.4,y+.6))
box(7.15,2.15,2.45,1.6,'R0 → R5\n10% → about 35%\nA/B learners\n100 epochs per cell')
for y in [4.9,3.2,1.5]:arrow((6.5,y),(7.15,2.95))
box(10.2,2.15,4.4,1.6,'YOLOv8n: sealed cold start\nIdentical validation membership\n100-epoch CSV maximum mAP50–95\nA/B mean → seed-wise six-point nAULC')
arrow((9.6,2.95),(10.2,2.95))
box(10.2,.15,4.4,1.15,'Full-label reference: 4 runs\n100% pool; A/B mean per dataset\nDescriptive reference, not upper bound')
ax.text(5.1,.10,'Entropy selection uses previous-round\nlearner A best; training always cold-starts.',fontsize=10,ha='center',va='bottom',color='#4d5868')
ax.text(10.2,4.35,'440 historical + 200 Entropy + 4 full-data\nShared R0 is reused; never counted as new training.',fontsize=11)
save(fig,'figure_1_research_workflow',E+' | Full-data n=2 learners/dataset')
# 2: full curves with seed SD, not standard errors.
fig,axes=plt.subplots(1,2,figsize=(14,6.5));fig.subplots_adjust(left=.07,right=.985,top=.80,bottom=.18,wspace=.22)
for ax,ds,letter in zip(axes,DS,'ab'):
    for m in M:
        g=sorted(subset(budget,dataset=ds,method=m),key=lambda r:int(r['round']))
        x=np.array([int(r['label_budget']) for r in g]);y=np.array([float(r['mean'])*100 for r in g]);sd=np.array([float(r['std'])*100 for r in g])
        ax.fill_between(x,y-sd,y+sd,color=C[m],alpha=.09,lw=0)
        ax.plot(x,y,'o-',color=C[m],label=m,lw=2.5,ms=6)
    ax.set_title(f'({letter}) {ds}');ax.set_xlabel('Labeled training images (R0–R5)');ax.set_ylabel('Development mAP50–95 (%)');ax.set_xticks(x);ax.grid(axis='y',alpha=.18)
fig.suptitle('Three-way label-efficiency curves',fontsize=21,y=.96)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=3,frameon=False)
fig.text(.5,.09,'Lines: mean of 10 A/B-averaged seeds. Shading: ±1 seed SD, not a confidence interval. Shared R0 is identical.',ha='center',fontsize=10)
save(fig,'figure_2_development_curves')
# 3: paired nAULC, 10 raw paired points and bootstrap CI.
contr=['DINO - Random','Entropy - Random','DINO - Entropy']
fig,axes=plt.subplots(1,2,figsize=(14,6.4));fig.subplots_adjust(left=.15,right=.975,top=.82,bottom=.21,wspace=.38)
for ax,ds,letter in zip(axes,DS,'ab'):
    for j,comp in enumerate(contr):
        g=sorted(subset(effects,dataset=ds,comparison=comp),key=lambda r:r['seed']);x=np.array([float(r['difference_pp']) for r in g]);ys=j+np.linspace(-.15,.15,10)
        ax.scatter(x,ys,s=28,color='#8996a5',alpha=.8,zorder=2)
        s=subset(summ,dataset=ds,comparison=comp)[0];mean=float(s['mean_difference_pp']);lo=float(s['ci95_low_pp']);hi=float(s['ci95_high_pp'])
        ax.errorbar(mean,j,xerr=[[mean-lo],[hi-mean]],fmt='D',color='#173a60',capsize=5,lw=2.7,ms=8,zorder=3)
        ax.text(.98,j+.31,f'{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]',transform=ax.get_yaxis_transform(),ha='right',fontsize=11)
    ax.set_yticks(range(3),[s.replace(' - ',' − ') for s in contr]);ax.set_ylim(2.55,-.55);ax.axvline(0,color='#555',ls='--',lw=1);ax.set_xlabel('Paired nAULC difference (percentage points)');ax.set_title(f'({letter}) {ds}');ax.grid(axis='x',alpha=.15)
fig.suptitle('Seed-wise paired nAULC effects',fontsize=21,y=.96)
fig.text(.5,.88,'Grey points: 10 paired seeds | Diamond: mean | Bar: 95% paired percentile bootstrap CI (100,000 draws)',ha='center',fontsize=11)
fig.text(.5,.10,'Descriptive development intervals; no confirmatory p-values. An interval crossing zero does not establish equivalence.',ha='center',fontsize=10)
save(fig,'figure_3_paired_nAULC')
# 4: display actual full A/B points and distinguish them from n=10 AL seeds.
fig,axes=plt.subplots(1,2,figsize=(14,6.5));fig.subplots_adjust(left=.07,right=.98,top=.84,bottom=.20,wspace=.25)
for ax,ds,letter in zip(axes,DS,'ab'):
    f=subset(full,dataset=ds)[0];fm=float(f['mean_map50_95'])*100
    ax.axhline(fm,color='#444',ls='--',lw=1.2)
    for j,m in enumerate(M):
        g=sorted(subset(pairs,dataset=ds,method=m,round=5),key=lambda r:r['seed']);y=np.array([float(r['map50_95'])*100 for r in g])
        ax.scatter(j+np.linspace(-.13,.13,10),y,s=32,color=C[m],alpha=.6)
        ax.scatter(j,y.mean(),marker='D',s=80,color=C[m],edgecolor='white',zorder=4)
        ax.text(j,y.max()+.5,f'{y.mean()/fm*100:.1f}% of full',ha='center',fontsize=10)
    ax.scatter([2.96,3.04],[float(f['A_map50_95'])*100,float(f['B_map50_95'])*100],marker='x',s=85,lw=2,color=C['Full'])
    ax.scatter(3,fm,marker='D',s=75,color=C['Full']);ax.text(3,fm+.75,f'{fm:.2f}%',ha='center',fontsize=11)
    ax.set_xticks(range(4),['Random','DINO','Entropy','Full A/B']);ax.set_ylabel('Development mAP50–95 (%)');ax.set_title(f'({letter}) {ds}');ax.grid(axis='y',alpha=.15)
    ax.set_ylim(min(float(r['map50_95'])*100 for r in subset(pairs,dataset=ds,round=5))-1,fm+2.1)
    frac=561/1605 if ds=='GC10-DET' else 441/1259
    ax.set_xlabel(f'AL: {frac*100:.2f}% of training images | Full: 100%')
fig.suptitle('R5 versus the full-label reference',fontsize=21,y=.96)
fig.text(.5,.10,'AL dots: 10 seed means. Full crosses: two individual learners. Diamonds: means. Image count is not annotation cost.',ha='center',fontsize=10)
save(fig,'figure_4_full_label_reference',E+' | Full-data n=2, not a statistical upper bound')
# 5: mechanisms and composition. Join historical class geometry BY NAME, never numeric class index.
co=read('selection_behavior_summary');qd=read('entropy_query_diagnostics')
geo=list(csv.DictReader((R/'historical_geometry/selection_vs_geometry.csv').open(encoding='utf-8-sig')))
fig,axes=plt.subplots(3,2,figsize=(16,17));fig.subplots_adjust(left=.15,right=.96,top=.91,bottom=.13,wspace=.45,hspace=.47)
for col,ds in enumerate(DS):
    ax=axes[0,col];g=subset(co,dataset=ds,method='DINO');names=[x['class_name'] for x in g];pos=np.arange(len(names))
    for k,m in enumerate(M):
        v={x['class_name']:float(x['mean_prevalence_difference_pp']) for x in subset(co,dataset=ds,method=m)}
        ax.scatter([v[n] for n in names],pos+(k-1)*.17,color=C[m],s=45,label=m)
    ax.set_yticks(pos,[n.replace('_',' ') for n in names],fontsize=11);ax.invert_yaxis();ax.axvline(0,color='#666',lw=1,ls='--');ax.set_xlabel('Selected − current candidate prevalence (pp)');ax.set_title(ds+' | class composition',fontsize=15);ax.grid(axis='x',alpha=.15)
    if col==0:ax.legend(loc='lower left',ncol=3,bbox_to_anchor=(-.13,1.06),frameon=False)
    ax=axes[1,col];g=subset(geo,dataset=ds,round=5)
    x=np.array([float(r['within_class_pair_median']) for r in g]);y=np.array([float(r['dino_minus_random']) for r in g])
    ax.scatter(x,y,color=C['DINO'],s=65);ax.axhline(0,color='#666',ls='--',lw=1)
    for r,xx,yy in zip(g,x,y):
        offset=(4,-13) if r['class_name']=='crazing' else (4,5)
        ax.annotate(r['class_name'].replace('_',' '),(xx,yy),xytext=offset,textcoords='offset points',fontsize=9)
    ax.margins(x=.35,y=.2);ax.set_xlabel('Within-class cosine distance median\nSmaller = more compact frozen DINO features');ax.set_ylabel('R5 cumulative selected images\nDINO − Random (10-seed mean)');ax.set_title('Existing DINO geometry diagnostic',fontsize=15);ax.grid(alpha=.15)
    ax=axes[2,col]
    for rnd in range(1,6):
        g=sorted(subset(qd,dataset=ds,round=rnd),key=lambda r:r['seed']);v=np.array([float(r['score_count_spearman']) for r in g]);ax.scatter(rnd+np.linspace(-.12,.12,10),v,s=27,color=C['Entropy'],alpha=.6)
    meds=[np.median([float(r['score_count_spearman']) for r in subset(qd,dataset=ds,round=rnd)]) for rnd in range(1,6)]
    ax.plot(range(1,6),meds,'D-',lw=2.2,color=C['Entropy']);ax.set_ylim(.84,1);ax.set_xticks(range(1,6),['R1','R2','R3','R4','R5']);ax.set_xlabel('Acquisition round (10 queries per round)');ax.set_ylabel('Within-query Spearman ρ');ax.set_title('Entropy SUM versus detection count',fontsize=15);ax.grid(axis='y',alpha=.15)
fig.suptitle('Different selection signals produce different sample compositions',fontsize=22,y=.975)
fig.text(.5,.942,'Top: newly selected image prevalence relative to each current candidate pool; mean of 5 rounds × 10 seeds.',ha='center',fontsize=12)
fig.text(.5,.052,'Middle: frozen historical post-hoc geometry reused; class names reconciled across ID orders. Bottom: saved scores only.\nClasses overlap in multi-label images; rounds are dependent. These associations do not establish downstream causation.',ha='center',fontsize=11)
save(fig,'figure_5_selection_behavior','Evidence: acquisition-pool labels + frozen feature diagnostics + saved score logs | 10 trajectories/dataset | Locked-final: not used')
print('Saved 5 figures × PNG/SVG/PDF')
