"""Frozen paired summaries, Full attainment and exact sign-flip/Holm4."""
import itertools
import math
from pathlib import Path
import statistics as st
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *

def stats(values):return {'n':len(values),'mean':st.mean(values),'sample_sd':st.stdev(values),'paired_values':values}
def area(xs,ys):
    require(len(xs)==len(ys)==6 and all(b>a for a,b in zip(xs,xs[1:])),'nAULC x grid')
    return sum((b-a)*(u+v)/2 for a,b,u,v in zip(xs,xs[1:],ys,ys[1:]))/(xs[-1]-xs[0])
def signflip(diff):
    observed=abs(st.mean(diff))
    return sum(abs(st.mean(d*s for d,s in zip(diff,signs)))>=observed-1e-15 for signs in itertools.product((-1,1),repeat=len(diff)))/(2**len(diff))

def main():
    verify_seal();models=read(E/'TEST_MODELS.json')['models'];lookup={};physical=[]
    for m in models:
        out=RUN/'models'/m['cell_id'];check_hash(out/'metrics.json',read(out/'RECEIPT.json')['metrics_sha'])
        r=read(out/'metrics.json');require(r['cell_id']==m['cell_id'] and r['model_sha']==m['best_sha'],'Metric provenance')
        lookup[(m['dataset'],m['method'],m['budget_percent'],m['pair'])]=r
        physical.append({'model':m,'metrics':r['overall'],'per_class':r['class_metrics']})
    logical=[];summaries=[];full_attainment=[];curves={}
    metric_keys=['metrics/mAP50(B)','metrics/mAP50-95(B)']
    for ds in DATASETS:
        for method in METHODS:
            for budget in (10,15,20,25,30,35):
                values={k:[] for k in metric_keys};ratios={k:[] for k in metric_keys}
                for pair in range(1,6):
                    r=lookup[(ds,'SHARED_R0' if budget==10 else method,budget,pair)]
                    f=lookup[(ds,'Full',100,pair)]
                    logical.append({'dataset':ds,'method':method,'budget_percent':budget,'pair':pair,'physical_cell':r['cell_id'],'metrics':r['overall']})
                    for key in metric_keys:
                        score=r['overall'][key];full=f['overall'][key];values[key].append(score*100)
                        ratios[key].append(None if full==0 else 100*score/full)
                summaries.append({'dataset':ds,'method':method,'budget_percent':budget,'metrics_percent':{k:stats(v) for k,v in values.items()}})
                full_attainment.append({'dataset':ds,'method':method,'budget_percent':budget,'paired_Full_attainment_percent':{k:stats(v) if all(x is not None for x in v) else {'undefined_zero_denominator':True,'paired_values':v} for k,v in ratios.items()}})
            auc=[]
            for pair in range(1,6):
                selected=[next(m for m in models if m['dataset']==ds and m['pair']==pair and m['budget_percent']==b and m['method']==('SHARED_R0' if b==10 else method)) for b in (10,15,20,25,30,35)]
                xs=[m['train_images'] for m in selected];ys=[lookup[(ds,m['method'],m['budget_percent'],pair)]['overall']['metrics/mAP50-95(B)'] for m in selected]
                auc.append(area(xs,ys)*100)
            curves[(ds,method)]=auc
        for pair in range(1,6):
            r=lookup[(ds,'Full',100,pair)];logical.append({'dataset':ds,'method':'Full','budget_percent':100,'pair':pair,'physical_cell':r['cell_id'],'metrics':r['overall']})
        summaries.append({'dataset':ds,'method':'Full','budget_percent':100,'metrics_percent':{k:stats([lookup[(ds,'Full',100,p)]['overall'][k]*100 for p in range(1,6)]) for k in metric_keys}})
    comparisons=[]
    for ds in DATASETS:
        for other in (METHODS[0],METHODS[2]):
            diffs=[a-b for a,b in zip(curves[(ds,METHODS[1])],curves[(ds,other)])]
            comparisons.append({'dataset':ds,'comparison':METHODS[1]+' minus '+other,'nAULC_difference_percentage_points':stats(diffs),'exact_two_sided_sign_flip_p':signflip(diffs),'assignments':32})
    running=0
    for rank,c in enumerate(sorted(comparisons,key=lambda x:x['exact_two_sided_sign_flip_p'])):
        running=max(running,(4-rank)*c['exact_two_sided_sign_flip_p']);c['Holm4_adjusted_p']=min(1,running)
    require(len(logical)==190 and len(physical)==170,'Physical/logical cardinality')
    dest=ROOT/'results';dest.mkdir(exist_ok=False)
    put(dest/'TEST_SUMMARY.json',{'partition':'test','physical_models':170,'logical_rows':190,'mean_sd_units':'percentage points; sample SD ddof1','summaries':summaries,'Full_attainment':full_attainment,'nAULC': [{'dataset':ds,'method':method,**stats(v)} for (ds,method),v in curves.items()],'paired_comparisons':comparisons,'limitations':read(E/'TEST_PLAN.json')['limitations']})
    put(dest/'PHYSICAL_MODEL_METRICS.json',physical);put(dest/'LOGICAL_190_ROWS.json',logical)
    lines=['# 로컬170모델 최종 test 결과','', '고정된 test 입력: GC10-DET343장, NEU-DET270장. 아래는 mAP50–95(%) 평균±표본표준편차,5 paired seeds.','', '|데이터셋|예산|Random|DINOv2|Entropy|','|---|---:|---:|---:|---:|']
    for ds in DATASETS:
        for b in (10,15,20,25,30,35):
            vals=[next(s for s in summaries if s['dataset']==ds and s['method']==m and s['budget_percent']==b)['metrics_percent']['metrics/mAP50-95(B)'] for m in METHODS]
            lines.append('|'+ds+'|'+str(b)+'%|'+'|'.join(f"{x['mean']:.2f} ± {x['sample_sd']:.2f}" for x in vals)+'|')
    for ds in DATASETS:
        f=next(s for s in summaries if s['dataset']==ds and s['method']=='Full')['metrics_percent']['metrics/mAP50-95(B)'];lines.extend(['',f"{ds} Full: {f['mean']:.2f} ± {f['sample_sd']:.2f}%",''])
    lines.extend(['','nAULC는 실제 이미지 수 구간의 사다리꼴 적분/전체 구간 길이로 계산.4개 paired 비교는 양측 exact sign-flip+Holm. n=5에서 유의성이나 동등성을 과장하지 않는다.','',*read(E/'TEST_PLAN.json')['limitations']])
    with (dest/'TEST_RESULTS_KO.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print('TEST_SUMMARY_WRITTEN',dest,flush=True)

if __name__=='__main__':main()
