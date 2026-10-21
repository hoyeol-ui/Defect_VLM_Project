"""Meaningful edge cases for the new geometry adapter and statistical summaries."""
from pathlib import Path
import os,sys,json,copy
P=Path(__file__).resolve().parent
os.environ['MODERN_YOLO_RUN']=str(P/'smoke/cpu_checks')
sys.path[:0]=[str(P/'src'),str(P)]
import numpy as np
from entropy_adapter import xyxy_to_xywh
from entropy import score
from summarize import summarize,naulc,COUNTS,BUDGETS,METHODS

def main():
    checks=[]
    for nc in [6,10]:
        raw=np.vstack([np.array([[1,10],[2,20],[5,30],[8,40]],np.float32),np.full((nc,2),.5,np.float32)])
        converted=xyxy_to_xywh(raw)
        assert np.array_equal(converted[4:],raw[4:])
        assert np.array_equal(converted[:4],[[3,20],[5,30],[4,20],[6,20]])
        assert abs(score(converted,nc)['uncertainty']-1)<1e-12
        assert score(converted*.0,nc)['uncertainty']==0
        threshold=converted.copy();threshold[4:]=.25
        assert score(threshold,nc)['detections']==0
        onehot=converted.copy();onehot[4:]=0;onehot[4]=.5
        assert score(onehot,nc)['uncertainty']==0
        checks.append(f'nc{nc}: geometry preserves all class scores; uniform/onehot/empty/exact-threshold entropy')
    raw=np.array([[10,10,10],[10,10,10],[4,4,4],[4,4,4],[.6,.6,0],[0,0,.6]],np.float32)
    assert score(raw,2)['anchor_indices']==[0,2]
    checks.append('stable confidence tie, same-class suppression, different-class retention')
    assert naulc([1,2,4,7,11,16],[.4]*6)==.4
    rows=[]
    for ds,counts in COUNTS.items():
        for pair in [1,2,3]:
            for method in METHODS:
                for k,b in enumerate(BUDGETS):
                    rows.append(dict(dataset=ds,pair=pair,method=method,budget_percent=b,train_n=counts[k],
                        mAP50_95=.1+.01*k+.001*pair,mAP50=.3,precision=.4,recall=.5,partition='test',technical_only=False))
            rows.append(dict(dataset=ds,pair=pair,method='FullySupervised',budget_percent=100,
                train_n=1604 if ds=='GC10-DET' else 1258,mAP50_95=.3+.001*pair,mAP50=.5,precision=.6,recall=.7,partition='test',technical_only=False))
    result=summarize(rows)
    assert len(result['naulc'])==6 and all(v['n']==3 for v in result['naulc'])
    assert all(v['mean']==0 and v['exact_two_sided_sign_flip_p']==1 for v in result['paired_comparisons'])
    assert abs(result['label_efficiency'][0]['ratio_of_means_percent']-100*.152/.302)<1e-12
    for mutation in ['missing','duplicate','dev','zero_full']:
        bad=copy.deepcopy(rows)
        if mutation=='missing':bad.pop()
        elif mutation=='duplicate':bad.append(copy.deepcopy(bad[0]))
        elif mutation=='dev':bad[0]['partition']='development'
        else:next(r for r in bad if r['method']=='FullySupervised')['mAP50_95']=0
        try:summarize(bad)
        except AssertionError:pass
        else:raise AssertionError('Invalid input accepted: '+mutation)
    checks.append('actual-count nAULC, paired n=3, ratio denominator, missing/duplicate/dev/zero-reference rejection')
    out=P/'evidence/cpu_checks.json';out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({'status':'PASS','checks':checks,'synthetic_values_not_research_results':True},indent=2),encoding='utf-8')
    print(out.read_text())

if __name__=='__main__':main()
