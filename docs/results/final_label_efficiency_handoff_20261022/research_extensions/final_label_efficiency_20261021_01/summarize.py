"""Strict final-result summarizer. CSV inputs must contain actual frozen test evaluations."""
import csv,json,statistics,itertools,math
from pathlib import Path
METHODS=['Random','DINO_F518','Entropy']
COUNTS={'GC10-DET':[160,241,321,401,481,561],'NEU-DET':[126,189,252,315,377,440]}
BUDGETS=[10,15,20,25,30,35]

def naulc(x,y):
    assert len(x)==len(y)==6 and all(b>a for a,b in zip(x,x[1:]))
    return math.fsum((b-a)*(c+d)/2 for a,b,c,d in zip(x,x[1:],y,y[1:]))/(x[-1]-x[0])

def stats(x):return {'mean':statistics.mean(x),'sd':statistics.stdev(x),'n':len(x)}

def summarize(rows):
    keyed={}
    for r in rows:
        assert r['partition']=='test' and str(r['technical_only']).lower()=='false','Use scientific frozen test results only'
        ds=r['dataset'];pair=int(r['pair']);method=r['method'];budget=int(r['budget_percent'])
        assert ds in COUNTS and pair in [1,2,3] and method in METHODS+['FullySupervised']
        assert (budget==100 if method=='FullySupervised' else budget in BUDGETS)
        key=(ds,pair,method,budget);assert key not in keyed,'Duplicate scientific cell'
        n=int(r['train_n']);assert n==({'GC10-DET':1604,'NEU-DET':1258}[ds] if budget==100 else COUNTS[ds][BUDGETS.index(budget)])
        values={m:float(r[m]) for m in ['mAP50_95','mAP50','precision','recall']}
        assert all(math.isfinite(v) and 0<=v<=1 for v in values.values())
        keyed[key]=values
    assert len(keyed)==114,'Need 108 AL curve rows (shared 10% repeated by method) and 6 Full rows'
    curves=[];eff=[];auc=[];comparisons=[]
    for ds,counts in COUNTS.items():
        full=[keyed[ds,p,'FullySupervised',100]['mAP50_95'] for p in [1,2,3]]
        assert min(full)>0,'Undefined Full reference ratio'
        for pair in [1,2,3]:
            assert all(keyed[ds,pair,m,10]==keyed[ds,pair,'Random',10] for m in METHODS),'Shared R0 mismatch'
        bymethod={}
        for method in METHODS:
            bymethod[method]=[naulc(counts,[keyed[ds,p,method,b]['mAP50_95'] for b in BUDGETS]) for p in [1,2,3]]
            auc.append({'dataset':ds,'method':method,**stats(bymethod[method])})
            for budget in BUDGETS:
                for metric in ['mAP50_95','mAP50','precision','recall']:
                    values=[keyed[ds,p,method,budget][metric] for p in [1,2,3]]
                    curves.append({'dataset':ds,'method':method,'budget_percent':budget,'metric':metric,**stats(values)})
            low=[keyed[ds,p,method,35]['mAP50_95'] for p in [1,2,3]]
            eff.append({'dataset':ds,'method':method,'r35':stats(low),'full':stats(full),
                'ratio_of_means_percent':100*statistics.mean(low)/statistics.mean(full),
                'paired_attainment_percent':stats([100*a/b for a,b in zip(low,full)]),
                'paired_gap_pp':stats([100*(a-b) for a,b in zip(low,full)])})
        for baseline in ['Random','Entropy']:
            delta=[a-b for a,b in zip(bymethod['DINO_F518'],bymethod[baseline])]
            observed=abs(statistics.mean(delta))
            p=sum(abs(statistics.mean(a*s for a,s in zip(delta,signs)))>=observed-1e-15 for signs in itertools.product([-1,1],repeat=3))/8
            comparisons.append({'dataset':ds,'comparison':'DINO_F518-'+baseline,**stats(delta),'exact_two_sided_sign_flip_p':p})
    ordered=sorted(range(len(comparisons)),key=lambda i:comparisons[i]['exact_two_sided_sign_flip_p'])
    previous=0
    for rank,i in enumerate(ordered):
        previous=max(previous,min(1,(4-rank)*comparisons[i]['exact_two_sided_sign_flip_p']))
        comparisons[i]['holm_4_primary']=previous
    return {'budget_statistics':curves,'naulc':auc,'label_efficiency':eff,'paired_comparisons':comparisons,
            'repeat_unit':'3 paired acquisition+learner seed configurations; sample SD ddof=1',
            'warning':'100% reference is outside nAULC. n=3 cannot establish p<.05 with a two-sided sign-flip test.'}

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    result=summarize(list(csv.DictReader(a.input.open(encoding='utf-8-sig'))))
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
