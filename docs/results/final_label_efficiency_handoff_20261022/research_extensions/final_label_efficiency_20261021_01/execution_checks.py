"""CPU checks for acquisition diagnostics, campaign ordering and empirical ETA."""
from pathlib import Path
import sys,json
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from entropy_diagnostics import describe
from campaign_lib import load_matrix,ordered_new,cell_id,eta,prior_seconds

def main():
    ranking=[dict(id='z',uncertainty=.9,detections=2),dict(id='a',uncertainty=.8,detections=1),
        dict(id='b',uncertainty=.8,detections=1),dict(id='c',uncertainty=0.,detections=0),dict(id='d',uncertainty=0.,detections=0)]
    r=describe(ranking,['z','a'],2)
    assert r['gate']=='PASS' and r['no_prediction_fraction']==.4
    assert r['candidates_in_exact_ties_fraction']==.8 and r['cutoff_tie_truncated_by_ID']
    assert r['cutoff_tie_selected_n']==1 and r['selected_by_cutoff_ID_tie_fraction']==.5
    allzero=[dict(id=str(i),uncertainty=0.,detections=0) for i in range(4)]
    assert describe(allzero,['0','1'],2)['gate']=='DEGENERATE_ALL_SCORES_TIED'
    assert describe(allzero,['0','1','2','3'],4)['ID_rank_descriptive_KS_D']==0
    assert describe(allzero,['0','1'],2)['ID_rank_descriptive_KS_D']==.5
    try:describe(ranking,['z','b'],2)
    except AssertionError:pass
    else:raise AssertionError('Wrong ID tie choice accepted')
    matrix=load_matrix();order=ordered_new(matrix)
    assert order[0]['dataset']=='NEU-DET' and order[0]['method']=='FullySupervised'
    assert [(r['dataset'],r['method']) for r in order[1:3]]==[('GC10-DET','SHARED_R0'),('NEU-DET','SHARED_R0')]
    assert all(r['method']=='SHARED_R0' for r in order[1:7])
    assert len(matrix)==102 and len(order)==101 and len(set(map(cell_id,order)))==101
    assert all(r['budget_percent'] in ['10','15','20','25','30','35','100'] for r in matrix)
    ledger=[dict(r,cell_id=cell_id(r),state='PENDING',training_seconds='') for r in order]
    original=eta(ledger)['remaining_training_hours']
    assert abs(original-44.273739838510025)<1e-10
    ledger[0].update(state='SUCCEEDED',training_seconds=2*prior_seconds(ledger[0]))
    updated=eta(ledger)
    assert updated['dataset_measured_multipliers']['NEU-DET']==2
    assert updated['dataset_measured_multipliers']['GC10-DET']==1
    assert updated['measured_complete_cells']==1
    out={'status':'PASS','checks':['fixed 101-cell order and six shared10% priority','empty prediction and exact-score tie rates',
        'cutoff ID tie audit and all-tied fail-stop','new empirical ETA calibrates only the measured dataset'],
        'synthetic_values_not_research_results':True}
    (P/'evidence/execution_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
