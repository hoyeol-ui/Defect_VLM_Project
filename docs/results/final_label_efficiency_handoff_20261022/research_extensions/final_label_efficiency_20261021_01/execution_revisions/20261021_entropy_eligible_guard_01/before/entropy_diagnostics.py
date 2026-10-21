"""Describe frozen exact-score/ID tie behavior without changing the acquisition rule."""
from collections import Counter
import math,statistics

def describe(ranking,chosen,q):
    assert ranking and 0<q<=len(ranking)
    assert ranking==sorted(ranking,key=lambda r:(-r['uncertainty'],r['id']))
    ids=[r['id'] for r in ranking]
    assert len(set(ids))==len(ids) and chosen==ids[:q]
    assert all(math.isfinite(r['uncertainty']) and 0<=r['uncertainty']<=1 for r in ranking)
    frequencies=Counter(r['uncertainty'] for r in ranking)
    n=len(ranking);boundary=ranking[q-1]['uncertainty']
    group=[r['id'] for r in ranking if r['uncertainty']==boundary]
    above=sum(r['uncertainty']>boundary for r in ranking);needed=q-above
    selected_group=[r for r in chosen if r in set(group)]
    assert selected_group==sorted(group)[:needed]
    positions={id:i for i,id in enumerate(sorted(ids))}
    def deciles(values):
        result=[0]*10
        for id in values:result[min(9,positions[id]*10//n)]+=1
        return result
    selected_set=set(chosen);selected_so_far=0;rank_cdf_difference=0.
    for i,id in enumerate(sorted(ids)):
        selected_so_far+=int(id in selected_set)
        rank_cdf_difference=max(rank_cdf_difference,abs(selected_so_far/q-(i+1)/n))
    no_pred=sum(r['detections']==0 for r in ranking)
    tie_cut=len(group)>needed
    result={'eligible_n':n,'query_n':q,'no_prediction_n':no_pred,'no_prediction_fraction':no_pred/n,
        'zero_uncertainty_n':frequencies.get(0.,0),'unique_exact_scores':len(frequencies),
        'candidates_in_exact_ties_n':sum(v for v in frequencies.values() if v>1),
        'candidates_in_exact_ties_fraction':sum(v for v in frequencies.values() if v>1)/n,
        'largest_exact_tie_n':max(frequencies.values()),'cutoff_score':boundary,
        'cutoff_tie_n':len(group),'cutoff_tie_selected_n':needed,'cutoff_tie_truncated_by_ID':tie_cut,
        'cutoff_tie_choice_matches_frozen_ID_rule':True,
        'selected_by_cutoff_ID_tie_fraction':needed/q if tie_cut else 0.,
        'eligible_ID_decile_counts':deciles(ids),'selected_ID_decile_counts':deciles(chosen),
        'selected_mean_global_ID_percentile':statistics.mean((positions[id]+.5)/n for id in chosen),
        'ID_rank_descriptive_KS_D':rank_cdf_difference,
        'semantic_bias_claim':'Opaque hash-ID ordering can favor low IDs within a boundary tie. ID ranks do not establish class/source bias; no candidate labels were read.',
        'algorithm_modified':False,
        'gate':'DEGENERATE_ALL_SCORES_TIED' if len(frequencies)==1 and q<n else 'PASS'}
    return result
