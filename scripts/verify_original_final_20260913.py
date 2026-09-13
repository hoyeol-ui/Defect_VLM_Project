"""Verify the published original-final tables without model/data access.

Uses only the standard library. Reconciles historical learner observations and
the 200 Entropy rows to A/B means, curves, nAULC, Full means, ratios and the
four-contrast exploratory exact sign-flip/Holm family. Bootstrap intervals are
preserved as published; this verifier does not regenerate their random draws.
"""
import csv
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'results/original_split_exploratory_completion_20260913'


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def close(actual, expected):
    assert math.isclose(float(actual), float(expected), abs_tol=1e-12, rel_tol=1e-10), (actual, expected)


def indexed(rows, keys):
    result = {tuple(row[k] for k in keys): row for row in rows}
    assert len(result) == len(rows), f'Duplicate keys: {keys}'
    return result


def main():
    manifest = json.loads((ROOT / 'docs/ORIGINAL_FINAL_PUBLIC_FILES_20260913.json').read_text())
    for path, expected in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, path
    expected_counts = {
        'three_way_original_final_summary.csv': 6,
        'original_final_naulc.csv': 60,
        'full_data_original_final_results.csv': 6,
        'original_final_budget_curves.csv': 36,
        'R5_vs_Full_descriptive_ratios.csv': 6,
        'original_final_statistics.csv': 4,
        'original_final_round_AB_means_360.csv': 360,
        'original_final_paired_deltas.csv': 40,
        'entropy_original_final_results.csv': 200,
        'per_class_original_final_results.csv': 1632,
    }
    data = {name: read(TABLES/name) for name in expected_counts}
    for name, rows in data.items():
        assert len(rows) == expected_counts[name], name
        assert all(r['analysis_scope']=='ORIGINAL_SPLIT_EXPLORATORY_COMPLETION' for r in rows)
        assert all(r['primary_thesis_evidence'].lower()=='false' for r in rows)

    # Independently reconcile the public individual learner observations.
    metrics = ['map50_95', 'map50', 'precision', 'recall']
    observations = defaultdict(list)
    historical = read(ROOT/'results/thesis_final/tables/locked_final_conceptual_480.csv')
    assert len(historical) == 480
    assert len({r['cell_id'] for r in historical}) == 440
    for row in historical:
        method = {'FrozenDINO-FF-v1':'DINO', 'Random-PCG64-v1':'Random'}[row['strategy']]
        seed = f"a{int(row['acquisition_index']):02d}"
        observations[row['dataset'], method, seed, row['round']].append(row)
        if row['round']=='0' and method=='Random':
            observations[row['dataset'], 'Entropy', seed, '0'].append(row)
    entropy = data['entropy_original_final_results.csv']
    assert len({r['cell_id'] for r in entropy}) == 200
    for row in entropy:
        assert row['round'] in {'1','2','3','4','5'}
        observations[row['dataset'], 'Entropy', row['seed'], row['round']].append(row)
    means = indexed(data['original_final_round_AB_means_360.csv'], ['dataset','method','seed','round'])
    assert set(means) == set(observations)
    for key, group in observations.items():
        assert len(group)==2
        assert len({r.get('learner',r.get('training_seed_slot')) for r in group})==2
        for metric in metrics:
            close(statistics.mean(float(r[metric]) for r in group), means[key][metric])

    seedwise = indexed(data['original_final_naulc.csv'], ['dataset','method','seed'])
    for key, row in seedwise.items():
        curve = [float(means[key+(str(i),)]['map50_95']) for i in range(6)]
        close(sum(w*v for w,v in zip([.1,.2,.2,.2,.2,.1],curve)), row['nAULC_map50_95'])
        close(curve[-1],row['R5_map50_95'])
    summaries = indexed(data['three_way_original_final_summary.csv'], ['dataset','method'])
    for key, row in summaries.items():
        group = [r for k,r in seedwise.items() if k[:2]==key]
        assert len(group)==10
        vals = [float(r['nAULC_map50_95']) for r in group]
        close(statistics.mean(vals),row['mean_nAULC_map50_95'])
        close(statistics.stdev(vals),row['sd_nAULC_map50_95'])
        close(statistics.mean(float(r['R5_map50_95']) for r in group),row['R5_mean_map50_95'])
    for row in data['original_final_budget_curves.csv']:
        vals=[float(r['map50_95']) for k,r in means.items() if k[:2]==(row['dataset'],row['method']) and k[3]==row['round']]
        assert len(vals)==10
        close(statistics.mean(vals),row['mean_map50_95'])
        close(statistics.stdev(vals),row['sd_acquisition_seed_map50_95'])
    full = indexed(data['full_data_original_final_results.csv'], ['dataset','learner'])
    assert sum(r['record_type']=='OBSERVED_LEARNER' for r in full.values())==4
    for ds in ['GC10-DET','NEU-DET']:
        assert full[ds,'mean']['n_learners']=='2'
        for metric in metrics:
            close(statistics.mean(float(full[ds,l][metric]) for l in ['A','B']),full[ds,'mean'][metric])
    for row in data['R5_vs_Full_descriptive_ratios.csv']:
        ds,method=row['dataset'],row['method']
        close(float(summaries[ds,method]['R5_mean_map50_95'])/float(full[ds,'mean']['map50_95']),row['R5_over_Full_descriptive_ratio'])
    deltas = indexed(data['original_final_paired_deltas.csv'], ['dataset','contrast','seed'])
    tests = data['original_final_statistics.csv']
    raw=[]
    for row in tests:
        ds=row['dataset']; positive,negative=row['contrast'].split(' - ')
        vals=[]
        for i in range(10):
            seed=f'a{i:02d}'
            delta=float(seedwise[ds,positive,seed]['nAULC_map50_95'])-float(seedwise[ds,negative,seed]['nAULC_map50_95'])
            close(delta,deltas[ds,row['contrast'],seed]['delta_nAULC'])
            vals.append(delta)
        close(statistics.mean(vals),row['mean_difference'])
        observed=abs(statistics.mean(vals))
        extreme=sum(abs(sum(s*v for s,v in zip(signs,vals))/10)>=observed-1e-12 for signs in itertools.product([-1,1],repeat=10))
        p=extreme/1024
        close(p,row['exact_sign_flip_p_raw']); raw.append(p)
    adjusted=[0]*4; previous=0
    for rank,i in enumerate(sorted(range(4),key=lambda i:raw[i])):
        previous=max(previous,min(1,(4-rank)*raw[i])); adjusted[i]=previous
    for row,p in zip(tests,adjusted):close(p,row['holm_adjusted_p'])
    assert sum(p<.05 for p in adjusted)==0
    print('PASS: original-final public hashes, 204 new observations (200 Entropy + 4 Full),')
    print('historical 440 reuse, A/B means, 60 nAULC, R5, Full, ratios and exact p/Holm.')
    print('No training, inference, checkpoint or raw dataset access. Bootstrap CI not regenerated.')


if __name__ == '__main__':
    main()
