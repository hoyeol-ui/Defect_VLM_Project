"""Verify public development CSVs without private data, models or network."""
from pathlib import Path
import csv,json,math,statistics,hashlib,sys,itertools,subprocess
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/development_20260913';T=R/'tables'
def read(name):
    with (T/(name+'.csv')).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b):
    if not math.isclose(float(a),float(b),abs_tol=1e-11,rel_tol=0):raise AssertionError((a,b))
physical=read('complete_development_runs');concept=read('conceptual_development_runs');pairs=read('seed_budget_AB_means');curves=read('seedwise_nAULC');summary=read('paired_nAULC_comparison')
assert len(physical)==644 and len({r['node_id'] for r in physical})==644
assert all(r['evaluation_split']=='development_validation' and int(r['epochs'])==100 for r in physical)
assert sum(r['origin']=='historical_440' for r in physical)==440
assert sum(r['method']=='Entropy' for r in physical)==200 and sum(r['method']=='Full' for r in physical)==4
DS=['GC10-DET','NEU-DET'];M=['Random','DINO','Entropy'];seeds=[f'a{x:02d}' for x in range(10)]
expected=set(itertools.product(DS,M,seeds,['A','B'],range(6)))
key=lambda r:(r['dataset'],r['method'],r['seed'],r['learner'],int(r['round']))
cells={key(r):r for r in concept};assert len(concept)==len(cells)==720 and set(cells)==expected
for ds,se,slot in itertools.product(DS,seeds,['A','B']):
    rr=[cells[ds,m,se,slot,0] for m in M]
    assert len({r['node_id'] for r in rr})==len({r['map50_95'] for r in rr})==1
budgets={'GC10-DET':[161,241,321,401,481,561],'NEU-DET':[126,189,252,315,378,441]}
means={}
for ds,m,se,rnd in itertools.product(DS,M,seeds,range(6)):
    rows=[cells[ds,m,se,slot,rnd] for slot in ['A','B']]
    assert all(int(r['label_budget'])==budgets[ds][rnd] for r in rows)
    means[ds,m,se,rnd]=statistics.mean(float(r['map50_95']) for r in rows)
assert len(pairs)==360
for r in pairs:close(r['map50_95'],means[r['dataset'],r['method'],r['seed'],int(r['round'])])
auc={}
for ds,m,se in itertools.product(DS,M,seeds):
    y=[means[ds,m,se,k] for k in range(6)];auc[ds,m,se]=(y[0]+2*sum(y[1:5])+y[5])/10
assert len(curves)==60 and len({(r['dataset'],r['method'],r['seed']) for r in curves})==60
for r in curves:close(r['nAULC'],auc[r['dataset'],r['method'],r['seed']])
for r in read('method_budget_summary'):
    v=[means[r['dataset'],r['method'],se,int(r['round'])] for se in seeds]
    close(r['mean'],statistics.mean(v));close(r['median'],statistics.median(v));close(r['std'],statistics.stdev(v));assert int(r['count'])==10
assert len(summary)==6
for r in summary:
    aa,bb=r['comparison'].split(' - ');v=[100*(auc[r['dataset'],aa,se]-auc[r['dataset'],bb,se]) for se in seeds]
    close(r['mean_difference_pp'],statistics.mean(v));close(r['median_difference_pp'],statistics.median(v))
    assert int(r['positive_seeds'])==sum(x>0 for x in v) and int(r['negative_seeds'])==sum(x<0 for x in v)
    if '--bootstrap' in sys.argv:
        import numpy as np
        rng=np.random.Generator(np.random.PCG64(int(r['bootstrap_seed'])))
        boots=np.array(v)[rng.integers(0,10,size=(100000,10))].mean(axis=1)
        lo,hi=np.percentile(boots,[2.5,97.5]);close(lo,r['ci95_low_pp']);close(hi,r['ci95_high_pp'])
for r in read('paired_nAULC_seed_effects'):
    aa,bb=r['comparison'].split(' - ');close(r['difference_pp'],100*(auc[r['dataset'],aa,r['seed']]-auc[r['dataset'],bb,r['seed']]))
full={}
for r in read('full_data_reference_summary'):
    vals=[float(x['map50_95']) for x in physical if x['dataset']==r['dataset'] and x['method']=='Full'];assert len(vals)==2
    full[r['dataset']]=statistics.mean(vals);close(full[r['dataset']],r['mean_map50_95']);assert int(r['learner_count'])==2
for r in read('r5_vs_full_data'):
    mean=statistics.mean(means[r['dataset'],r['method'],se,5] for se in seeds)
    close(r['R5_map50_95_pct'],100*mean);close(r['full_map50_95_pct'],100*full[r['dataset']]);close(r['R5_minus_full_pp'],100*(mean-full[r['dataset']]))
    close(r['percent_of_full_reference'],100*mean/full[r['dataset']]);assert int(r['full_learners'])==2
q=read('entropy_query_diagnostics');assert len(q)==100 and len({(r['dataset'],r['seed'],r['round']) for r in q})==100
completion=json.loads((R/'evidence/continuation_completion.json').read_text())
assert completion['entropy_complete']==completion['entropy_expected']==200 and completion['full_data_completed']==4 and completion['acquisitions_completed']==100
assert completion['locked_test_accesses']==completion['duplicate_executions']==0
manifest=json.loads((R/'audit/PUBLIC_FILE_MANIFEST.json').read_text())
# This dated release manifest also sealed the then-current navigation/verifier.
# Validate their original Git blobs instead of replacing the historical manifest
# or comparing its hashes against deliberately updated current documentation.
SNAPSHOT='077503ca6f68fb7c92420cddb4bf739439165b8a'
CURRENT_FILES={'README.md','THESIS_FILE_INDEX.md','scripts/verify_development_20260913.py'}
for r in manifest['files']:
    p=(ROOT/r['path']).resolve();assert ROOT in p.parents
    b=p.read_bytes()
    if r['path'] in CURRENT_FILES and hashlib.sha256(b).hexdigest()!=r['sha256']:
        try:
            b=subprocess.check_output(['git','show',f"{SNAPSHOT}:{r['path']}"],cwd=ROOT,stderr=subprocess.PIPE)
        except (OSError,subprocess.CalledProcessError) as exc:
            raise RuntimeError('Historical navigation hash verification needs a Git clone containing '+SNAPSHOT) from exc
    assert len(b)==r['size'] and hashlib.sha256(b).hexdigest()==r['sha256'],r['path']
for mapping in manifest['figure_sources']:
    assert all((R/p).is_file() for p in mapping['sources'])
    assert all((R/f"figures/{mapping['stem']}.{ext}").is_file() for ext in ['png','svg','pdf'])
assert len(manifest['figure_sources'])==5
print('DEVELOPMENT PUBLIC VERIFICATION: PASS (644 physical / 720 conceptual / 10 seeds / 100 queries / Full 4; '+('bootstrap 100000 x 6 PASS' if '--bootstrap' in sys.argv else 'use --bootstrap for optional CI recomputation')+')')
