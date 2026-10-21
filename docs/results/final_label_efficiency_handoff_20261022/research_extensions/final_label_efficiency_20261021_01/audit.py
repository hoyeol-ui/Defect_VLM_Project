"""Read saved evidence; write only this campaign and the requested new docs tables."""
from pathlib import Path
import csv, json, hashlib, collections, statistics, subprocess, sys

P = Path(__file__).resolve().parent
B = P.parents[1]
D = B / 'docs'
FULL = B / 'research_extensions/full_data_26s_sgd_cosine_640_20261018_01'
HIST = B / 'output/final_main_results_retrieval_20260922/metadata_verified/final_main_v2_20260922'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda: f.read(1048576), b''):
            h.update(v)
    return h.hexdigest()

def put(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding='utf-8')

def table(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def rel(p):
    return Path(p).relative_to(B).as_posix()

def splits():
    out = {}
    for ds in ('GC10-DET', 'NEU-DET'):
        base = B / 'output/dataset_phase1b_v2_20260919' / ds
        groups = {part: (base / f'{part}_sample_ids.txt').read_text().splitlines()
                  for part in ('train', 'dev', 'test')}
        records = list(csv.DictReader((base / 'SPLIT_MEMBERSHIP.csv').open(encoding='utf-8-sig')))
        comps = collections.defaultdict(set)
        for r in records:
            comps[r['component_id']].add(r['split'])
        index = list(csv.DictReader((base / 'CANONICAL_INDEX.csv').open(encoding='utf-8-sig')))
        checks = {
            'unique_ids': all(len(x) == len(set(x)) for x in groups.values()),
            'no_id_overlap': not any(set(groups[a]) & set(groups[b]) for a, b in [('train','dev'),('train','test'),('dev','test')]),
            'component_disjoint': all(len(v) == 1 for v in comps.values()),
            'membership_matches_lists': all(set(groups[k]) == {r['sample_id'] for r in records if r['split']==k} for k in groups),
        }
        assert all(checks.values()), checks
        out[ds] = {'counts': {k: len(v) for k,v in groups.items()}, 'checks': checks,
                   'split_sha': sha(base/'SPLIT_MEMBERSHIP.csv'),
                   'canonical_rows': len(index), 'eligibility_counts': dict(collections.Counter(r['eligibility'] for r in index)),
                   'raw_alias_count': sum(int(r['raw_alias_count']) for r in index),
                   'split_seed': sorted({r['split_seed'] for r in records})}
    put(P/'evidence/split_recheck.json', out)
    return out

def inventory():
    rows=[]
    # Do not recursively traverse copied images, vendored libraries or duplicate handoff evidence.
    paths = sorted((B/'research_extensions').glob('*/runs/*')) + sorted((B/'research_extensions').glob('*/smoke/*'))
    for run in paths:
        if not run.is_dir() or run.is_relative_to(P):
            continue
        r = read(run/'RESULT.json') if (run/'RESULT.json').exists() else {}
        args = read(run/'resolved_args.json') if (run/'resolved_args.json').exists() else {}
        request = read(run/'REQUEST.json') if (run/'REQUEST.json').exists() else {}
        metric = read(run/'sealed_dev_metrics.json') if (run/'sealed_dev_metrics.json').exists() else {}
        best = read(run/'best_selection.json') if (run/'best_selection.json').exists() else {}
        env = read(run/'ENVIRONMENT.json') if (run/'ENVIRONMENT.json').exists() else {}
        model = read(run/'MODEL_ACTUAL.json') if (run/'MODEL_ACTUAL.json').exists() else {}
        if not (r or args or (run/'FAILURE.json').exists()):
            continue
        train_ids = read(run/'loader_ids.json') if (run/'loader_ids.json').exists() else {}
        n = r.get('train_n', len(train_ids.get('train',[])))
        ds = r.get('dataset', request.get('dataset','확인 불가'))
        pool = {'GC10-DET':1604,'NEU-DET':1258}.get(ds)
        checkpoints = {}
        for kind in ('best','last'):
            path=run/f'train/weights/{kind}.pt'
            checkpoints[kind] = sha(path) if path.exists() else None
        source_bad=[]
        for name,h in r.get('source_hashes',{}).items():
            path=run.parents[1]/'src'/name
            if not path.exists() or sha(path)!=h:
                source_bad.append(name)
        overall=metric.get('overall',{})
        rows.append(dict(campaign=run.parents[1].name,run_id=run.name,dataset=ds,
            model=r.get('model',request.get('model',Path(args.get('model','unknown')).stem)),
            phase=r.get('phase','failed_or_incomplete'),status=r.get('status','FAILED' if (run/'FAILURE.json').exists() else 'INCOMPLETE'),
            train_n=n,pool_n=pool,actual_train_fraction=n/pool if pool and n else '',
            dev_n=r.get('dev_n',len(train_ids.get('dev',[]))),epochs_requested=args.get('epochs',request.get('epochs','')),
            epochs_completed=r.get('epochs',''),best_epoch=best.get('epoch',''),
            optimizer=args.get('optimizer',''),cos_lr=args.get('cos_lr',''),close_mosaic=args.get('close_mosaic',''),
            lr0=args.get('lr0',''),lrf=args.get('lrf',''),batch=args.get('batch',''),imgsz=args.get('imgsz',''),
            learner_seed=args.get('seed',''),acquisition_seed=request.get('acquisition_seed',''),
            mAP50=overall.get('metrics/mAP50(B)',''),mAP50_95=overall.get('metrics/mAP50-95(B)',''),
            precision=overall.get('metrics/precision(B)',''),recall=overall.get('metrics/recall(B)',''),
            metric_partition=metric.get('partition','unknown'),technical_only=metric.get('technical_only',r.get('phase')=='smoke'),
            total_seconds=r.get('total_seconds',''),gpu=env.get('gpu',''),torch=env.get('torch',''),ultralytics=env.get('ultralytics',''),
            end2end=model.get('end2end',''),best_exists=checkpoints['best'] is not None,
            best_sha_matches=bool(checkpoints['best']) and checkpoints['best']==r.get('best_sha')==best.get('best_sha'),
            last_sha_matches=bool(checkpoints['last']) and checkpoints['last']==r.get('last_sha')==best.get('last_sha'),
            source_hash_mismatches=';'.join(source_bad),evidence=rel(run),
            result_sha=sha(run/'RESULT.json') if r else '',metric_sha=sha(run/'sealed_dev_metrics.json') if metric else ''))
    table(D/'20261021_experiment_inventory.csv',rows)
    put(P/'evidence/inventory_summary.json', {'rows':len(rows),'status':dict(collections.Counter((r['phase']+':'+r['status']) for r in rows))})
    return rows

def reuse():
    run=next((FULL/'runs').iterdir());r=read(run/'RESULT.json');args=read(run/'resolved_args.json')
    ids=read(run/'loader_ids.json');ds=FULL/'datasets/GC10-DET'
    split=B/'output/dataset_phase1b_v2_20260919/GC10-DET'
    expected={p:set((split/f'{p}_sample_ids.txt').read_text().splitlines()) for p in ('train','dev','test')}
    checks={
        'full_train_membership':set(ids['train'])==expected['train'],
        'development_membership':set(ids['dev'])==expected['dev'],
        'no_test_ids_in_train_or_dev':not (set(ids['train']+ids['dev'])&expected['test']),
        'all_100_epochs':r['epochs']==len(read(run/'epoch_metrics.json'))==100,
        'best_sha':sha(run/'train/weights/best.pt')==r['best_sha']==read(run/'best_selection.json')['best_sha'],
        'last_sha':sha(run/'train/weights/last.pt')==r['last_sha'],
        'pretrained_sha':sha(FULL/'assets/yolo26s.pt')==read(run/'REQUEST.json')['weight_sha'],
        'fixed_settings':all(args[k]==v for k,v in {'epochs':100,'optimizer':'SGD','cos_lr':True,'close_mosaic':10,'imgsz':640,'batch':8,'seed':2026091900,'lr0':.01,'lrf':.01}.items()),
        'test_not_evaluated_in_this_run':r['test_evaluated'] is False,
    }
    seal=read(FULL/'evidence/INPUT_SEAL.json')
    bad=[name for name,h in seal['files'].items() if name.startswith('datasets/GC10-DET/') and (not (FULL/name).exists() or sha(FULL/name)!=h)]
    view_bad=[]
    for part in ('train','dev'):
        for typ in ('images','labels'):
            for source in (ds/typ/part).iterdir():
                target=run/'view'/typ/part/source.name
                if not target.exists() or sha(target)!=sha(source):view_bad.append(rel(target))
    source_bad=[n for n,h in r['source_hashes'].items() if sha(FULL/'src'/n)!=h]
    checks.update(input_seal_bytes=not bad,actual_loader_view_bytes=not view_bad,source_hashes=not source_bad,
                  config_sha=sha(FULL/'config/common_args.json')==r['common_args_sha'])
    assert all(checks.values()),checks
    evidence={'decision':'REUSE_TRAINED_GC10_FULL_SEED_2026091900','checks':checks,'input_files_checked':sum(n.startswith('datasets/GC10-DET/') for n in seal['files']),
              'source':rel(run),'best_sha':r['best_sha'],'remaining_required':'same frozen final evaluation as AL, after all model/settings are locked',
              'interpretation':'Saved dev result is not the final test score. Reuse valid only with identical local runtime/numerical recipe/head.'}
    put(P/'evidence/full_reuse_check.json',evidence)
    config=read(FULL/'config/common_args.json')
    put(P/'config/common_args.json',config)
    return run

def historical():
    # Saved final-test metrics only. No inference or test image access.
    base=B/'final_test_evaluation_20260925/statistics'
    curve=list(csv.DictReader((base/'curves.csv').open(encoding='utf-8-sig')))
    refs=list(csv.DictReader((base/'R5_vs_full_data.csv').open(encoding='utf-8-sig')))
    output=[]
    for v in refs:
        a=float(v['r5_mean_mAP50_95']);f=float(v['full_data_mAP50_95'])
        output.append({**v,'attainment_percent_ratio_of_means':100*a/f,'note':'YOLOv8n F518 TEST; full reference n=1, no full SD; NOT YOLO26'})
    table(D/'20261021_historical_label_efficiency.csv',output)
    raw=list((B/'final_test_evaluation_20260925/raw/runs').glob('*/result.json'))
    put(P/'evidence/historical_read_check.json',{'raw_results_count':len(raw),'curve_rows':len(curve),
        'tables_sha':{p.name:sha(p) for p in [base/'curves.csv',base/'R5_vs_full_data.csv',base/'paired_comparisons.csv']},
        'raw_results_readable':all(isinstance(read(p),dict) for p in raw),
        'scope':'Recomputed ratios from saved table; historical 20261017 full 162-result audit is cited separately, not rerun here.'})

def matrix(run):
    # Three paired acquisition/learner configurations; no crossed-factor claim.
    pairs=[(2026091901,2026091900),(2026091902,2026091908),(2026091903,2026091909)]
    rows=[]
    counts={'GC10-DET':[160,241,321,401,481,561],'NEU-DET':[126,189,252,315,377,440]}
    for ds in counts:
        for i,(acq,learn) in enumerate(pairs):
            for k,n in enumerate(counts[ds]):
                for method in (['SHARED_R0'] if k==0 else ['Random','DINO_F518','Entropy']):
                    rows.append(dict(dataset=ds,pair=i+1,acquisition_seed=acq,learner_seed=learn,method=method,
                        budget_percent=[10,15,20,25,30,35][k],train_n=n,round=k,status='NEW_TRAINING_REQUIRED',reuse_path=''))
            reused=ds=='GC10-DET' and i==0
            rows.append(dict(dataset=ds,pair=i+1,acquisition_seed='',learner_seed=learn,method='FullySupervised',budget_percent=100,
                train_n=1604 if ds=='GC10-DET' else 1258,round='',status='REUSE_TRAINED_CHECKPOINT' if reused else 'NEW_TRAINING_REQUIRED',reuse_path=rel(run) if reused else ''))
    table(P/'execution_matrix.csv',rows)
    put(P/'config/protocol.json',{'date_KST':'2026-10-21','status':'PROTOCOL_FIXED_EXECUTION_GATES_SEPARATE','model':'yolo26s',
        'pretrained_sha':'646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b',
        'budgets_percent':[10,15,20,25,30,35],'counts':counts,'seed_pairs':[{'acquisition':a,'learner':l} for a,l in pairs],
        'repeat_unit':'paired acquisition+learner seed configuration; effects cannot be separated',
        'full_learner_seeds':[l for a,l in pairs],'physical_scientific_cells':len(rows),'new_trainings':sum(r['status']=='NEW_TRAINING_REQUIRED' for r in rows),
        'runtime':{'python':'3.11.9','torch':'2.12.1+cu126','torchvision':'0.27.1+cu126','ultralytics':'8.4.90','gpu':'RTX 3070'},
        'selection':'reuse verified detector-independent Random and DINO manifests; recompute all Entropy using preceding YOLO26 checkpoint',
        'evaluation_head':'one2one/end2end=True in installed 8.4.90; no current-docs default substitution',
        'entropy_head':'one2one raw class scores before top-k; xyxy converted to xywh for unchanged ENTROPY_NCS_MEAN_V2 stable-NMS score function',
        'test_access':'No test inference during preparation; final settings frozen before fixed evaluation; historical test exposure disclosed'})

def main():
    P.mkdir(exist_ok=True);D.mkdir(exist_ok=True)
    splits();rows=inventory();run=reuse();historical();matrix(run)
    summary={'recent_runs':len(rows),'scientific_complete':[{'dataset':r['dataset'],'model':r['model'],'n':r['train_n'],'map':r['mAP50_95'],'hours':round(r['total_seconds']/3600,3),'run':r['run_id']} for r in rows if r['phase']=='train' and r['status']=='PASS'],
             'output':str(P)}
    put(P/'evidence/audit_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
