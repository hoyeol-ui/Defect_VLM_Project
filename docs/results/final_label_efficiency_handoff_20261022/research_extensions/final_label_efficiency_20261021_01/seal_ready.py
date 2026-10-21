"""Seal the concrete local preparation after technical checks; starts no training."""
from pathlib import Path
import sys,json,csv,hashlib,importlib.metadata as meta
P=Path(__file__).resolve().parent;B=P.parents[1]
sys.path.insert(0,str(P))
from audit import read,sha,put,table

def main():
    assert read(P/'evidence/cpu_checks.json')['status']=='PASS'
    gc=sorted((P/'smoke').glob('saved_checkpoint_check_*/RESULT.json'))[-1]
    neu=sorted((P/'smoke').glob('NEU-DET_*/RESULT.json'))[-1]
    checks={}
    for label,path in [('GC10-DET',gc),('NEU-DET',neu)]:
        r=read(path);assert r['status']=='PASS' and r['technical_only'] and not r['test_evaluated'] and r['optimizer_updates']>0
        checks[label]={'result':str(path),'sha':sha(path),'updates':r['optimizer_updates'],'initial_state_sha':r['initial_state_sha']}
    full=read(P/'evidence/full_reuse_check.json')
    neu_real=read(P/'evidence/neu_trained_entropy_check.json')
    assert neu_real['status']=='PASS' and sum(r['detections'] for r in neu_real['images'])>0
    initial=read(B/full['source']/'training_backend.json')['initial_model_state_sha']
    assert checks['GC10-DET']['initial_state_sha']==initial
    # Record installed numerical sources, including local compatibility changes; version numbers alone are insufficient.
    import ultralytics
    runtime=list(Path(ultralytics.__file__).parent.rglob('*.py'))
    runtime += [B/'research_extensions/full_data_26s_sgd_cosine_640_20261018_01/windows_job.py',
                B/'research_extensions/full_data_26s_sgd_cosine_640_20261018_01/assets/yolo26s.pt']
    files=list(P.glob('*.py'))+list((P/'src').glob('*.py'))+list((P/'config').glob('*.json'))+list((P/'inputs').glob('*/index.json'))+list((P/'selections').glob('*/*/*/R*.json'))
    put(P/'evidence/execution_freeze.json',{'files':{f.relative_to(P).as_posix():sha(f) for f in sorted(files)},
        'runtime_files':{str(f):sha(f) for f in sorted(runtime)},
        'versions':{v:meta.version(v) for v in ['torch','torchvision','ultralytics','numpy','PyYAML','Pillow']}})
    matrix=list(csv.DictReader((P/'execution_matrix.csv').open(encoding='utf-8-sig')))
    assert len(matrix)==102 and sum(r['status']=='NEW_TRAINING_REQUIRED' for r in matrix)==101
    template=[]
    for r in matrix:
        methods=['Random','DINO_F518','Entropy'] if r['method']=='SHARED_R0' else [r['method']]
        for method in methods:
            template.append({k:v for k,v in dict(dataset=r['dataset'],pair=r['pair'],method=method,
                budget_percent=r['budget_percent'],train_n=r['train_n'],mAP50_95='',mAP50='',precision='',recall='',
                partition='test',technical_only='false').items()})
    assert len(template)==114
    table(P/'results_input_template.csv',template)
    readiness={'status':'READY_LOCAL_TRAIN_DEV','date_KST':'2026-10-21','smoke_checks':checks,
        'GC10_initial_state_matches_reused_full':True,'scientific_cells':102,'reused_training':1,'new_training_required':101,
        'new_scientific_trainings_started':0,'test_inference_this_task':0,
        'scope':'Local fixed-runtime per-cell train/dev runner and Entropy acquisition prepared; one-epoch technical checks passed. Entire 100-epoch campaign and final evaluation are pending.',
        'additional_nc6_nonempty_entropy_check':'evidence/neu_trained_entropy_check.json; existing trained YOLO26n, same nc6 one2one format; final s model size unchanged',
        'preserved_failures':['GC10 64-image one-epoch smoke: 0 optimizer updates; retained',
                              'GC10 128-image smoke trained/saved; initial post-training adapter geometry error corrected and checked without retraining'],
        'limitations':['No remote environment validation','No new independent test split; historical exposure disclosed',
                       'No complete scientific 100-epoch run of the new campaign executed in this task',
                       'Final-size NEU26s smoke had no retained candidates on two scoring images; nc6 nonempty scoring separately passed on an existing trained YOLO26n. No full scientific NEU26s acquisition executed yet.']}
    put(P/'evidence/readiness.json',readiness)
    from integrity import verify
    verify()
    print(json.dumps({'status':readiness['status'],'new_trainings':101,'technical_updates':{k:v['updates'] for k,v in checks.items()},
                      'frozen_files':len(files),'runtime_files':len(runtime),'template_rows':len(template)},indent=2))

if __name__=='__main__':main()
