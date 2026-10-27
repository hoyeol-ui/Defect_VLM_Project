"""Utilities for the new170 campaign; no ML imports or server mutations."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]
E=ROOT/'evidence'
P=ROOT/'server_snapshot/runpod_200e_5seed_blackwell_entropy04'
PYTHON=BASE/'Defect_VLM_Project/.python311/python.exe'
O=BASE/'docs/method_closure_20261022'
GPU_LOCK=BASE/'research_extensions/GPU_TRAINING_ACTIVE.lock'
MUTEX=ROOT/'FINAL_TEST_ACTIVE.lock'
RUN=ROOT/'test_execution'
DATASETS=('GC10-DET','NEU-DET')
METHODS=('Random','DINOv2_S14_F518_FF','Entropy_NCS_MEAN_V2')

def require(ok,message):
    if not ok:raise RuntimeError(message)
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check_hash(p,expected):require(sha(p)==expected,'SHA mismatch: '+str(p))
def put(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def verify_seal():
    seal=read(E/'TEST_PACKAGE_LOCK.json')
    for p,h in seal['files'].items():check_hash(p,h)
    return seal
def verify_inputs():
    inputs=read(E/'TEST_INPUTS.json')
    for ds,d in inputs['datasets'].items():
        require(ds in DATASETS and len(d['records'])==d['n'],'Input cardinality')
        for r in d['records']:
            check_hash(r['image'],r['image_sha']);check_hash(r['label'],r['label_sha'])
        for kind,key in [('images','image'),('labels','label')]:
            folder=ROOT/'test_inputs'/ds/kind/'test'
            require({str(p.resolve()) for p in folder.iterdir()}=={str(Path(r[key]).resolve()) for r in d['records']},'Input directory membership changed')
    return inputs
def verify_models():
    models=read(E/'TEST_MODELS.json')['models']
    require(len(models)==len({m['cell_id'] for m in models})==170,'170 physical models required')
    require(len({m['best_sha'] for m in models})==170,'Duplicate physical best checkpoints')
    audit=read(P/'state/MODEL_HASH_AUDIT.json')
    require(audit['checked']==170 and audit['status']=='MODELS_170_ARTIFACTS_VERIFIED','Server final artifact audit incomplete')
    for i,m in enumerate(models,1):
        for key,hkey in [('best_path','best_sha'),('last_path','last_sha'),('complete_path','complete_sha'),('loader_path','loader_sha')]:check_hash(m[key],m[hkey])
        if i%25==0 or i==170:print('LOCAL_ARTIFACTS_VERIFIED',i,'/170',flush=True)
    return models
