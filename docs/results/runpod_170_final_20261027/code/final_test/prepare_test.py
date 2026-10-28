"""Prepare fixed inputs and evaluation plan, then seal after verified backup."""
import argparse
import importlib.metadata as md
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *

def inputs():
    require(not (E/'TEST_INPUTS.json').exists(),'Input preparation already exists')
    data=BASE/'research_extensions/runpod_final_200e_20261023_01/data'
    manifest={'created_utc':now(),'partition':'test','datasets':{},'inference_performed':False}
    for ds,n in [('GC10-DET',343),('NEU-DET',270)]:
        index=read(data/ds/'INDEX.json')
        check_hash(data/ds/'CANONICAL.json',index['canonical_sha'])
        require(index['group_cross_partition']==index['identities_cross_partition']==0,'Split overlaps')
        require(len(index['test'])==n and len({r['id'] for r in index['test']})==n,'Test cardinality')
        parts=[{r['id'] for r in index[k]} for k in ('train','dev','test')]
        require(not(parts[0]&parts[1] or parts[0]&parts[2] or parts[1]&parts[2]),'Test/train/dev ID overlap')
        out=ROOT/'test_inputs'/ds
        (out/'images/test').mkdir(parents=True,exist_ok=False);(out/'labels/test').mkdir(parents=True,exist_ok=False)
        records=[]
        for row in sorted(index['test'],key=lambda x:x['id']):
            r={k:row[k] for k in ('id','width','height','image_sha','label_sha','component_id')}
            # Canonical serialized Windows paths contain doubled separators; use frozen ID filenames.
            for kind,ext,key in [('images','.jpg','image'),('labels','.txt','label')]:
                source=data/ds/kind/(row['id']+ext);target=out/kind/'test'/source.name
                check_hash(source,row[key+'_sha'])
                # Separate copies keep Ultralytics cache writes away from canonical input files.
                with source.open('rb') as src,target.open('xb') as dst:dst.write(src.read())
                check_hash(target,row[key+'_sha']);r[key]=str(target)
            lines=Path(r['label']).read_text(encoding='utf-8').splitlines();r['bbox_rows']=len(lines)
            require(r['bbox_rows']>0 and all(len(x.split())==5 for x in lines),'Malformed canonical labels')
            records.append(r)
        text='path: '+json.dumps(out.as_posix())+'\ntrain: images/test\nval: images/test\ntest: images/test\nnames: '+json.dumps(index['names'])+'\n'
        yaml=out/'test.yaml'
        with yaml.open('x',encoding='utf-8') as f:f.write(text)
        manifest['datasets'][ds]={'n':n,'names':index['names'],'records':records,'yaml':str(yaml),'index_path':str(data/ds/'INDEX.json'),'index_sha':sha(data/ds/'INDEX.json'),'canonical_path':str(data/ds/'CANONICAL.json'),'canonical_sha':index['canonical_sha'],'historical_exposure_erased':False,'partition':'test'}
    put(E/'TEST_INPUTS.json',manifest)
    plan={'created_utc':now(),'physical_models':170,'logical_score_rows':190,'checkpoint':'best.pt chosen on dev mAP50-95; no test-driven selection','evaluation':{'imgsz':640,'batch':8,'rect':False,'conf':0.001,'iou':0.7,'max_det':300,'quantize':None,'workers':0,'device':'0','augment':False,'plots':False,'save_json':False,'deterministic':True},'backend':{'matmul_TF32':False,'cudnn_TF32':False,'cudnn_benchmark':False},'runtime':{'python':sys.version.split()[0],**{k:md.version(k) for k in ('torch','torchvision','ultralytics','numpy','Pillow')},'gpu_contains':'RTX 3070'},'limitations':['Local Windows RTX3070 CUDA12.6 evaluation differs from Linux Blackwell CUDA12.9 training; no claim of bitwise device equivalence.','Canonical source has documented historical exposure; this test is held out from the new170 training, not a never-seen external dataset.','Five paired seeds cannot yield two-sided exact sign-flip p<.05; do not alter tests to pursue significance.'],'inference_batch_rationale':'Fixed batch8 before test for local8GB VRAM; no auto-batch or automatic fallback','retry_policy':'Stop on first failure; preserve all results, no automatic retest or checkpoint substitution','training':False,'DINO_reextraction':False,'RunPod_stop':False}
    put(E/'TEST_PLAN.json',plan)
    verify_inputs();print('TEST_INPUTS_VERIFIED: GC10=343, NEU=270. No inference.',flush=True)

def seal():
    require((E/'DOWNLOAD_COMPLETE.json').exists(),'Backup not complete')
    require(not (E/'TEST_PACKAGE_LOCK.json').exists(),'Already sealed')
    inv=read(E/'DOWNLOAD_COMPLETE.json');check_hash(E/'DOWNLOADED_FILES.jsonl',inv['manifest_sha256'])
    source_inventory=read(E/'DOWNLOAD_INVENTORY.json')
    require(inv['files']==source_inventory['files'] and inv['unique_bytes']==source_inventory['unique_bytes'],'Transfer inventory total mismatch')
    downloaded={r['path']:r for r in (json.loads(line) for line in (E/'DOWNLOADED_FILES.jsonl').read_text(encoding='utf-8').splitlines())}
    require(len(downloaded)==inv['files'],'Backup manifest membership/count mismatch')
    for i,(relative,row) in enumerate(downloaded.items(),1):
        target=ROOT/'server_snapshot'/relative
        require(target.stat().st_size==row['bytes'],'Local backup file size mismatch: '+relative)
        check_hash(target,row['sha256'])
        if i%2000==0 or i==len(downloaded):print('BACKUP_DISK_READBACK_VERIFIED',i,'/',len(downloaded),flush=True)
    put(E/'LOCAL_BACKUP_READBACK_PASS.json',{'time_utc':now(),'selected_files_verified_from_disk':len(downloaded),'manifest_sha':inv['manifest_sha256'],'server_mutations':False})
    done=read(P/'state/MODELS_COMPLETE.json');audit=read(P/'state/MODEL_HASH_AUDIT.json')
    require(done['models']==audit['checked']==len(audit['models'])==170,'Final audit cardinality')
    check_hash(P/'state/MODEL_HASH_AUDIT.json',done['model_audit_sha'])
    matrix=read(P/'config/MODEL_MATRIX.json');require(len(matrix)==170,'Matrix cardinality')
    rows=[];audit_by_id={x['cell_id']:x for x in audit['models']}
    for c in matrix:
        cell=f"{c['dataset']}_{c['method']}_pair{c['pair']}_R{c['round']}";folder=P/'runs/models'/cell
        complete=read(folder/'COMPLETE.json');training=read(folder/'TRAINING_COMPLETE.json');loader=read(folder/'loader_ids.json')
        require(complete['status']=='PASS' and complete['completed_epochs']==training['completed_epochs']==200,'Incomplete training: '+cell)
        require(complete['seed']==c['learner'] and complete['batch']==32 and complete['train_n']==c['train_images'],'Training matrix mismatch: '+cell)
        require(not complete['test_inference'] and not complete['technical_only'],'Unexpected test/technical training record')
        require(len(loader['train'])==c['train_images'],'Train membership cardinality')
        local_index=read(BASE/'research_extensions/runpod_final_200e_20261023_01/data'/c['dataset']/'INDEX.json')
        require(loader['dev']==[r['id'] for r in local_index['dev']],'Original local dev membership mismatch')
        require(set(loader['train'])<={r['id'] for r in local_index['train']},'Train outside canonical training pool')
        for which in ('best','last'):
            path=folder/f'train/weights/{which}.pt';rel=path.relative_to(ROOT/'server_snapshot').as_posix()
            require(downloaded[rel]['sha256']==complete[which+'_sha']==training[which+'_sha']==audit_by_id[cell][which+'_sha'],'Audit/download digest mismatch: '+cell)
        rows.append({'cell_id':cell,'dataset':c['dataset'],'method':c['method'],'pair':c['pair'],'budget_percent':c['budget_percent'],'round':c['round'],'train_images':c['train_images'],'learner_seed':c['learner'],'acquisition_seed':c['acquisition'],'best_epoch':complete['best_epoch'],'best_path':str(folder/'train/weights/best.pt'),'best_sha':complete['best_sha'],'last_path':str(folder/'train/weights/last.pt'),'last_sha':complete['last_sha'],'complete_path':str(folder/'COMPLETE.json'),'complete_sha':sha(folder/'COMPLETE.json'),'loader_path':str(folder/'loader_ids.json'),'loader_sha':complete['loader_ids_sha']})
    require(len(rows)==len({r['cell_id'] for r in rows})==170,'Physical model IDs')
    put(E/'TEST_MODELS.json',{'models':rows,'physical_count':170,'source_audit_sha':sha(P/'state/MODEL_HASH_AUDIT.json')})
    # Verify every received checkpoint independently from local disk before shutdown readiness.
    verify_models();verify_inputs()
    files={str(p):sha(p) for p in (ROOT/'tools').glob('*.py')}
    for p in [E/'EXECUTION_AUTHORIZATION.json',E/'TEST_INPUTS.json',E/'TEST_PLAN.json',E/'TEST_MODELS.json',E/'DOWNLOAD_COMPLETE.json',E/'DOWNLOADED_FILES.jsonl',E/'DOWNLOAD_INVENTORY.json',E/'LOCAL_BACKUP_READBACK_PASS.json',P/'config/REPORTING_PLAN.json',P/'config/MODEL_MATRIX.json',P/'state/MODELS_COMPLETE.json',P/'state/MODEL_HASH_AUDIT.json',O/'gpu_guard_revision_20261023_01/gpu_guard.py',O/'gpu_guard_revision_20261023_02/gpu_guard.py',O/'verify_closure.py',BASE/'research_extensions/full_data_26s_sgd_cosine_640_20261018_01/windows_job.py',ROOT/'Run-Final-Test.ps1']:
        files[str(p)]=sha(p)
    # Freeze the native evaluation/loader source without importing ML or creating CUDA contexts.
    dist=md.distribution('ultralytics')
    for member in dist.files:
        if str(member).replace('\\','/').startswith('ultralytics/') and str(member).endswith('.py'):
            native=Path(dist.locate_file(member));files[str(native)]=sha(native)
    for d in read(E/'TEST_INPUTS.json')['datasets'].values():
        for p in (d['yaml'],d['index_path'],d['canonical_path']):files[p]=sha(p)
    put(E/'TEST_PACKAGE_LOCK.json',{'created_utc':now(),'files':files,'policy':'Fixed before first test inference; no automatic retry'})
    put(E/'LOCAL_PREFLIGHT_PASS.json',{'time_utc':now(),'checkpoints_verified':340,'models_verified':170,'test_images':613,'GPU_test_started':False,'backup_manifest_sha':inv['manifest_sha256']})
    print('LOCAL_BACKUP_AND_FIXED_TEST_PACKAGE_READY',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['inputs','seal']);a=ap.parse_args()
    inputs() if a.action=='inputs' else seal()
