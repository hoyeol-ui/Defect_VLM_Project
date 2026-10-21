"""Frozen-rule regression and all train-pool raw geometry audit. Never trains or reads test images."""
from pathlib import Path
import os,sys,json,hashlib,time,importlib.util,numpy as np,traceback
R=Path(__file__).resolve().parent;P=R.parents[1];B=P.parents[1]
OUT=P/'campaign/recovery/20261021_entropy_eligible_guard_01'
sys.path.insert(0,str(P));sys.path.insert(0,str(P/'src'))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def progress(v):
    p=OUT/'progress.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def modules():
    import entropy as new
    import entropy_adapter as adapter
    old=load('entropy_before_recovery',R/'before/src/entropy.py')
    sys.modules['entropy']=old
    try:old_adapter=load('entropy_adapter_before_recovery',R/'before/src/entropy_adapter.py')
    finally:sys.modules['entropy']=new
    return old,old_adapter,new,adapter

def oracle(raw,nc,old,adapter):
    # Independent old implementation on eligible anchors only; restore original indices.
    eligible=np.flatnonzero(raw[4:].max(axis=0)>np.float32(.25))
    result=old.score(adapter.xyxy_to_xywh(raw[:,eligible]),nc)
    result['anchor_indices']=[int(eligible[i]) for i in result['anchor_indices']]
    return result

def cpu_regression(old,old_adapter,new,adapter):
    cases=[];rng=np.random.default_rng(20261021)
    for nc in [6,10]:
        for n in [0,1,17,360]:
            xywh=np.vstack([rng.uniform(20,600,(2,n)),rng.uniform(1,80,(2,n)),rng.uniform(0,1,(nc,n))]).astype(np.float32)
            if n:xywh[4:]*=rng.choice(np.array([.1,1],np.float32),size=(1,n))
            cases.append((f'nc{nc}_n{n}',nc,xywh))
        for kind in ['empty_predictions','threshold','ties','max300','zero_width_legacy']:
            n=380 if kind=='max300' else 12
            raw=np.zeros((4+nc,n),np.float32);raw[0]=np.arange(n)*100+20;raw[1]=20;raw[2:4]=10
            raw[4:]=.1
            if kind!='empty_predictions':raw[4]=.8
            if kind=='threshold':raw[4,:3]=[np.nextafter(np.float32(.25),np.float32(0)),np.float32(.25),np.nextafter(np.float32(.25),np.float32(1))]
            if kind=='ties':raw[:2,1:4]=raw[:2,:1];raw[4,1:4]=.8;raw[5,3]=.9
            if kind=='zero_width_legacy':raw[2,0]=0
            cases.append((f'nc{nc}_{kind}',nc,raw))
    checked=[]
    for name,nc,raw in cases:
        before=old.score(raw,nc);after=new.score(raw,nc);assert before==after,name
        xyxy=raw.copy();xyxy[:2]=raw[:2]-raw[2:4]/2;xyxy[2:4]=xyxy[:2]+raw[2:4]
        assert np.array_equal(old_adapter.xyxy_to_xywh(xyxy),adapter.xyxy_to_xywh(xyxy)),name
        assert old.score(old_adapter.xyxy_to_xywh(xyxy),nc)==new.score(adapter.xyxy_to_xywh(xyxy),nc),name
        checked.append({'case':name,'equal':True,'output_sha':digest(after)})
    adversarial=[]
    for nc in [6,10]:
        raw=np.zeros((4+nc,3),np.float32);raw[:2]=20;raw[2:4]=10;raw[4:]=.01;raw[4,0]=.8
        raw[2,1]=-3;raw[4,1]=.0000021
        out=new.score(raw,nc);assert out==oracle(np.vstack([raw[:2]-raw[2:4]/2,raw[:2]+raw[2:4]/2,raw[4:]]),nc,old,adapter)
        assert 1 not in out['anchor_indices']
        low_xyxy=np.vstack([raw[:2]-raw[2:4]/2,raw[:2]+raw[2:4]/2,raw[4:]]).astype(np.float32)
        assert np.array_equal(adapter.xyxy_to_xywh(low_xyxy),raw)
        for at,expected_fail in [(np.float32(.25),False),(np.nextafter(np.float32(.25),np.float32(1)),True)]:
            probe=raw.copy();probe[4,1]=at
            xyxy=low_xyxy.copy();xyxy[4,1]=at
            for label,func,arg in [('score',lambda x:new.score(x,nc),probe),('adapter',adapter.xyxy_to_xywh,xyxy)]:
                failed=False
                try:func(arg)
                except (AssertionError,RuntimeError):failed=True
                assert failed==expected_fail,(label,at,failed)
                adversarial.append({'nc':nc,'path':label,'confidence':float(at),'expected_failure':expected_fail,'matched':True})
    # Actual captured failing raw from the previous read-only investigation.
    saved=B/'journal_manuscript_jkics/work'
    failure_samples=[]
    for f in sorted(saved.glob('failure_raw_*.npz')):
        raw=np.load(f)['raw_xyxy'];nc=raw.shape[0]-4
        after=new.score(adapter.xyxy_to_xywh(raw),nc);ref=oracle(raw,nc,old,adapter)
        assert after==ref
        bad=np.flatnonzero((raw[2]<raw[0])|(raw[3]<raw[1]))
        assert not set(bad)&set(after['anchor_indices'])
        failure_samples.append({'raw_file':str(f),'bad_anchors':bad.tolist(),'retained_anchors':after['anchor_indices'],'equal_to_eligible_old_oracle':True})
    assert len(failure_samples)==2
    # Include an all-zero tie and exact-score ties straddling the selection cutoff.
    from entropy_diagnostics import describe
    tie_checks=[]
    for values in [[0]*12,[.7,.7,.4,.4,.4,0,0,0],[.9,.8,.7,.6,.5,.4]]:
        ranking=[{'id':f'{i:03}','uncertainty':v,'detections':int(v>0)} for i,v in enumerate(values)]
        ranking.sort(key=lambda x:(-x['uncertainty'],x['id']));chosen=[x['id'] for x in ranking[:3]]
        # Both paths have the same scores and hence the same exact ties and ID choices.
        duplicate=json.loads(json.dumps(ranking));assert describe(ranking,chosen,3)==describe(duplicate,[x['id'] for x in duplicate[:3]],3)
        tie_checks.append(describe(ranking,chosen,3))
    report={'status':'PASS','normal_cases':checked,'adversarial_checks':adversarial,'saved_failure_checks':failure_samples,'tie_selection_checks':tie_checks,'legacy_zero_width_condition_preserved':True,'threshold_nms_entropy_and_anchor_rules_unchanged':True}
    put(OUT/'cpu_regression.json',report)
    return report

def full_scan(old,old_adapter,new,adapter):
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(2026091900,deterministic=True);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    plans=[('GC10-DET','GC10-DET_Random_pair1_R0',False),('NEU-DET','NEU-DET_FullySupervised_pair1_R0',True)]
    reports=[]
    from entropy_diagnostics import describe
    cfg=read(P/'config/protocol.json')
    for ds,cell,diagnostic_only in plans:
        checkpoint=P/'runs'/cell/'train/weights/best.pt';complete=read(P/'runs'/cell/'TRAINING_COMPLETE.json');assert sha(checkpoint)==complete['best_sha']
        model=YOLO(str(checkpoint)).model.float().cuda().eval()
        idx=read(P/'inputs'/ds/'index.json');nc=len(idx['names']);rows=idx['train']
        previous=read(P/'selections'/ds/'2026091901/Random/R0.json')['cumulative_ids'];prior_set=set(previous)
        assert len(previous)==cfg['counts'][ds][0]
        excluded={r['id'] for part in ['dev','test'] for r in idx.get(part,[])}
        assert not {r['id'] for r in rows}&excluded
        ranking=[];reference_ranking=[];invalid_images=invalid_boxes=invalid_eligible=zero_eligible=old_normal=0;wrapper_checks=[]
        dsout=OUT/ds;dsout.mkdir(exist_ok=False);(dsout/'raw').mkdir()
        started=time.perf_counter();results_file=dsout/'per_image.jsonl'
        with results_file.open('x',encoding='utf-8') as log:
            for k,row in enumerate(rows):
                assert sha(row['image'])==row['image_sha']
                holder=[]
                def capture(raw):
                    holder.append(raw)
                    # All original XYXY coordinates, class scores and anchor indices, including below-gate boxes.
                    np.savez_compressed(dsout/'raw'/(row['id']+'.npz'),raw_xyxy=raw,anchor_indices=np.arange(raw.shape[1],dtype=np.int32))
                x=adapter.image_tensor(row['image'])
                result=adapter.score_one2one(model,x,nc,raw_capture=capture);raw=holder[0]
                bad=np.flatnonzero((raw[2]<raw[0])|(raw[3]<raw[1]));eligible=np.flatnonzero(raw[4:].max(axis=0)>np.float32(.25))
                invalid_images+=bool(len(bad));invalid_boxes+=len(bad);invalid_eligible+=len(set(bad)&set(eligible))
                zero_eligible+=int(((raw[2,eligible]==raw[0,eligible])|(raw[3,eligible]==raw[1,eligible])).sum())
                canonical={key:result[key] for key in ['uncertainty','anchor_indices','boxes_xyxy','confidence','class','raw_class_scores','box_entropy','detections','max_det_reached']}
                ref=oracle(raw,nc,old,adapter);assert canonical==ref,(ds,row['id'],'eligible oracle mismatch')
                assert not set(bad)&set(canonical['anchor_indices'])
                if not len(bad):
                    assert np.array_equal(old_adapter.xyxy_to_xywh(raw),adapter.xyxy_to_xywh(raw))
                    assert canonical==old.score(old_adapter.xyxy_to_xywh(raw),nc)
                    old_normal+=1
                    if len(wrapper_checks)<3:
                        before=old_adapter.score_one2one(model,x,nc)
                        assert before=={key:result[key] for key in before}
                        wrapper_checks.append({'id':row['id'],'before_after_wrapper_identical':True,'raw_sha':before['raw_sha']})
                if k in [0,len(rows)//2,len(rows)-1]:
                    again=adapter.score_one2one(model,x,nc);assert again==result
                record={'id':row['id'],'result':result,'raw_capture':str(dsout/'raw'/(row['id']+'.npz')),'oracle_equal':True,'legacy_direct_equal':not bool(len(bad))}
                log.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                if row['id'] not in prior_set:
                    ranking.append({'id':row['id'],'uncertainty':result['uncertainty'],'raw_sha':result['raw_sha'],'detections':result['detections']})
                    reference_ranking.append({'id':row['id'],'uncertainty':ref['uncertainty'],'raw_sha':result['raw_sha'],'detections':ref['detections']})
                if k%32==0 or k==len(rows)-1:
                    log.flush();progress({'stage':'ALL_TRAIN_INPUT_GEOMETRY_REGRESSION','dataset':ds,'scored':k+1,'total':len(rows),'seconds':time.perf_counter()-started,'invalid_raw_images':invalid_images,'invalid_raw_boxes':invalid_boxes,'invalid_eligible_boxes':invalid_eligible,'legacy_direct_equal_images':old_normal})
        ranking.sort(key=lambda v:(-v['uncertainty'],v['id']));reference_ranking.sort(key=lambda v:(-v['uncertainty'],v['id']))
        q=cfg['counts'][ds][1]-len(previous);chosen=[r['id'] for r in ranking[:q]]
        assert ranking==reference_ranking and chosen==[r['id'] for r in reference_ranking[:q]]
        stats=describe(ranking,chosen,q);assert stats==describe(reference_ranking,[r['id'] for r in reference_ranking[:q]],q)
        assert invalid_eligible==0 and wrapper_checks and len(ranking)==len(rows)-len(previous)
        report={'status':'PASS','dataset':ds,'checkpoint':str(checkpoint),'parent_best_sha':complete['best_sha'],'checkpoint_role':'Full diagnostic only; not the future shared10 acquisition' if diagnostic_only else 'actual shared10 parent','diagnostic_only_selection':diagnostic_only,'train_images_scanned':len(rows),'remaining_pool_images':len(ranking),'invalid_raw_images':invalid_images,'invalid_raw_image_fraction':invalid_images/len(rows),'invalid_raw_boxes':invalid_boxes,'invalid_raw_box_fraction':invalid_boxes/(len(rows)*8400),'invalid_eligible_boxes':invalid_eligible,'zero_size_eligible_boxes':zero_eligible,'normal_direct_legacy_equal_images':old_normal,'all_images_equal_to_eligible_legacy_oracle':True,'wrapper_checks':wrapper_checks,'ranking_and_chosen_ids_equal':True,'tie_diagnostics':stats,'seconds':time.perf_counter()-started,'test_images_read':0,'labels_read_for_scoring':0}
        put(dsout/'ranking.json',ranking);put(dsout/'selected_diagnostic_ids.json',{'ids':chosen,'not_a_training_manifest':True});put(dsout/'summary.json',report)
        if not diagnostic_only:assert stats['gate']=='PASS','Actual shared10 acquisition is degenerate; do not resume training'
        assert sha(checkpoint)==complete['best_sha'];reports.append(report)
        del model;torch.cuda.empty_cache()
    return reports

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    os.environ.update(MODERN_YOLO_RUN=str(P/'runs/recovery_validation'),YOLO_CONFIG_DIR=str(OUT/'settings'),MPLCONFIGDIR=str(OUT/'mpl'),YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    from integrity import verify
    verify();oldsha=sha(P/'evidence/execution_freeze.json');science=sha(P/'config/protocol.json')
    import torch,ultralytics
    assert torch.__version__=='2.12.1+cu126' and ultralytics.__version__=='8.4.90'
    old,old_adapter,new,adapter=modules();start=time.perf_counter()
    cpu=cpu_regression(old,old_adapter,new,adapter);scans=full_scan(old,old_adapter,new,adapter)
    verify();assert sha(P/'evidence/execution_freeze.json')==oldsha
    put(OUT/'VALIDATION_PASS.json',{'status':'PASS','protocol_sha':science,'execution_freeze_sha':oldsha,'normal_synthetic_cases':len(cpu['normal_cases']),'adversarial_checks':len(cpu['adversarial_checks']),'datasets':scans,'seconds':time.perf_counter()-start,'training_performed':False,'test_inference':False,'all_required_regression_checks_passed':True})
    print(json.dumps(read(OUT/'VALIDATION_PASS.json'),ensure_ascii=False,indent=2))

if __name__=='__main__':
    assert not (P/'campaign/CAMPAIGN_ACTIVE.lock').exists()
    lock=B/'research_extensions/GPU_TRAINING_ACTIVE.lock'
    with lock.open('x',encoding='utf-8') as f:json.dump({'controller_pid':os.getpid(),'action':'entropy_recovery_validation'},f)
    try:main()
    except BaseException:
        if OUT.exists():put(OUT/'VALIDATION_FAILURE.json',{'error':traceback.format_exc(),'training_performed':False,'test_inference':False})
        raise
    finally:
        if lock.exists() and read(lock).get('controller_pid')==os.getpid():lock.unlink()
