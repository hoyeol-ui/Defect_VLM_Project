"""Finish the already-trained GC10 shared10 cell. No training entry point exists here."""
from pathlib import Path
import os,sys,json,time,csv,hashlib,argparse,traceback,platform
R=Path(__file__).resolve().parent;P=R.parents[1];B=P.parents[1]
C=P/'campaign';OUT=C/'recovery/20261021_entropy_eligible_guard_01';RUN=P/'runs/GC10-DET_Random_pair1_R0'
sys.path.insert(0,str(P));sys.path.insert(0,str(P/'src'))
from integrity import verify,sha
from campaign_lib import read,kst

def put(path,data):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)

def gate():
    verify();v=read(OUT/'VALIDATION_PASS.json');assert v['status']=='PASS' and v['all_required_regression_checks_passed']
    assert v['protocol_sha']==sha(P/'config/protocol.json') and v['execution_freeze_sha']==sha(P/'evidence/execution_freeze.json')
    b=read(R/'before_manifest.json');assert sha(P/'config/protocol.json')==b['protocol_sha']
    for name in ['best.pt','last.pt']:
        rel=f'runs/GC10-DET_Random_pair1_R0/train/weights/{name}'
        assert sha(P/rel)==b['preserved'][rel]['sha256']==sha(R/'before'/rel)
    return v

def postprocess():
    validation=gate();assert not (RUN/'RESULT.json').exists()
    for f in ['entropy_adapter_check.json','sealed_dev_metrics.json','ENVIRONMENT.json']:
        assert not (RUN/f).exists(),f'Existing artifact must not be overwritten: {f}'
    assert not (RUN/'sealed_dev').exists()
    os.environ.update(MODERN_YOLO_RUN=str(RUN),YOLO_CONFIG_DIR=str(OUT/'postprocess_settings'),MPLCONFIGDIR=str(OUT/'postprocess_mpl'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    import torch,numpy as np,ultralytics,importlib.metadata as meta
    from core import put as cell_put
    import ultralytics.data.utils as du
    import ultralytics.utils as utils
    utils.ONLINE=False;du.check_font=lambda *a,**kw:None
    from entropy_adapter import score_one2one,image_tensor
    from trainer_adapter import finite
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    assert torch.__version__=='2.12.1+cu126' and ultralytics.__version__=='8.4.90'
    init_seeds(2026091900,deterministic=True);torch.set_num_threads(4)
    idx=read(P/'inputs/GC10-DET/index.json');ids=read(RUN/'loader_ids.json')
    complete=read(RUN/'TRAINING_COMPLETE.json');assert complete['epochs']==100 and complete['optimizer_updates']>0 and complete['technical_only'] is False
    selection=read(P/'selections/GC10-DET/2026091901/Random/R0.json')['cumulative_ids']
    assert ids['train']==sorted(selection) and ids['dev']==[r['id'] for r in idx['dev']]
    pool={r['id']:r for r in idx['train']};train=[pool[i] for i in sorted(selection)]
    start=time.perf_counter();checks=[]
    detector=YOLO(str(RUN/'train/weights/best.pt')).model.float().cuda().eval()
    for r in train[:2]:
        x=image_tensor(r['image']);first=score_one2one(detector,x,len(idx['names']));again=score_one2one(detector,x,len(idx['names']));assert first==again
        checks.append({'id':r['id'],'raw_sha':first['raw_sha'],'shape':first['shape'],'detections':first['detections'],'uncertainty':first['uncertainty'],'repeat_equal':True,'score_lineage':True,'geometry_diagnostics':first['geometry_diagnostics']})
    cell_put(RUN/'entropy_adapter_check.json',{'status':'PASS','head':'one2one_before_topk','images':checks,'no_GT_in_scoring':True,'recovered_from_preserved_checkpoint':True,'execution_freeze_sha':validation['execution_freeze_sha']})
    del detector;torch.cuda.empty_cache()
    # Exactly the original pending development evaluation arguments; never split=test.
    m=YOLO(str(RUN/'train/weights/best.pt')).val(data=str(RUN/'data.yaml'),split='val',imgsz=640,batch=8,
        conf=.001,iou=.7,max_det=300,device=0,workers=4,quantize=32,rect=False,
        seed=2026091900,deterministic=True,plots=False,save_json=False,project=str(RUN),name='sealed_dev',exist_ok=False)
    finite(m.results_dict,'development_metrics')
    cell_put(RUN/'sealed_dev_metrics.json',{'overall':{k:float(v) for k,v in m.results_dict.items()},'all_AP_by_class_and_IoU':{idx['names'][int(k)]:[float(x) for x in ap] for k,ap in zip(m.box.ap_class_index,m.box.all_ap)},'partition':'development','technical_only':False,'head':'one2one'})
    for part,rows in [('train',train),('dev',idx['dev'])]:
        for r in rows:
            for typ,key in [('images','image'),('labels','label')]:
                assert sha(r[key])==r[key+'_sha']==sha(RUN/'view'/typ/part/Path(r[key]).name)
    assert sha(RUN/'train/weights/best.pt')==complete['best_sha'] and sha(RUN/'train/weights/last.pt')==complete['last_sha']
    cell_put(RUN/'ENVIRONMENT.json',{'python':sys.version,'torch':torch.__version__,'ultralytics':ultralytics.__version__,'gpu':torch.cuda.get_device_name(0),'torchvision':meta.version('torchvision')})
    seconds=time.perf_counter()-start
    original=next(r for r in csv.DictReader((C/'execution_ledger.csv').open(encoding='utf-8-sig',newline='')) if r['cell_id']==RUN.name)
    total=float(original['total_cell_seconds'])+seconds
    cell_put(RUN/'RESULT.json',{'status':'PASS','technical_only':False,'dataset':'GC10-DET','epochs':100,'optimizer_updates':complete['optimizer_updates'],'seconds':total,'training_seconds':complete['training_seconds'],'test_evaluated':False,'best_sha':complete['best_sha'],'last_sha':complete['last_sha'],'initial_state_sha':read(RUN/'training_backend.json')['initial_model_state_sha'],'recovered_postprocessing_only':True,'training_execution_freeze_sha':read(RUN/'ENVIRONMENT_AT_START.json')['execution_freeze_sha'],'postprocessing_execution_freeze_sha':validation['execution_freeze_sha'],'recovery_seconds':seconds})
    put(OUT/'POSTPROCESS_PASS.json',{'status':'PASS','date_KST':kst(),'postprocess_seconds':seconds,'training_repeated':False,'test_inference':False,'checkpoint_reused':True,'best_sha':complete['best_sha'],'last_sha':complete['last_sha'],'result_sha':sha(RUN/'RESULT.json'),'execution_freeze_sha':validation['execution_freeze_sha'],'development_only':True})
    print(json.dumps(read(OUT/'POSTPROCESS_PASS.json'),ensure_ascii=False,indent=2))

def finalize():
    validation=gate();post=read(OUT/'POSTPROCESS_PASS.json');assert post['status']=='PASS'
    diag=C.parent/'acquisition_diagnostics/GC10-DET/2026091901/R1_diagnostics.json'
    manifest=P/'selections/GC10-DET/2026091901/Entropy/R1.json'
    d=read(diag);m=read(manifest);assert d['gate']=='PASS' and d['geometry_validation']['invalid_eligible_boxes']==0
    assert m['diagnostics_sha']==sha(diag) and m['parent_best_sha']==post['best_sha']
    assert read(P/'acquisition_diagnostics/GC10-DET/2026091901/R1_ranking.json')==read(OUT/'GC10-DET/ranking.json')
    assert m['selected_ids']==read(OUT/'GC10-DET/selected_diagnostic_ids.json')['ids']
    verify()
    # Original logs, failure JSON and all training artifacts must remain unchanged.
    before=read(R/'before_manifest.json')
    originals=[]
    for rel,v in before['preserved'].items():
        if rel.startswith('runs/GC10-DET_Random_pair1_R0/'):
            assert sha(P/rel)==v['sha256'],f'Original training artifact changed: {rel}'
            originals.append(rel)
    path=C/'execution_ledger.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    r=next(r for r in rows if r['cell_id']==RUN.name);assert r['state']=='FAILED'
    result=read(RUN/'RESULT.json');r.update(state='SUCCEEDED',ended_KST=kst(),updated_KST=kst(),epochs_completed=100,training_seconds=result['training_seconds'],total_cell_seconds=result['seconds'],best_sha=result['best_sha'],best_epoch=read(RUN/'best_selection.json')['epoch'],acquire_round='1')
    # Keep the original traceback in the error field as well as its original files.
    from campaign_controller import event,save_ledger,status
    put(OUT/'RECOVERY_PASS.json',{'status':'PASS','date_KST':kst(),'protocol_sha':validation['protocol_sha'],'execution_freeze_sha':validation['execution_freeze_sha'],'training_repeated':False,'test_inference':False,'checkpoint_reused':True,'original_training_artifacts_unchanged':originals,'all_pool_regression_passed':True,'GC10_R1_acquisition_passed':True,'GC10_R1_manifest_sha':sha(manifest),'NEU_scope':'Full-model whole train-pool diagnostic; its shared10 parent is not trained yet','normal_and_oracle_rankings_identical':True,'regular_campaign_resume_allowed':True})
    save_ledger(rows);event('CELL_RECOVERED_WITHOUT_RETRAINING',cell=RUN.name,best_sha=result['best_sha'],training_seconds=result['training_seconds'],postprocess_seconds=post['postprocess_seconds'],original_failure_preserved=True,report=str(OUT/'RECOVERY_PASS.json'))
    status(rows,'RECOVERED_READY_TO_RESUME')
    print(json.dumps({'status':'PASS','new_succeeded':2,'reused':1,'not_yet_trained':99,'ETA':read(C/'status.json')['remaining_training_plus_acquisition_hours']},ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['postprocess','finalize'],required=True);a=ap.parse_args()
    assert not (C/'CAMPAIGN_ACTIVE.lock').exists()
    lock=B/'research_extensions/GPU_TRAINING_ACTIVE.lock'
    with lock.open('x',encoding='utf-8') as f:json.dump({'controller_pid':os.getpid(),'action':'post_training_recovery_'+a.stage},f)
    try:postprocess() if a.stage=='postprocess' else finalize()
    except BaseException:
        if OUT.exists():put(OUT/(a.stage.upper()+'_FAILURE.json'),{'error':traceback.format_exc(),'training_repeated':False,'test_inference':False})
        raise
    finally:
        if lock.exists() and read(lock).get('controller_pid')==os.getpid():lock.unlink()
