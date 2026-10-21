"""One explicit scientific cell or a bounded technical train/dev smoke. Never touches test."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,time,traceback,argparse
P=Path(__file__).resolve().parent;B=P.parents[1]
PY=B/'Defect_VLM_Project/.python311/python.exe'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def worker(a):
    out=Path(a.output).resolve()
    assert out.parent in [P/'smoke',P/'runs']
    permit=read(out/'JOB_READY.json')
    assert os.getppid()==a.parent==permit['controller_pid'] and os.getpid()==permit['worker_pid']
    os.environ.update(MODERN_YOLO_RUN=str(out),YOLO_CONFIG_DIR=str(out/'settings'),MPLCONFIGDIR=str(out/'mpl'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',YOLO_AUTOINSTALL='false',
        YOLO_OFFLINE='true',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    sys.path.insert(0,str(P/'src'))
    import torch,numpy as np,yaml,ultralytics,importlib.metadata as meta
    from core import put
    import ultralytics.data.utils as du
    import ultralytics.utils as utils
    utils.ONLINE=False;du.check_font=lambda *args,**kwargs:None
    from amp_compat import install
    install()
    from modern_trainer import ModernTrainer
    from trainer_adapter import finite
    from ultralytics.utils.torch_utils import init_seeds
    assert torch.__version__=='2.12.1+cu126' and ultralytics.__version__=='8.4.90'
    assert torch.cuda.is_available() and 'RTX 3070' in torch.cuda.get_device_name(0)
    torch.set_num_threads(4)
    protocol=read(P/'config/protocol.json');pair=protocol['seed_pairs'][a.pair-1]
    prep=read(P/'evidence/preparation.json')
    assert sha(P/'inputs'/a.dataset/'index.json')==prep['datasets'][a.dataset]['index_sha']
    index=read(P/'inputs'/a.dataset/'index.json');pool={r['id']:r for r in index['train']}
    if a.smoke:
        # Deterministically include every class in each technical subset, then fill by ID.
        def subset(rows,n):
            chosen={};covered=set()
            for r in rows:
                classes={int(line.split()[0]) for line in Path(r['label']).read_text().splitlines() if line.strip()}
                if classes-covered:chosen[r['id']]=r;covered|=classes
            assert len(covered)==len(index['names']) and len(chosen)<=n
            for r in rows:
                if len(chosen)>=n:break
                chosen[r['id']]=r
            assert len(chosen)==n
            return [chosen[k] for k in sorted(chosen)]
        train=subset(index['train'],128);dev=subset(index['dev'],32)
    else:
        if a.method=='FullySupervised':selected=sorted(pool)
        else:
            method='Random' if a.round==0 else a.method
            m=read(P/'selections'/a.dataset/str(pair['acquisition'])/method/f'R{a.round}.json')
            assert m['seed']==pair['acquisition'] and m['round']==a.round
            selected=m['cumulative_ids']
            assert len(selected)==protocol['counts'][a.dataset][a.round]
            if method=='Entropy':
                prior_method='Random' if a.round==1 else 'Entropy'
                previous=read(P/'selections'/a.dataset/str(pair['acquisition'])/prior_method/f'R{a.round-1}.json')['cumulative_ids']
                prior=P/'runs'/f'{a.dataset}_{prior_method}_pair{a.pair}_R{a.round-1}'
                assert m['cumulative_ids']==previous+m['selected_ids']
                assert m['parent_best_sha']==read(prior/'RESULT.json')['best_sha']==sha(prior/'train/weights/best.pt')
                assert m['score']=='ENTROPY_NCS_MEAN_V2' and m['head']=='one2one_before_topk'
        assert len(set(selected))==len(selected) and set(selected)<=set(pool)
        train=[pool[k] for k in sorted(selected)];dev=index['dev']
    assert not ({r['id'] for r in train}&{r['id'] for r in dev})
    sources=[]
    for part,rows in [('train',train),('dev',dev)]:
        for typ in ['images','labels']:(out/'view'/typ/part).mkdir(parents=True)
        for r in rows:
            for typ,key in [('images','image'),('labels','label')]:
                source=Path(r[key]);assert sha(source)==r[key+'_sha']
                dest=out/'view'/typ/part/source.name
                # Hardlinks are private views; native cache is written next to the new label directories.
                os.link(source,dest);sources.append((dest,r[key+'_sha']))
    ids={'train':[r['id'] for r in train],'dev':[r['id'] for r in dev]}
    put(out/'loader_ids.json',ids)
    data={'path':str(out/'view'),'train':str(out/'view/images/train'),'val':str(out/'view/images/dev'),'names':dict(enumerate(index['names']))}
    (out/'data.yaml').write_text(yaml.safe_dump(data,allow_unicode=True),encoding='utf-8')
    weight=B/'research_extensions/full_data_26s_sgd_cosine_640_20261018_01/assets/yolo26s.pt'
    assert sha(weight)==protocol['pretrained_sha']
    args=read(P/'config/common_args.json')
    args.update(model=str(weight),data=str(out/'data.yaml'),project=str(out),name='train',seed=pair['learner'])
    if a.smoke:args.update(epochs=1,patience=1,workers=0)
    put(out/'REQUEST.json',{'dataset':a.dataset,'method':a.method,'pair':a.pair,'acquisition_seed':None if a.method=='FullySupervised' else pair['acquisition'],
        'learner_seed':pair['learner'],'train_n':len(train),'dev_n':len(dev),'technical_only':a.smoke,
        'smoke_overrides':{'epochs':1,'patience':1,'workers':0,'train_n':128,'dev_n':32} if a.smoke else {},'test_access':False})
    put(out/'ENVIRONMENT_AT_START.json',{'python':sys.version,'torch':torch.__version__,'ultralytics':ultralytics.__version__,
        'gpu':torch.cuda.get_device_name(0),'torchvision':meta.version('torchvision'),
        'protocol_sha':sha(P/'config/protocol.json'),'common_args_sha':sha(P/'config/common_args.json'),
        'execution_freeze_sha':sha(P/'evidence/execution_freeze.json')})
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats();init_seeds(pair['learner'],deterministic=True)
    trainer=ModernTrainer(overrides=args,expected=args,artifact_dir=out,controller_pid=a.parent)
    trainer.train();torch.cuda.synchronize()
    assert len(trainer.epoch_records)==args['epochs']
    stats=read(out/'amp_summary.json');updates=stats['updates'] if 'updates' in stats else sum(v['updates'] for v in stats['by_epoch'].values())
    assert updates>0
    training_seconds=time.perf_counter()-start
    put(out/'TRAINING_COMPLETE.json',{'epochs':args['epochs'],'optimizer_updates':updates,'training_seconds':training_seconds,
        'best_sha':sha(out/'train/weights/best.pt'),'last_sha':sha(out/'train/weights/last.pt'),'technical_only':a.smoke})
    del trainer;torch.cuda.empty_cache()
    from ultralytics import YOLO
    from entropy_adapter import score_one2one,image_tensor
    # Use an unfused model so the complete per-class branch is still available.
    detector=YOLO(str(out/'train/weights/best.pt')).model.float().cuda().eval()
    assert detector.end2end and detector.model[-1].nc==len(index['names'])
    entropy_checks=[]
    for r in train[:2]:
        x=image_tensor(r['image']);first=score_one2one(detector,x,len(index['names']));again=score_one2one(detector,x,len(index['names']))
        assert first==again
        entropy_checks.append({'id':r['id'],'raw_sha':first['raw_sha'],'shape':first['shape'],
            'detections':first['detections'],'uncertainty':first['uncertainty'],'repeat_equal':True,'score_lineage':True})
    put(out/'entropy_adapter_check.json',{'status':'PASS','head':'one2one_before_topk','images':entropy_checks,'no_GT_in_scoring':True})
    del detector;torch.cuda.empty_cache()
    m=YOLO(str(out/'train/weights/best.pt')).val(data=str(out/'data.yaml'),split='val',imgsz=640,batch=8,
        conf=.001,iou=.7,max_det=300,device=0,workers=0 if a.smoke else 4,quantize=32,rect=False,
        seed=pair['learner'],deterministic=True,plots=False,save_json=False,project=str(out),name='sealed_dev',exist_ok=False)
    finite(m.results_dict,'development_metrics')
    put(out/'sealed_dev_metrics.json',{'overall':{k:float(v) for k,v in m.results_dict.items()},
        'all_AP_by_class_and_IoU':{index['names'][int(k)]:[float(x) for x in ap] for k,ap in zip(m.box.ap_class_index,m.box.all_ap)},
        'partition':'development','technical_only':a.smoke,'head':'one2one'})
    for dest,h in sources:assert sha(dest)==h
    put(out/'ENVIRONMENT.json',{'python':sys.version,'torch':torch.__version__,'ultralytics':ultralytics.__version__,
        'gpu':torch.cuda.get_device_name(0),'torchvision':meta.version('torchvision')})
    put(out/'RESULT.json',{'status':'PASS','technical_only':a.smoke,'dataset':a.dataset,'epochs':args['epochs'],
        'optimizer_updates':updates,'seconds':time.perf_counter()-start,'training_seconds':training_seconds,'test_evaluated':False,
        'best_sha':sha(out/'train/weights/best.pt'),'last_sha':sha(out/'train/weights/last.pt'),
        'initial_state_sha':read(out/'training_backend.json')['initial_model_state_sha']})

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',choices=['GC10-DET','NEU-DET'],required=True)
    ap.add_argument('--method',choices=['Random','DINO_F518','Entropy','FullySupervised'],default='Random')
    ap.add_argument('--pair',type=int,choices=[1,2,3],default=1)
    ap.add_argument('--round',type=int,choices=range(6),default=0)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--worker',action='store_true')
    ap.add_argument('--parent',type=int);ap.add_argument('--output')
    a=ap.parse_args()
    if a.worker:
        try:worker(a)
        except BaseException:
            write(Path(a.output)/'FAILURE.json',{'traceback':traceback.format_exc(),'automatic_retry':False});raise
        return
    sys.path.insert(0,str(P))
    from integrity import verify
    if not a.smoke:verify()
    if not a.smoke:
        assert read(P/'evidence/readiness.json')['status']=='READY_LOCAL_TRAIN_DEV','Run preparation checks first'
        assert not (a.dataset=='GC10-DET' and a.method=='FullySupervised' and a.pair==1),'Reuse this existing checkpoint'
    if a.round==0 and a.method!='FullySupervised':a.method='Random'
    assert a.method!='FullySupervised' or a.round==0
    lock=B/'research_extensions/GPU_TRAINING_ACTIVE.lock'
    handle=lock.open('x',encoding='utf-8');json.dump({'controller_pid':os.getpid(),'campaign':str(P)},handle);handle.close()
    proc=None;job=None
    try:
        suffix=time.strftime('%Y%m%d_%H%M%S') if a.smoke else f'{a.method}_pair{a.pair}_R{a.round}'
        out=P/('smoke' if a.smoke else 'runs')/(a.dataset+'_'+suffix)
        out.mkdir(parents=True,exist_ok=False)
        # Inherit the existing Windows Job lifecycle helper; no detached orphan on controller exit.
        sys.path.insert(0,str(B/'research_extensions/full_data_26s_sgd_cosine_640_20261018_01'))
        from windows_job import WindowsJob
        job=WindowsJob()
        cmd=[str(PY),'-I','-B',str(Path(__file__).resolve()),'--worker','--dataset',a.dataset,'--method',a.method,
            '--pair',str(a.pair),'--round',str(a.round),'--parent',str(os.getpid()),'--output',str(out)]
        if a.smoke:cmd.append('--smoke')
        # The child waits on this permit before importing ML libraries.
        with (out/'console.log').open('x',encoding='utf-8') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=out,
                env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1'},creationflags=subprocess.CREATE_NO_WINDOW)
            job.assign(proc);write(out/'JOB_READY.json',{'controller_pid':os.getpid(),'worker_pid':proc.pid})
            code=proc.wait(timeout=300 if a.smoke else None)
        assert code==0,f'Cell failed; see {out}/console.log'
        print(json.dumps({'status':'PASS','output':str(out),'result':read(out/'RESULT.json')},ensure_ascii=False))
    finally:
        if job:job.close()
        if proc and proc.poll() is None:proc.kill();proc.wait(timeout=20)
        if lock.exists() and read(lock).get('controller_pid')==os.getpid():lock.unlink()

if __name__=='__main__':
    # Permit barrier, before the worker's model import.
    if '--worker' in sys.argv:
        where=Path(sys.argv[sys.argv.index('--output')+1]);deadline=time.monotonic()+20
        while not (where/'JOB_READY.json').exists():
            if time.monotonic()>deadline:raise RuntimeError('Missing controller permit')
            time.sleep(.05)
    main()
