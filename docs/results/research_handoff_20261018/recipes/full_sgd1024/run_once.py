"""Exactly one authorized Full1604 best checkpoint Dev343 evaluation at 1024."""
from pathlib import Path
import os,sys,json,hashlib,shutil,time,traceback,subprocess

P=Path(__file__).resolve().parent; B=P.parents[1]
FULL=B/'research_extensions/full_data_26s_sgd_640_20261017_01'
RUN=FULL/'runs/GC10_FULL1604_yolo26s_SGD_e100_b8_i640'
WEIGHT=RUN/'train/weights/best.pt'
LOCK=B/'research_extensions/GPU_TRAINING_ACTIVE.lock'
os.environ.update(YOLO_CONFIG_DIR=str(P/'settings'),MPLCONFIGDIR=str(P/'mpl'),
    TORCH_HOME=str(P/'offline_cache'),HF_HOME=str(P/'offline_cache'),
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_OFFLINE='true',
    YOLO_AUTOINSTALL='false',WANDB_MODE='disabled',CUBLAS_WORKSPACE_CONFIG=':4096:8',
    OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
    PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def put(name,x):
    with (P/name).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def main():
    start=time.perf_counter()
    assert not (P/'RUN_ONCE.json').exists(),'Already attempted; no automatic retry'
    with LOCK.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),workspace=str(P),task='one1024_dev_evaluation'),f)
    try:
        put('RUN_ONCE.json',dict(pid=os.getpid(),authorization='User explicitly requested exactly one 1024 Full1604 frozen-best evaluation.',automatic_retry=False))
        sourcefiles=[WEIGHT,RUN/'sealed_dev_metrics.json',RUN/'sealed_dev/predictions.json',
            RUN/'loader_ids.json',RUN/'best_selection.json',RUN/'data.yaml',FULL/'src/worker.py',
            B/'research_extensions/yolo26s_sgd_300_20261017_01/diagnostics/DEV_GT.json']
        ids=read(RUN/'loader_ids.json');assert len(ids['train'])==1604 and len(ids['dev'])==343
        assert sha(WEIGHT)==read(RUN/'RESULT.json')['best_sha']==read(RUN/'best_selection.json')['best_sha']
        # Isolated dev-only filesystem view: cache creation cannot touch the baseline directory.
        for kind in ('images','labels'):
            dest=P/'view'/kind/'dev';dest.mkdir(parents=True,exist_ok=False)
            files=sorted((RUN/'view'/kind/'dev').iterdir())
            assert len(files)==343 and {f.stem for f in files}==set(ids['dev'])
            for src in files:
                sourcefiles.append(src)
                if kind=='images':os.link(src,dest/src.name)
                else:shutil.copyfile(src,dest/src.name)
        before={str(f):sha(f) for f in sourcefiles};put('INPUT_HASHES_BEFORE.json',before)
        import torch,numpy as np,yaml,ultralytics
        import ultralytics.data.utils as datautils
        import ultralytics.utils as utils
        from ultralytics.utils.torch_utils import init_seeds
        assert torch.__version__=='2.12.1+cu126' and ultralytics.__version__=='8.4.90'
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        utils.ONLINE=False;datautils.check_font=lambda *a,**k:None
        torch.set_num_threads(4)
        # Match the recorded baseline backend, including TF32 defaults.
        oldbackend=read(RUN/'training_backend.json')
        torch.backends.cuda.matmul.allow_tf32=oldbackend['matmul_TF32']
        torch.backends.cudnn.allow_tf32=oldbackend['cudnn_TF32']
        torch.backends.cudnn.benchmark=oldbackend['cudnn_benchmark']
        init_seeds(2026091900,deterministic=True)
        data=yaml.safe_load((RUN/'data.yaml').read_text(encoding='utf-8'))
        data.update(path=str(P/'view'),val=str(P/'view/images/dev'))
        assert 'test' not in data and len(data['names'])==10 and data['names'][6]=='inclusion'
        with (P/'data.yaml').open('x',encoding='utf-8') as f:yaml.safe_dump(data,f,allow_unicode=True)
        kwargs=dict(data=str(P/'data.yaml'),split='val',imgsz=1024,batch=8,conf=.001,
            iou=.7,max_det=300,device=0,workers=4,quantize=32,rect=False,
            seed=2026091900,deterministic=True,plots=False,save_json=True,
            project=str(P),name='sealed_dev_1024',exist_ok=False)
        put('REQUEST.json',dict(checkpoint=str(WEIGHT),checkpoint_sha=sha(WEIGHT),
            checkpoint_best_epoch=read(RUN/'best_selection.json')['epoch'],validation_kwargs=kwargs,
            changed_scientific_variable='imgsz640_to1024_only',baseline=str(RUN/'sealed_dev_metrics.json'),
            test_used=False,training=False,automatic_batch_change=False))
        put('ENVIRONMENT.json',dict(torch=torch.__version__,ultralytics=ultralytics.__version__,numpy=np.__version__,
            python=sys.version,gpu=torch.cuda.get_device_name(0),
            baseline_backend=oldbackend,validator_source_sha=sha(Path(ultralytics.__file__).parent/'engine/validator.py'),
            detection_validator_source_sha=sha(Path(ultralytics.__file__).parent/'models/yolo/detect/val.py')))
        from ultralytics import YOLO
        model=YOLO(str(WEIGHT));assert dict(model.names)==dict(data['names'])
        observations={};gt=read(sourcefiles[7])
        def on_start(v):
            assert not v.training and v.args.imgsz==1024 and v.args.batch==8
            assert v.args.rect is False and v.args.quantize is None # 32 request resolves to native FP32
            assert v.args.conf==.001 and v.args.iou==.7 and v.args.max_det==300 and v.args.split=='val'
            assert v.dataloader.batch_size==8 and not v.dataloader.dataset.rect
            assert sorted(Path(x).stem for x in v.dataloader.dataset.im_files)==sorted(ids['dev'])
            assert next(model.model.parameters()).dtype==torch.float32
            assert not any(p.requires_grad for p in model.model.parameters())
            for label in v.dataloader.dataset.labels:
                sid=Path(label['im_file']).stem
                original=np.loadtxt(P/'view/labels/dev'/(sid+'.txt'),dtype=np.float32,ndmin=2)
                decoded=np.concatenate([label['cls'],label['bboxes']],axis=1)
                order=lambda a:np.array(sorted(map(tuple,a.tolist())),np.float32)
                assert original.shape==decoded.shape and np.allclose(order(original),order(decoded),atol=1e-6,rtol=0)
                assert len(original)==len(gt[sid])
            observations.update(loader_n=len(v.dataloader.dataset),batch=v.dataloader.batch_size,
                rect=v.dataloader.dataset.rect,precision='FP32',resolved_quantize=v.args.quantize,
                imgsz=v.args.imgsz,conf=v.args.conf,iou=v.args.iou,max_det=v.args.max_det,
                end2end=bool(getattr(model.model,'end2end',False)),args=vars(v.args).copy())
            put('ACTUAL_EVALUATION.json',observations)
        model.add_callback('on_val_start',on_start)
        torch.cuda.reset_peak_memory_stats()
        evaluation_start=time.perf_counter()
        metrics=model.val(**kwargs) # THE ONLY VALIDATION / INFERENCE CALL IN THIS SCRIPT
        torch.cuda.synchronize()
        allap={int(k):[float(x) for x in a] for k,a in zip(metrics.box.ap_class_index,metrics.box.all_ap)}
        names=[data['names'][i] for i in range(10)]
        assert set(allap)==set(range(10))
        put('sealed_dev_metrics.json',dict(overall={k:float(v) for k,v in metrics.results_dict.items()},
            per_class_AP50_95={name:float(np.mean(allap[i])) for i,name in enumerate(names)},
            all_AP_by_class_and_IoU={name:allap[i] for i,name in enumerate(names)},
            IoU_thresholds=[.5+.05*i for i in range(10)],speed_ms_per_image=metrics.speed,
            partition='development',technical_only=False))
        assert observations['loader_n']==343 and (P/'sealed_dev_1024/predictions.json').is_file()
        after={str(f):sha(f) for f in sourcefiles};assert before==after,'Baseline input changed'
        put('VALIDATION.json',dict(status='PASS',evaluation_calls=1,baseline_inputs_unchanged=True,
            checkpoint_sha=sha(WEIGHT),dev_images=343,test_used=False,training=False,
            automatic_retry=False,batch_changed=False,evaluation_seconds=time.perf_counter()-evaluation_start,
            total_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        print(json.dumps(dict(status='COMPLETE',overall=metrics.results_dict,inclusion=allap[6],output=str(P)),ensure_ascii=False),flush=True)
    except BaseException:
        put('FAILURE.json',dict(error=traceback.format_exc(),automatic_retry=False,batch_changed=False))
        raise
    finally:
        if LOCK.exists() and read(LOCK).get('pid')==os.getpid():LOCK.unlink()

if __name__=='__main__':main()
