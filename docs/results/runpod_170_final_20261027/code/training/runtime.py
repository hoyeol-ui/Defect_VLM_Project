"""Portable audit/IO utilities. Importing this module never initializes CUDA."""
from pathlib import Path
import contextlib, datetime, functools, hashlib, importlib.metadata, inspect, ast, json, os, shutil, subprocess, sys, time, xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
def require(ok,msg):
    if not ok:raise RuntimeError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def put(p,x,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if replace:
        temp=p.with_name(p.name+'.tmp')
        with temp.open('w',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
        os.replace(temp,p)
    else:
        with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def event(kind,**kw):
    p=ROOT/'state/events.jsonl';p.parent.mkdir(exist_ok=True)
    with p.open('a',encoding='utf-8') as f:f.write(json.dumps({'time':now(),'event':kind,**kw},allow_nan=False)+'\n')
def checked_path(rel):
    p=(ROOT/rel).resolve();require(p.is_relative_to(ROOT) and not p.is_symlink(),'Unsafe package path');return p
def verify_package(data=True):
    m=read(ROOT/'MANIFEST.json')
    for rel,info in m['files'].items():
        if not data and rel.startswith('data/'):continue
        p=checked_path(rel);require(p.is_file() and p.stat().st_size==info['bytes'] and sha(p)==info['sha256'],'Package mismatch: '+rel)
    from design import verify_design
    verify_design(ROOT)
    return sha(ROOT/'MANIFEST.json')
@functools.lru_cache(maxsize=2)
def index(ds):
    require(ds in ['GC10-DET','NEU-DET'],'Unknown dataset');path=ROOT/'data'/ds/'index.json'
    require(sha(path)==read(ROOT/'MANIFEST.json')['files'][path.relative_to(ROOT).as_posix()]['sha256'],'Index hash mismatch')
    idx=read(path)
    require(set(idx)=={'dataset','names','counts','train','dev','provenance'},'Unexpected index schema/test partition')
    return idx
def setup_environment(job):
    os.environ.update(MODERN_YOLO_RUN=str(job),YOLO_CONFIG_DIR=str(job/'settings'),MPLCONFIGDIR=str(job/'mpl'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0,str(ROOT/'src'))
def asthash(obj):return digest(ast.dump(ast.parse(inspect.getsource(obj).strip() if not inspect.getsource(obj).startswith(' ') else __import__('textwrap').dedent(inspect.getsource(obj))),include_attributes=False))
def environment():
    import torch, torchvision, ultralytics, transformers, numpy, PIL, cv2
    require(sys.platform=='linux' and sys.version_info[:3]==(3,11,9),'Use pinned Linux Python3.11.9')
    versions={'torch':torch.__version__,'torchvision':torchvision.__version__,'ultralytics':ultralytics.__version__,
        'transformers':transformers.__version__,'numpy':numpy.__version__,'Pillow':PIL.__version__,'opencv':cv2.__version__}
    expected=read(ROOT/'config/ENVIRONMENT.json')
    for k,v in expected['versions'].items():require(versions[k]==v,'Dependency version mismatch: '+k)
    from ultralytics.engine.trainer import BaseTrainer
    for name,h in read(ROOT/'config/NATIVE_CONTRACT.json').items():require(asthash(getattr(BaseTrainer,name))==h,'Native trainer implementation mismatch: '+name)
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Exactly one CUDA GPU required')
    require(list(torch.cuda.get_device_capability(0))==expected['target_capability'],'Unqualified GPU capability')
    require(torch.cuda.get_device_name(0)==expected['target_gpu_name'],'Unqualified GPU name')
    identity=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader'],text=True).strip()
    require(identity==expected['target_uuid']+', '+expected['target_driver'],'Unqualified GPU UUID/driver')
    require(torch.version.cuda=='12.9','Expected cu129 runtime')
    require('sm_120' in torch.cuda.get_arch_list(),'Wheel lacks native sm_120 support')
    require(torch.cuda.get_device_properties(0).total_memory>=20*1024**3,'At least20GiB VRAM required for target qualification')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    return {'python':sys.version,'versions':versions,'GPU':torch.cuda.get_device_name(0),'VRAM_bytes':torch.cuda.get_device_properties(0).total_memory,
        'capability':list(torch.cuda.get_device_capability(0)),'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
        'resolved_packages':sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions()),'TF32':False,'AMP':False}
def gpu_snapshot():
    require(sys.platform=='linux','RunPod GPU guard is Linux-only; Windows uses the calibrated local guard')
    x=subprocess.run(['nvidia-smi','-q','-x'],capture_output=True,text=True,check=True,timeout=20)
    tree=ET.fromstring(x.stdout);devices=tree.findall('gpu');require(len(devices)==1,'Ambiguous multi-GPU/MIG host')
    g=devices[0];procs=g.findall('./processes/process_info')
    require(not procs,'GPU process/context present; no graphics-name exemptions')
    target=read(ROOT/'config/ENVIRONMENT.json')
    require(g.findtext('uuid')==target['target_uuid'] and g.findtext('product_name')==target['target_gpu_name'],'GPU guard target identity differs')
    require(tree.findtext('driver_version')==target['target_driver'],'GPU guard driver changed')
    compute=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)
    require(not compute.stdout.strip(),'Another CUDA compute context is present')
    require(g.findtext('./mig_mode/current_mig') in [None,'N/A','Disabled'],'MIG cannot be qualified by this guard')
    free=g.findtext('./fb_memory_usage/free');require(free and free.endswith(' MiB') and int(free.split()[0])>=16*1024,'Insufficient/unknown free GPU memory')
    return {'time':now(),'uuid':g.findtext('uuid'),'driver':tree.findtext('driver_version'),'name':g.findtext('product_name'),'free':free,'xml':x.stdout}
@contextlib.contextmanager
def gpu_job(job):
    require(not (ROOT/'state/STOP.json').exists(),'Campaign STOP preserved; inspect failure, no automatic retry')
    require((ROOT/'state/GPU_SESSION_START.json').exists(),'Register actual Pod billing start with session command first')
    lock=Path('/workspace/GPU_TRAINING_ACTIVE.lock');require(not lock.exists(),'Existing GPU lock; do not remove it')
    first=gpu_snapshot();time.sleep(1);second=gpu_snapshot()
    require(first['uuid']==second['uuid'] and first['driver']==second['driver'],'GPU identity changed')
    job.mkdir(parents=True,exist_ok=False)
    put(job/'GPU_PREFLIGHT.json',{'decision':'ALLOW','snapshots':[first,second]})
    put(lock,{'pid':os.getpid(),'job':str(job),'time':now(),'UUID':first['uuid']})
    put(job/'RESERVED.json',{'pid':os.getpid(),'time':now(),'automatic_retry':False})
    try:
        setup_environment(job)
        yield
    except BaseException:
        import traceback
        error={'time':now(),'job':str(job.relative_to(ROOT)),'traceback':traceback.format_exc(),'automatic_retry':False}
        put(job/'FAILURE.json',error);put(ROOT/'state/STOP.json',error,replace=True);event('FAILED',job=str(job.relative_to(ROOT)))
        raise
    finally:
        if lock.exists() and read(lock).get('pid')==os.getpid():lock.unlink()
def make_view(job,ds,train_ids):
    idx=index(ds);lookup={r['id']:r for r in idx['train']}
    require(len(train_ids)==len(set(train_ids)) and set(train_ids)<=set(lookup),'Invalid training selection')
    ids={'train':list(train_ids),'dev':[r['id'] for r in idx['dev']]}
    for part,rows in [('train',[lookup[x] for x in train_ids]),('dev',idx['dev'])]:
        for kind in ['images','labels']:(job/'view'/kind/part).mkdir(parents=True,exist_ok=False)
        for r in rows:
            for kind,key in [('images','image'),('labels','label')]:
                src=checked_path(r[key]);require(sha(src)==r[key+'_sha'],'Selected input hash mismatch')
                dst=job/'view'/kind/part/src.name
                # Links keep all170 views bounded by original data size. Both
                # methods retain identical bytes; neither re-encodes inputs.
                try:os.link(src,dst)
                except OSError:dst.symlink_to(src)
    put(job/'loader_ids.json',ids)
    put(job/'data.yaml',{'path':str(job/'view'),'train':'images/train','val':'images/dev','names':dict(enumerate(idx['names']))})
    return idx
def args_for(job,batch,seed):
    a=read(ROOT/'config/common_args_200e.json');a.update(batch=batch,seed=seed,model=str(ROOT/'assets/yolo26s.pt'),data=str(job/'data.yaml'),project=str(job),name='train',device='0')
    require(batch in [16,32] and a['epochs']==200 and a['amp'] is False,'Explicit16/32 FP32 required');return a
def train(job,ds,train_ids,batch,seed,stop):
    import torch,ultralytics.data.utils as data_utils,ultralytics.utils as utils
    from campaign_trainer_portable import CampaignTrainer
    env=environment();idx=make_view(job,ds,train_ids);a=args_for(job,batch,seed)
    utils.ONLINE=False;data_utils.check_font=lambda *a,**k:None
    put(job/'REQUEST.json',{'dataset':ds,'args':a,'stop_after_epochs':stop,'technical_only':stop is not None,'test_access':False})
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    t=CampaignTrainer(expected=a,artifact_dir=job,smoke_epochs=stop)
    def budget_guard(trainer):
        if 'RUNPOD_DEADLINE_UNIX' in os.environ:require(time.time()<float(os.environ['RUNPOD_DEADLINE_UNIX']),'GPU wall-time cost cap reached; stop without retry')
    def space_guard(trainer):require(shutil.disk_usage(ROOT).free>=5*1024**3,'Less than5GiB free; stop before checkpoint IO failure')
    t.add_callback('on_train_batch_start',budget_guard);t.add_callback('on_train_epoch_start',space_guard);t.train();torch.cuda.synchronize()
    done=read(job/'TRAINING_COMPLETE.json');require(done['completed_epochs']==(stop or 200),'Epoch count mismatch')
    for name in ['best','last']:
        require(sha(job/'train/weights'/(name+'.pt'))==done[name+'_sha'],'Checkpoint byte mismatch')
    out={'status':'PASS','time':now(),'dataset':ds,'batch':batch,'seed':seed,'seconds':time.perf_counter()-started,
        'environment':env,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'last_epoch_seconds':read(job/'PROGRESS.json')['epoch_seconds'],'train_n':len(train_ids),'dev_n':len(idx['dev']),
        'request_sha':sha(job/'REQUEST.json'),'loader_ids_sha':sha(job/'loader_ids.json'),**done}
    put(job/'COMPLETE.json',out);return out
def check_freeze():
    f=read(ROOT/'state/FROZEN_PLAN.json');require(f['manifest_sha']==verify_package(data=False),'Package changed after plan freeze')
    require(sha(ROOT/'config/PROTOCOL.json')==f['protocol_sha'],'Protocol changed')
    for rel,h in f['precomputed_selection_files'].items():require(sha(checked_path(rel))==h,'Frozen selection changed: '+rel)
    return f
def bill_so_far(gpu_usd_hour,krw_usd,tax_factor,storage_reserve_krw):
    session=read(ROOT/'state/GPU_SESSION_START.json');hours=max(0,(time.time()-session['epoch_seconds'])/3600)
    return hours*gpu_usd_hour*krw_usd*tax_factor+storage_reserve_krw
