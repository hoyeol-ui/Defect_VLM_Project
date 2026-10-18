"""One fixed-input scientific cell or one full-data one-epoch technical smoke."""
from core import *
import shutil,traceback,platform,subprocess,importlib.metadata
def run(a):
 try:
  execute(a)
 except BaseException as e:
  put(STATE/'FAILURE.json',{'error':str(e),'type':type(e).__name__,'traceback':traceback.format_exc(),'time':time.time(),'automatic_retry':False,'batch_changed':False})
  raise
def execute(a):
 start=time.perf_counter();require(STATE.exists(),'Launcher creates unique run')
 import torch,numpy as np,yaml,ultralytics
 import ultralytics.data.utils as data_utils
 import ultralytics.utils as utils
 utils.ONLINE=False;data_utils.check_font=lambda *args,**kwargs:None
 from modern_trainer import ModernTrainer
 from trainer_adapter import finite
 from ultralytics.utils.torch_utils import init_seeds
 torch.set_num_threads(4)
 require(torch.cuda.is_available() and torch.cuda.device_count()==1,'One CUDA GPU required')
 require('RTX 3070' in torch.cuda.get_device_name(0),'Common local GPU changed; create new campaign')
 require(ultralytics.__version__=='8.4.90','Ultralytics version drift')
 require(torch.__version__=='2.12.1+cu126','Torch version drift')
 dataset=WORK/'datasets'/a.dataset;sealed=readj(WORK/'evidence/INPUT_SEAL.json')
 # Verify the copied cell inputs at the point of use, without rescanning unrelated originals.
 selected_files={rel:h for rel,h in sealed['files'].items() if rel.startswith('datasets/'+a.dataset+'/')}
 for rel,h in selected_files.items():require(sha(WORK/rel)==h,'Copied data changed '+rel)
 weight=WORK/'assets'/(a.model+'.pt')
 wh=sealed['files'].get('assets/'+weight.name) or sealed['downloaded_weights'][weight.name]
 require(sha(weight)==wh,'Pretrained SHA mismatch')
 ids=readj(dataset/'loader_ids.json');names=readj(dataset/'names.json')
 view=STATE/'view'
 for part in ['train','dev']:
  for typ in ['images','labels']:
   dest=view/typ/part;dest.mkdir(parents=True)
   for src in sorted((dataset/typ/part).iterdir()):os.link(src,dest/src.name) # isolated copies only
 shutil.copyfile(dataset/'selection_manifest.json',STATE/'selection_manifest.json')
 put(STATE/'loader_ids.json',ids)
 data={'path':str(view),'train':str(view/'images/train'),'val':str(view/'images/dev'),'names':dict(enumerate(names))}
 (STATE/'data.yaml').write_text(yaml.safe_dump(data,allow_unicode=True),encoding='utf-8')
 epochs=1 if a.phase=='smoke' else 100
 kwargs=readj(WORK/'config/common_args.json');kwargs.update(model=str(weight),data=str(STATE/'data.yaml'),project=str(STATE),name='train',epochs=epochs)
 put(STATE/'REQUEST.json',{'dataset':a.dataset,'model':a.model,'phase':a.phase,'epochs':epochs,'weight_sha':wh,'acquisition_seed':None,'diagnostic_full_train':True,'not_active_learning_budget':True,'learner_seed':2026091900,'selection_sha':sha(dataset/'selection_manifest.json'),'test_access':False,'technical_scores_not_for_model_selection':a.phase=='smoke'})
 env={'python':platform.python_version(),'torch':torch.__version__,'torchvision':importlib.metadata.version('torchvision'),'ultralytics':ultralytics.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()}
 put(STATE/'ENVIRONMENT.json',env)
 torch.cuda.reset_peak_memory_stats();init_seeds(2026091900,deterministic=True)
 trainer=ModernTrainer(overrides=kwargs,expected=kwargs,artifact_dir=STATE,controller_pid=a.controller_pid)
 trainer.train();torch.cuda.synchronize();training_end=time.perf_counter()
 require(len(trainer.epoch_records)==epochs,'Incomplete epochs')
 train_observation=readj(STATE/'OBSERVATION.json')
 best=STATE/'train/weights/best.pt';last=STATE/'train/weights/last.pt'
 for p in [best,last]:
  ckpt=torch.load(p,map_location='cpu',weights_only=False)
  require(all(k in ckpt for k in ['ema','optimizer','scheduler','scaler','rng']),'Full checkpoint state missing')
  require(all(t.dtype==torch.float32 for t in ckpt['ema'].parameters()),'EMA checkpoint is not FP32')
  require(ckpt['epoch']==(epochs-1 if p==last else readj(STATE/'best_selection.json')['epoch']-1),'Checkpoint epoch mismatch')
  del ckpt
 del trainer;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 from ultralytics import YOLO
 metrics=YOLO(str(best)).val(data=str(STATE/'data.yaml'),split='val',imgsz=640,batch=8,conf=.001,iou=.7,max_det=300,device=0,workers=4,quantize=32,rect=False,seed=2026091900,deterministic=True,plots=False,save_json=True,project=str(STATE),name='sealed_dev',exist_ok=False)
 torch.cuda.synchronize();finite(metrics.results_dict,'sealed_dev')
 class_ap={int(k):float(np.mean(v)) for k,v in zip(metrics.box.ap_class_index,metrics.box.all_ap)}
 class_all_ap={int(k):[float(x) for x in v] for k,v in zip(metrics.box.ap_class_index,metrics.box.all_ap)}
 put(STATE/'sealed_dev_metrics.json',{'overall':{k:float(v) for k,v in metrics.results_dict.items()},'per_class_AP50_95':{name:class_ap.get(i,0.) for i,name in enumerate(names)},'all_AP_by_class_and_IoU':{name:class_all_ap.get(i,[0.]*10) for i,name in enumerate(names)},'IoU_thresholds':[.5+.05*i for i in range(10)],'speed_ms_per_image':{k:float(v) for k,v in metrics.speed.items()},'partition':'development','technical_only':a.phase=='smoke','missing_prediction_class_rule':'zero; development GT contains every class'})
 devpeak={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
 for rel,h in selected_files.items():require(sha(WORK/rel)==h,'Copied dataset modified '+rel)
 backup=STATE/'backup';backup.mkdir()
 for p in [best,last]:
  target=backup/p.name;shutil.copyfile(p,target);require(sha(target)==sha(p),'Backup/restore byte mismatch')
 elapsed=time.perf_counter()-start
 put(STATE/'RESULT.json',{'status':'PASS','phase':a.phase,'dataset':a.dataset,'model':a.model,'epochs':epochs,'train_batches':train_observation['train_batches'],'train_n':len(ids['train']),'dev_n':len(ids['dev']),'train_batch':8,'epoch_dev_batch':16,'sealed_dev_batch':8,'AMP':True,'train_peak_allocated_bytes':train_observation['peak_allocated_bytes'],'train_peak_reserved_bytes':train_observation['peak_reserved_bytes'],'sealed_dev_peak_bytes':devpeak,'setup_train_epochdev_seconds':training_end-start,'total_seconds':elapsed,'best_sha':sha(best),'last_sha':sha(last),'test_evaluated':False,'automatic_batch_change':False,'numeric_recipe':'original strict_loop+guarded_step','source_hashes':{p.name:sha(p) for p in ROOT.glob('*.py')},'common_args_sha':sha(WORK/'config/common_args.json')})
 print('RESEARCH_A_RESULT '+json.dumps(readj(STATE/'RESULT.json')),flush=True)
