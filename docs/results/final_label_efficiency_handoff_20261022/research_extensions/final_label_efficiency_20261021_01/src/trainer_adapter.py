"""Strict pinned trainer: no OOM retries, NaN recovery, sanitization or lossy checkpoint stripping."""
from core import *
import copy,random,math
import numpy as np
import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
import ultralytics.engine.trainer as upstream
from ultralytics.utils.torch_utils import unwrap_model
def finite(value,name):
    if isinstance(value,torch.Tensor):require(bool(torch.isfinite(value).all()),'NaN/Inf '+name)
    elif isinstance(value,(float,np.floating)):require(math.isfinite(value),'NaN/Inf '+name)
    elif isinstance(value,dict):
        for k,v in value.items():finite(v,name+'.'+str(k))
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value):finite(v,name+'.'+str(i))
def amp_check(model):
    # Avoid upstream downloading an unrelated yolo26 model and silently disabling AMP.
    # Probe supplied cold-start model, preserve BN/RNG/mode; actual one-epoch smoke remains mandatory.
    rng=torch.get_rng_state();crng=torch.cuda.get_rng_state_all();was=model.training
    try:
        model.eval();x=torch.zeros((1,3,640,640),device='cuda',dtype=torch.float32)
        with torch.inference_mode():
            fp=model(x)[0]
            with torch.autocast('cuda',enabled=True):mixed=model(x)[0]
        finite(fp,'AMP_probe_FP32');finite(mixed,'AMP_probe_FP16');require(fp.shape==mixed.shape and torch.allclose(fp,mixed.float(),rtol=.1,atol=.5),'AMP probe mismatch; no automatic AMP disable')
    finally:model.train(was);torch.set_rng_state(rng);torch.cuda.set_rng_state_all(crng)
    return True
namespace={};exec(compile((ROOT/'strict_loop.py').read_text(),str(ROOT/'strict_loop.py'),'exec'),dict(vars(upstream)),namespace)
class StrictTrainer(DetectionTrainer):
    _do_train=namespace['_do_train']
    def __init__(self,*a,expected=None,artifact_dir=None,controller_pid=None,**kw):
        self.expected=expected;self.artifact_dir=Path(artifact_dir);self.controller_pid=controller_pid;self.epoch_records=[];self.best_epoch_record=None
        upstream.check_amp=amp_check
        upstream.callbacks.add_integration_callbacks=lambda trainer:None
        super().__init__(*a,**kw)
        self.add_callback('on_pretrain_routine_end',self._ready)
        self.add_callback('on_train_batch_start',self._parent_check)
        self.add_callback('on_train_batch_end',self._batch_check)
        self.add_callback('on_fit_epoch_end',self._heartbeat)
    def get_dataset(self):
        import yaml
        d=yaml.safe_load(Path(self.args.data).read_text());require('test' not in d and d.get('val') and d.get('train'),'Test/fallback dataset forbidden')
        return super().get_dataset()
    def _ready(self,trainer):
        self._parent_check(trainer)
        resolved=vars(self.args).copy()
        for k,v in self.expected.items():
            actual=resolved.get(k);require(str(actual)==str(v) if k=='device' else actual==v,'Resolved arg drift '+k+': '+str(actual))
        require(self.batch_size==8 and self.train_loader.batch_size==8 and self.test_loader.batch_size==16,'Actual loader batch drift')
        require(self.amp is True and self.start_epoch==0 and not self.args.resume,'AMP/resume drift')
        put(self.artifact_dir/'resolved_args.json',resolved)
        put(self.artifact_dir/'training_backend.json',{'amp':self.amp,'matmul_TF32':torch.backends.cuda.matmul.allow_tf32,'cudnn_TF32':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'train_loader_batch':self.train_loader.batch_size,'development_loader_batch':self.test_loader.batch_size,'initial_model_state_sha':digest({k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in self.model.state_dict().items()}),'unrelated_AMP_probe_model_downloaded':False})
        for part,loader in [('train',self.train_loader),('dev',self.test_loader)]:
            expected=readj(self.artifact_dir/'loader_ids.json')[part];actual=sorted(Path(x).stem for x in loader.dataset.im_files);require(actual==sorted(expected),'Actual loader membership drift '+part)
            for label in loader.dataset.labels:
                id=Path(label['im_file']).stem;source=self.artifact_dir/'view/labels'/part/(id+'.txt')
                original=np.loadtxt(source,dtype=np.float32,ndmin=2);decoded=np.concatenate([label['cls'],label['bboxes']],axis=1)
                require(original.shape==decoded.shape,'Native loader dropped or duplicated annotations '+id)
                order=lambda a:np.array(sorted(map(tuple,a.tolist())),np.float32)
                require(np.allclose(order(original),order(decoded),atol=1e-6,rtol=0),'Native annotation conversion drift '+id)
    def _parent_check(self,trainer):
        require(os.getppid()==self.controller_pid,'Controller died; stop orphaned worker')
        require(not (STATE/'STOP.json').exists(),'Global STOP latch during training')
    def _batch_check(self,trainer):
        require(self.batch_size==8 and self.args.batch==8 and self._oom_retries==0,'Hidden batch/recovery change')
        finite(self.loss,'loss');finite(self.loss_items,'loss_items')
        marker=self.artifact_dir/'TRAINING_STARTED.json'
        if not marker.exists():put(marker,{'pid':os.getpid(),'controller_pid':self.controller_pid,'time':time.time(),'device':str(self.device),'batch':self.batch_size,'AMP':self.amp,'first_batch_completed':True,'runtime_patch_sha':patch_identity()})
    def optimizer_step(self):
        from amp_runtime import guarded_step
        guarded_step(self,finite)
    def _handle_nan_recovery(self,epoch):
        finite(self.loss,'loss');finite(self.loss_items,'loss_items');finite(self.fitness,'fitness');finite(self.metrics,'metrics');require(self.nan_recovery_attempts==0,'Recovery attempt')
        require(getattr(self,'amp_stats',{}).get('by_epoch',{}).get(str(epoch+1),{}).get('updates',0)>0,'No finite optimizer update in epoch; unresolved numerical failure')
        return False
    def _load_checkpoint_state(self,ckpt):raise Stop('Hidden checkpoint recovery/resume forbidden')
    def save_model(self):
        finite(self.model.state_dict(),'model');finite(self.ema.ema.state_dict(),'EMA');finite(self.optimizer.state_dict(),'optimizer');finite(self.metrics,'metrics')
        require(self.fitness is not None and self.best_fitness is not None,'dev-best fitness missing')
        metrics=dict(self.metrics);require('metrics/mAP50-95(B)' in metrics and abs(float(self.fitness)-float(metrics['metrics/mAP50-95(B)']))<1e-10,'dev-best metric is not mAP50-95')
        checkpoint={'epoch':self.epoch,'best_fitness':self.best_fitness,'model':None,'ema':copy.deepcopy(unwrap_model(self.ema.ema)).float(),'updates':self.ema.updates,'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),'scaler':self.scaler.state_dict(),'train_args':vars(self.args),'train_metrics':{**metrics,'fitness':self.fitness},'train_results':self.read_results_csv(),'version':'8.4.90','rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'loader_generator':self.train_loader.generator.get_state() if self.train_loader.generator is not None else None},'automatic_resume_permitted':False}
        temp=self.last.with_suffix('.tmp');torch.save(checkpoint,temp);os.replace(temp,self.last)
        if self.best_fitness==self.fitness:
            import shutil
            shutil.copyfile(self.last,self.best);self.best_epoch_record=self.epoch+1
        self.epoch_records.append({'epoch':self.epoch+1,'fitness':float(self.fitness),'best_epoch':self.best_epoch_record})
        put(self.artifact_dir/'epoch_metrics.json',self.epoch_records,exclusive=False)
        return True
    def _heartbeat(self,trainer):
        put(self.artifact_dir/'progress.json',{'epoch':self.epoch+1,'time':time.time(),'checkpoint':str(self.last),'last_sha':sha(self.last) if self.last.exists() else None},exclusive=False)
    def final_eval(self):
        # Do not strip optimizer or replace best metrics with last; separate sealed dev batch8 eval follows.
        require(self.best.exists() and self.last.exists(),'Checkpoint missing')
        require(len(self.epoch_records)==self.expected['epochs'],'Incomplete epoch count')
        put(self.artifact_dir/'amp_summary.json',self.amp_stats)
        top=max(r['fitness'] for r in self.epoch_records);last_best=max(r['epoch'] for r in self.epoch_records if r['fitness']==top)
        require(self.best_epoch_record==last_best,'dev-best last-tie epoch mismatch')
        put(self.artifact_dir/'best_selection.json',{'metric':'development_mAP50-95','epoch':last_best,'fitness':top,'best_sha':sha(self.best),'last_sha':sha(self.last),'all_epochs':len(self.epoch_records),'serialization':'FP32 EMA and full optimizer/scheduler/scaler/RNG; no nan_to_num or strip_optimizer'})
