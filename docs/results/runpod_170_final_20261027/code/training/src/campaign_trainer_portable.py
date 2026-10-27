"""Native pinned learning rules with FP32 checks and a1epoch smoke stop at horizon200."""
from core import *
import copy,math,random,time
import numpy as np
import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model
from ultralytics.data import build_yolo_dataset
import ultralytics.engine.trainer as upstream


def finite(value,name):
    if isinstance(value,torch.Tensor):require(bool(torch.isfinite(value).all()),'Nonfinite '+name)
    elif isinstance(value,(float,np.floating)):require(math.isfinite(value),'Nonfinite '+name)
    elif isinstance(value,dict):
        for k,v in value.items():finite(v,name+'.'+str(k))
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value):finite(v,name+'.'+str(i))


namespace={}
exec(compile((ROOT/'strict_loop.py').read_text(encoding='utf-8'),str(ROOT/'strict_loop.py'),'exec'),dict(vars(upstream)),namespace)


class CampaignTrainer(DetectionTrainer):
    _do_train=namespace['_do_train']
    def __init__(self,*,expected,artifact_dir,smoke_epochs=None):
        self.expected=expected;self.artifact_dir=Path(artifact_dir);self.smoke_epochs=smoke_epochs
        self.records=[];self.optimizer_updates=0;self.microbatches=0;self.best_epoch=None
        upstream.callbacks.add_integration_callbacks=lambda trainer:None
        super().__init__(overrides=expected)
        self.add_callback('on_pretrain_routine_end',self._ready)
        self.add_callback('on_train_batch_start',self._batch_start)
        self.add_callback('on_train_epoch_start',self._epoch_start)
        self.add_callback('on_train_batch_end',self._batch_end)
        self.add_callback('on_fit_epoch_end',self._epoch_end)

    def build_dataset(self, img_path, mode='train', batch=None):
        gs=max(int(unwrap_model(self.model).stride.max()),32)
        # Pinned native builder is identical except that it hardcodes mode=='val'.
        # The new campaign declares rect=False for both partitions.
        return build_yolo_dataset(self.args,img_path,batch,self.data,mode=mode,rect=bool(self.args.rect),stride=gs)

    def get_dataset(self):
        import yaml
        data=yaml.safe_load(Path(self.args.data).read_text(encoding='utf-8'))
        require('test' not in data and data.get('train') and data.get('val'),'Train/dev only YAML required')
        return super().get_dataset()

    def _ready(self,trainer):
        for k,v in self.expected.items():
            actual=getattr(self.args,k,None)
            require(str(actual)==str(v) if k=='device' else actual==v,'Resolved argument changed: '+k)
        require(self.args.epochs==200 and self.args.patience==200,'Schedule horizon drift')
        require(self.batch_size==self.expected['batch'] and self.train_loader.batch_size==self.expected['batch'] and self.test_loader.batch_size==2*self.expected['batch'],'Loader batch drift')
        require(not self.amp and not self.scaler.is_enabled() and not self.args.resume and self.start_epoch==0,'FP32 cold start drift')
        require(self.expected['batch'] in [4,16,32] and self.accumulate==64//self.expected['batch'] and self.args.nbs==64,'Nominal accumulation drift')
        require(isinstance(self.optimizer,torch.optim.SGD) and self.args.lr0==.01 and self.args.cos_lr,'Optimizer drift')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        require(torch.are_deterministic_algorithms_enabled(),'Deterministic policy drift')
        require(self.train_loader.dataset.rect is False and self.test_loader.dataset.rect is False,'Square loader policy drift')
        head=self.model.model[-1]
        require(self.model.end2end and head.end2end and head.nc==self.data['nc'] and head.nc in [6,10],'Native dataset one2one head drift')
        finite(self.model.state_dict(),'initial_model')
        model_sha=digest({k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in self.model.state_dict().items()})
        ema_sha=digest({k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in self.ema.ema.state_dict().items()})
        require(model_sha==ema_sha,'Initial model and EMA differ')
        for part,loader in [('train',self.train_loader),('dev',self.test_loader)]:
            expected=readj(self.artifact_dir/'loader_ids.json')[part]
            require(sorted(Path(p).stem for p in loader.dataset.im_files)==sorted(expected),'Loader IDs differ: '+part)
            for label in loader.dataset.labels:
                sid=Path(label['im_file']).stem
                original=np.loadtxt(self.artifact_dir/'view/labels'/part/(sid+'.txt'),dtype=np.float32,ndmin=2)
                decoded=np.concatenate([label['cls'],label['bboxes']],axis=1)
                require(original.shape==decoded.shape,'Native loader removed annotations: '+sid)
                order=lambda x:np.array(sorted(map(tuple,x.tolist())),np.float32)
                require(np.allclose(order(original),order(decoded),atol=1e-6,rtol=0),'Native bbox conversion drift: '+sid)
        put(self.artifact_dir/'ACTUAL_TRAINING_SETUP.json',{'epochs':self.args.epochs,'patience':self.args.patience,
            'train_batch':self.train_loader.batch_size,'dev_batch':self.test_loader.batch_size,'nbs':64,
            'nominal_accumulation':64//self.expected['batch'],'warmup_accumulation':'native ramp from1 to nbs/batch; 200epoch horizon preserved',
            'amp':self.amp,'TF32_matmul':torch.backends.cuda.matmul.allow_tf32,'TF32_cudnn':torch.backends.cudnn.allow_tf32,
            'initial_model_sha':model_sha,'initial_EMA_sha':ema_sha,'head':'one2one','reg_max':head.reg_max,
            'optimizer':type(self.optimizer).__name__,'lf':{str(i):float(self.lf(i)) for i in [0,1,50,100,190,200]},
            'loader_membership_and_annotations_verified':True,'test_access':False,'stop_after_completed_epochs':self.smoke_epochs})
        put(self.artifact_dir/'resolved_args.json',vars(self.args))

    def _batch_start(self,trainer):
        require(self.args.epochs==200 and self.args.batch==self.expected['batch'] and not self.args.amp,'Runtime policy drift')
        require(not (WORK/'STOP.json').exists(),'STOP requested')

    def _batch_end(self,trainer):
        require(self._oom_retries==0,'Hidden OOM retry')
        finite(self.loss,'loss');finite(self.loss_items,'loss_items');self.microbatches+=1

    def optimizer_step(self):
        require(not self.scaler.is_enabled(),'FP32 scaler enabled')
        self.scaler.unscale_(self.optimizer)
        finite({n:p.grad for n,p in self.model.named_parameters() if p.grad is not None},'gradients')
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=10.0,error_if_nonfinite=True)
        finite(norm,'gradient_norm')
        self.scaler.step(self.optimizer);self.scaler.update();self.optimizer.zero_grad()
        if self.ema:self.ema.update(self.model)
        self.optimizer_updates+=1

    def _handle_nan_recovery(self,epoch):
        finite(self.loss,'loss');finite(self.metrics,'dev_metrics');finite(self.fitness,'dev_fitness')
        require(self.nan_recovery_attempts==0 and self.optimizer_updates>0,'Recovery or no real optimizer update')
        return False

    def _load_checkpoint_state(self,checkpoint):raise Stop('Automatic checkpoint recovery is forbidden')

    def save_model(self):
        finite(self.model.state_dict(),'model');finite(self.ema.ema.state_dict(),'EMA');finite(self.optimizer.state_dict(),'optimizer')
        require(abs(float(self.fitness)-float(self.metrics['metrics/mAP50-95(B)']))<1e-10,'Unexpected best metric')
        payload={'epoch':self.epoch,'best_fitness':self.best_fitness,'model':None,'ema':copy.deepcopy(unwrap_model(self.ema.ema)).float(),
            'raw_model_state_dict':{k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()},'updates':self.ema.updates,
            'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),'scaler':self.scaler.state_dict(),
            'train_args':vars(self.args),'train_metrics':{**self.metrics,'fitness':self.fitness},'train_results':self.read_results_csv(),
            'version':'8.4.90','automatic_resume_permitted':False,'technical_only':self.smoke_epochs is not None}
        tmp=self.last.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,self.last)
        if self.best_fitness==self.fitness:
            import shutil
            shutil.copyfile(self.last,self.best);self.best_epoch=self.epoch+1
        self.records.append({'epoch':self.epoch+1,'metrics':{k:float(v) for k,v in self.metrics.items()},'fitness':float(self.fitness),
                             'optimizer_updates':self.optimizer_updates,'microbatches':self.microbatches})
        put(self.artifact_dir/'EPOCH_RECORDS.json',self.records,exclusive=False)
        return True

    def _epoch_start(self,trainer):
        self._epoch_started=time.perf_counter()

    def _epoch_end(self,trainer):
        put(self.artifact_dir/'PROGRESS.json',{'completed_epochs':self.epoch+1,'schedule_epochs':200,'epoch_seconds':time.perf_counter()-self._epoch_started,
            'optimizer_updates':self.optimizer_updates,'microbatches':self.microbatches,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'technical_only':self.smoke_epochs is not None},exclusive=False)
        if self.smoke_epochs is not None and self.epoch+1>=self.smoke_epochs:self.stop=True

    def final_eval(self):
        completed=self.smoke_epochs if self.smoke_epochs is not None else 200
        require(len(self.records)==completed and self.best.exists() and self.last.exists(),'Unexpected epoch/checkpoint completion')
        require(self.optimizer_updates>0,'No optimizer update')
        put(self.artifact_dir/'TRAINING_COMPLETE.json',{'completed_epochs':len(self.records),'schedule_epochs':200,
            'optimizer_updates':self.optimizer_updates,'microbatches':self.microbatches,'best_epoch':self.best_epoch,
            'best_sha':sha(self.best),'last_sha':sha(self.last),'technical_only':self.smoke_epochs is not None,'test_inference':False})
