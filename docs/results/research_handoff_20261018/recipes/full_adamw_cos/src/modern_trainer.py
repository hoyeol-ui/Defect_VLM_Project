"""Preserve verified numerical trainer; add Windows-safe lifecycle observations."""
from trainer_adapter import StrictTrainer,finite
from core import *
import torch,numpy as np
class ModernTrainer(StrictTrainer):
 def __init__(self,*a,**kw):
  self.observation={'train_batches':0,'peak_allocated_bytes':0,'peak_reserved_bytes':0,'amp_probe':'same original model probe; no unrelated model download'}
  super().__init__(*a,**kw)
  self.add_callback('on_train_start',self._start_observation)
  self.add_callback('on_train_batch_end',self._observe)
 def _start_observation(self,trainer):
  # on_train_start precedes self.epoch in strict_loop; never assume it exists.
  put(self.artifact_dir/'LIFECYCLE_START.json',{'phase':'before_epoch','start_epoch':self.start_epoch,'time':time.time(),'epoch_attribute_exists':hasattr(self,'epoch')})
 def _ready(self,trainer):
  super()._ready(trainer)
  require(isinstance(self.optimizer,torch.optim.AdamW),'Authorized recipe requires AdamW')
  require(self.args.lr0==.001 and self.args.cos_lr is True,'AdamW/cosine recipe drift')
  require(abs(self.lf(25)-(.01+.99*(1+np.cos(np.pi*.25))/2))<1e-10,'Actual cosine scheduler drift')
  put(self.artifact_dir/'OPTIMIZER_SCHEDULER_ACTUAL.json',{'optimizer':type(self.optimizer).__name__,'defaults':self.optimizer.defaults,'cos_lr':self.args.cos_lr,'lr0':self.args.lr0,'lrf':self.args.lrf,'lr_multipliers':{str(e):float(self.lf(e)) for e in (0,25,50,75,100)},'nbs':self.args.nbs,'warmup_epochs':self.args.warmup_epochs,'warmup_bias_lr':self.args.warmup_bias_lr})
  require(self.accumulate==8 and self.args.nbs==64,'Accumulation drift')
  require(self.train_loader.dataset.rect is False,'Train padding drift')
  head=self.model.model[-1]
  self.observation.update(model_parameters=sum(p.numel() for p in self.model.parameters()),head_class=type(head).__name__,end2end=bool(getattr(self.model,'end2end',False)),reg_max=getattr(head,'reg_max',None),train_batch=self.train_loader.batch_size,dev_batch=self.test_loader.batch_size,train_rect=self.train_loader.dataset.rect,dev_rect=self.test_loader.dataset.rect,initial_accumulation=self.accumulate)
  put(self.artifact_dir/'MODEL_ACTUAL.json',self.observation)
 def _observe(self,trainer):
  self.observation['train_batches']+=1
  self.observation['peak_allocated_bytes']=max(self.observation['peak_allocated_bytes'],torch.cuda.max_memory_allocated())
  self.observation['peak_reserved_bytes']=max(self.observation['peak_reserved_bytes'],torch.cuda.max_memory_reserved())
  self.observation['criterion_class']=type(getattr(self.model,'criterion',None)).__name__
 def _heartbeat(self,trainer):
  super()._heartbeat(trainer)
  self._observe_memory()
  put(self.artifact_dir/'OBSERVATION.json',self.observation,exclusive=False)
 def _observe_memory(self):
  self.observation['peak_allocated_bytes']=max(self.observation['peak_allocated_bytes'],torch.cuda.max_memory_allocated())
  self.observation['peak_reserved_bytes']=max(self.observation['peak_reserved_bytes'],torch.cuda.max_memory_reserved())
 def final_eval(self):
  super().final_eval();self._observe_memory()
  put(self.artifact_dir/'OBSERVATION.json',self.observation,exclusive=False)
