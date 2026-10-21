"""RUNTIME_PATCH_v2_0_1: preserve native GradScaler semantics and finite-state guards."""
from core import *
import math
import torch

def guarded_step(trainer,finite):
    t=trainer
    finite(t.loss,'loss_before_optimizer');finite(t.loss_items,'loss_items_before_optimizer')
    finite(t.model.state_dict(),'model_before_optimizer');finite(t.optimizer.state_dict(),'optimizer_before_step')
    if t.ema:finite(t.ema.ema.state_dict(),'EMA_before_step')
    require(t.amp is True and t.scaler.is_enabled(),'AMP must remain enabled')
    scale_before=t.scaler.get_scale();require(math.isfinite(scale_before) and scale_before>0,'Invalid AMP scale')
    t.scaler.unscale_(t.optimizer) # exactly once, after accumulated scaled backward
    found={str(device):float(value) for device,value in t.scaler._found_inf_per_device(t.optimizer).items()}
    require(found and all(v in (0.,1.) for v in found.values()),'Invalid GradScaler found_inf state')
    overflow=any(found.values());params=[p for p in t.model.parameters() if p.grad is not None]
    require(params,'No gradients at optimizer boundary')
    grad_norm=None;before=None
    if overflow:
        # Unscale cannot turn an already overflowed Inf/NaN back into a finite gradient.
        # Native scaler skips the optimizer update; retain all finite-state checks.
        before=[p.detach().clone() for p in t.model.parameters()]
    else:
        finite({n:p.grad for n,p in t.model.named_parameters() if p.grad is not None},'unscaled_gradient')
        norm=torch.nn.utils.clip_grad_norm_(t.model.parameters(),max_norm=10.0,error_if_nonfinite=True)
        finite(norm,'gradient_norm');grad_norm=float(norm)
        finite({n:p.grad for n,p in t.model.named_parameters() if p.grad is not None},'clipped_gradient')
    actual_steps=[]
    handle=t.optimizer.register_step_post_hook(lambda *args,**kwargs:actual_steps.append(True))
    try:t.scaler.step(t.optimizer)
    finally:handle.remove()
    require(len(actual_steps)==int(not overflow),'GradScaler optimizer step/skip mismatch')
    t.scaler.update();scale_after=t.scaler.get_scale()
    require(math.isfinite(scale_after) and scale_after>0,'Nonfinite/zero AMP scale')
    if overflow:
        require(scale_after==scale_before*t.scaler.get_backoff_factor(),'Unexpected scaler backoff')
        require(all(torch.equal(a,b) for a,b in zip(before,t.model.parameters())),'Overflow skip mutated parameters')
    finite(t.model.state_dict(),'model_after_step');finite(t.optimizer.state_dict(),'optimizer_after_step')
    t.optimizer.zero_grad()
    # Match upstream EMA behavior, including on scaler-skipped steps.
    if t.ema:t.ema.update(t.model);finite(t.ema.ema.state_dict(),'EMA_after_step')
    stats=getattr(t,'amp_stats',{'optimizer_boundaries':0,'updates':0,'overflow_skips':0,'by_epoch':{}})
    stats['optimizer_boundaries']+=1;stats['updates']+=len(actual_steps);stats['overflow_skips']+=int(overflow)
    epoch=str(t.epoch+1);e=stats['by_epoch'].setdefault(epoch,{'updates':0,'overflow_skips':0})
    e['updates']+=len(actual_steps);e['overflow_skips']+=int(overflow);t.amp_stats=stats
    row={'time':time.time(),'epoch':t.epoch+1,'boundary':stats['optimizer_boundaries'],'AMP':True,'scale_before':scale_before,'found_inf':found,'optimizer_update':bool(actual_steps),'scale_after':scale_after,'unscaled_grad_norm':grad_norm,'loss':float(t.loss.detach()),'finite_training_state':True,'native_nan_recovery_attempts':t.nan_recovery_attempts}
    with (t.artifact_dir/'amp_events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    if overflow:print('AMP_DYNAMIC_SCALE_SKIP '+json.dumps(row),flush=True)
    return row
