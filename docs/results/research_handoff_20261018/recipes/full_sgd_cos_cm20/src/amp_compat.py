"""YOLO26 AMP validation compares anchor-aligned logits, not discontinuous top-k ordering."""
from core import *
import torch
def install():
 import trainer_adapter
 original=trainer_adapter.amp_check
 def check(model):
  if not getattr(model,'end2end',False):return original(model)
  rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all();was=model.training
  buffers={n:b.detach().clone() for n,b in model.named_buffers()}
  result={'model':'YOLO26','comparison':'anchor_aligned_raw_one2many_and_one2one','rtol':.1,'atol':.5,'max_relative_L2':.02,'tensors':{}}
  try:
   model.eval();x=torch.zeros((1,3,640,640),device='cuda',dtype=torch.float32)
   with torch.inference_mode():
    fp=model(x)
    with torch.autocast('cuda',enabled=True):mixed=model(x)
   trainer_adapter.finite(fp,'AMP_FP32');trainer_adapter.finite(mixed,'AMP_FP16')
   result['postprocessed_direct_allclose']=bool(torch.allclose(fp[0],mixed[0].float(),rtol=.1,atol=.5))
   for branch in ['one2many','one2one']:
    for key in ['boxes','scores']:
     a,b=fp[1][branch][key],mixed[1][branch][key].float()
     require(a.shape==b.shape,'AMP dense shape mismatch')
     delta=a-b;relative=float(delta.norm()/a.norm().clamp_min(1e-8))
     result['tensors'][branch+'/'+key]={'shape':list(a.shape),'max_abs':float(delta.abs().max()),'relative_L2':relative}
     require(torch.allclose(a,b,rtol=.1,atol=.5) and relative<=.02,'AMP dense numerical mismatch '+branch+'/'+key)
   require(all(torch.equal(buffers[n],b) for n,b in model.named_buffers()),'AMP probe modified registered buffers')
  finally:
   model.train(was);torch.set_rng_state(rng);torch.cuda.set_rng_state_all(gpu_rng)
  require(torch.equal(torch.get_rng_state(),rng) and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),gpu_rng)),'AMP RNG restore failed')
  result.update(status='PASS',RNG_restored=True,buffers_unchanged=True,mode_restored=model.training==was)
  put(STATE/'AMP_COMPAT_PROBE.json',result)
  return True
 trainer_adapter.amp_check=check
