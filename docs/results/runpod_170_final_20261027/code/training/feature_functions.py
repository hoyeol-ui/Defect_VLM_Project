"""Original numerical functions with portable read-only configuration imports."""
import hashlib
import numpy as np
from runtime import ROOT,sha,require,read
def config():return {"features":read(ROOT/"config/FEATURE_SPEC.json")}

def geometry(w,h):
    require(w>0 and h>0,'Invalid image geometry');s=max(w,h)
    rw=max(1,(w*518+s//2)//s);rh=max(1,(h*518+s//2)//s)
    return rw,rh,(518-rw)//2,(518-rh)//2

def render(path):
    import torch
    from PIL import Image
    from torchvision.transforms import functional as TF,InterpolationMode
    with Image.open(path) as im:t=TF.pil_to_tensor(im.convert('RGB'))
    _,h,w=t.shape;rw,rh,left,top=geometry(w,h);resized=TF.resize(t,[rh,rw],interpolation=InterpolationMode.BICUBIC,antialias=True)
    require(resized.dtype==torch.uint8,'Renderer must resize uint8')
    canvas=torch.tensor([124,116,104],dtype=torch.uint8)[:,None,None].expand(3,518,518).clone();canvas[:,top:top+rh,left:left+rw]=resized
    c=config()['features'];tensor=(canvas.float()/255-torch.tensor(c['mean'])[:,None,None])/torch.tensor(c['std'])[:,None,None]
    require(tuple(tensor.shape)==(3,518,518),'F518 tensor shape')
    if top:require(torch.equal(canvas[:,:top,:],torch.tensor([124,116,104],dtype=torch.uint8)[:,None,None].expand(3,top,518)),'Padding mismatch')
    return tensor,{'width':w,'height':h,'resized_width':rw,'resized_height':rh,'left':left,'top':top,'right':518-rw-left,'bottom':518-rh-top,'crop':False,'tensor_sha':hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}

def load_model():
    import torch
    from transformers import Dinov2Model
    c=config()['features'];require(sha(ROOT/'assets/dinov2-small/model.safetensors')==c['weight_sha'],'DINO weight SHA')
    torch.manual_seed(2026091900);torch.cuda.manual_seed_all(2026091900);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    model=Dinov2Model.from_pretrained(ROOT/'assets/dinov2-small',local_files_only=True,attn_implementation='eager').float().cuda().eval();model.requires_grad_(False)
    require(model.config.hidden_size==384 and model.config.patch_size==14,'DINO architecture mismatch');return model

def infer(model,path):
    import torch
    tensor,g=render(path)
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        o=model(pixel_values=tensor.unsqueeze(0).cuda());require(tuple(o.last_hidden_state.shape)==(1,1370,384),'DINO token geometry')
        require(torch.equal(o.pooler_output,o.last_hidden_state[:,0]),'CLS/pooler discrepancy')
        z=o.pooler_output;require(torch.isfinite(z).all() and torch.linalg.vector_norm(z)>0,'Nonfinite/zero feature');z=torch.nn.functional.normalize(z,p=2,dim=1).cpu().numpy()[0]
    return z,g
