from pathlib import Path
import os, sys, json, hashlib, csv
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
PACKAGE = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('VLM_OUTPUT_ROOT', str(PACKAGE)))
DEV = Path(os.environ.get('VLM_SOURCE_ROOT', str(PACKAGE.parents[1])))
ROOT = Path(os.environ.get('VLM_WORKSPACE_ROOT', str(DEV.parent)))
PRIVATE = Path(os.environ.get('VLM_INPUT_ROOT', str(DEV.parent / 'private_inputs')))
DATA = PRIVATE / 'data'
FROZEN = PRIVATE / 'frozen'
SNAPSHOT = Path(os.environ.get('VLM_DINO_SNAPSHOT', str(PRIVATE / 'models/dinov2-small/ed25f3a31f01632728cabb09d1542f84ab7b0056')))
REGISTRY = FROZEN / 'confirmatory_v1/data_freeze/private_source_registry.csv'
if not REGISTRY.exists():
    REGISTRY = PACKAGE / 'tables/verified_pool_records.csv'
sys.path.insert(0, str(DEV / 'confirmatory_v1/src'))
from confirmatory_v1 import protocol_core as core
from confirmatory_v1 import frozen_dino_ff_v1 as adapter
import numpy as np
import pandas as pd
from PIL import Image
DATASETS = ['GC10-DET','NEU-DET']
SEEDS = list(range(2026091501,2026091506))
PREFIXES = ['001a46d9','2f98a5f1','017eec02','709d85cb']
GC_NAMES = ['punching_hole','welding_line','crescent_gap','water_spot','oil_spot','silk_spot','inclusion','rolled_pit','crease','waist_folding']
for name in ['checks','tables','figures','features','selections','reports']:
    (OUT/name).mkdir(parents=True,exist_ok=True)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(1024*1024), b''): h.update(c)
    return h.hexdigest()
def dump(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def savecsv(name,rows):
    pd.DataFrame(rows).to_csv(OUT/'tables'/name,index=False,encoding='utf-8-sig')
def registry():
    f=pd.read_csv(REGISTRY)
    return f[f.split=='acquisition'].sort_values(['dataset','sample_id']).reset_index(drop=True)
def cache(ds):
    p=FROZEN/f'confirmatory_v1_1/cache_seals/{ds}/cache_A'
    if not (p/'embeddings.npy').exists():
        p=PACKAGE/'features'/ds
        return np.load(p/'sample_ids.npy',allow_pickle=False).astype(str),np.load(p/'C224.npy',allow_pickle=False),p
    ids=np.load(p/'sample_ids.npy',allow_pickle=False).astype(str)
    x=np.load(p/'embeddings.npy',allow_pickle=False)
    return ids,x,p
def rank_ids(ids, purpose):
    return sorted(ids,key=lambda sid:hashlib.sha256((purpose+'|'+sid).encode()).hexdigest())
def geometry(w,h,mode):
    if mode=='C224':
        # Installed BitImageProcessor TorchvisionBackend: integer truncation.
        if w<=h: nw,nh=256,int(256*h/w)
        else: nw,nh=int(256*w/h),256
        return dict(nw=nw,nh=nh,left=-((nw-224)//2),top=-((nh-224)//2),size=224)
    s=int(mode[1:])
    # Fixed half-up rational arithmetic; longest side exactly s.
    den=max(w,h)
    nw=max(1,(2*w*s+den)//(2*den)); nh=max(1,(2*h*s+den)//(2*den))
    return dict(nw=nw,nh=nh,left=(s-nw)//2,top=(s-nh)//2,size=s)
def render_input(im,mode):
    g=geometry(*im.size,mode)
    import torch
    from torchvision.transforms.v2 import functional as tvF
    tensor=torch.from_numpy(np.array(im,dtype=np.uint8,copy=True)).permute(2,0,1)
    resized=tvF.resize(tensor,[g['nh'],g['nw']],interpolation=tvF.InterpolationMode.BICUBIC,antialias=True)
    rs=Image.fromarray(resized.permute(1,2,0).numpy())
    if mode=='C224':
        return rs.crop((-g['left'],-g['top'],-g['left']+224,-g['top']+224))
    canvas=Image.new('RGB',(g['size'],g['size']),(124,116,104)); canvas.paste(rs,(g['left'],g['top']))
    return canvas
def configure_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':['Malgun Gothic','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'figure.dpi':120,'savefig.dpi':170})
    return plt
def savefig(fig,name):
    for ext in ['png','svg']:
        fig.savefig(OUT/'figures'/f'{name}.{ext}',bbox_inches='tight',facecolor='white')
