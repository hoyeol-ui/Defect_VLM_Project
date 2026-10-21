"""Finish post-training technical checks using an already saved smoke checkpoint."""
from pathlib import Path
import os,sys,json,time,hashlib
P=Path(__file__).resolve().parent;B=P.parents[1]
source=Path(sys.argv[1]).resolve()
assert source.parent==P/'smoke'
out=P/'smoke'/('saved_checkpoint_check_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir()
os.environ.update(MODERN_YOLO_RUN=str(out),YOLO_CONFIG_DIR=str(out/'settings'),MPLCONFIGDIR=str(out/'mpl'),
    CUBLAS_WORKSPACE_CONFIG=':4096:8',YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',WANDB_MODE='disabled')
sys.path.insert(0,str(P/'src'))
import torch,ultralytics
from ultralytics import YOLO
from entropy_adapter import score_one2one,image_tensor
from core import put,readj,sha
torch.set_num_threads(4)
assert readj(source/'best_selection.json')['all_epochs']==1 and readj(source/'amp_summary.json')['updates']>0
checkpoint=source/'train/weights/best.pt'
assert sha(checkpoint)==readj(source/'best_selection.json')['best_sha']
req=readj(source/'REQUEST.json');ds=req['dataset'];index=readj(P/'inputs'/ds/'index.json');nc=len(index['names'])
model=YOLO(str(checkpoint)).model.float().cuda().eval();checks=[]
for r in index['train'][:4]:
    x=image_tensor(r['image']);a=score_one2one(model,x,nc);b=score_one2one(model,x,nc);assert a==b
    checks.append({'id':r['id'],'raw_sha':a['raw_sha'],'shape':a['shape'],'detections':a['detections'],'repeat_equal':True})
del model;torch.cuda.empty_cache()
import ultralytics.data.utils as du
du.check_font=lambda *args,**kwargs:None
import ultralytics.utils as utils
utils.ONLINE=False
import ultralytics.engine.validator as uv
uv.callbacks.add_integration_callbacks=lambda validator:None
m=YOLO(str(checkpoint)).val(data=str(source/'data.yaml'),split='val',imgsz=640,batch=8,conf=.001,iou=.7,
    max_det=300,device=0,workers=0,quantize=32,rect=False,seed=2026091900,deterministic=True,
    plots=False,save_json=False,project=str(out),name='sealed_dev',exist_ok=False)
assert all(__import__('math').isfinite(v) for v in m.results_dict.values())
put(out/'RESULT.json',{'status':'PASS','technical_only':True,'dataset':ds,'parent_training':str(source),
    'best_sha':sha(checkpoint),'optimizer_updates':readj(source/'amp_summary.json')['updates'],'entropy_checks':checks,
    'dev_metrics_finite':True,'test_evaluated':False,'initial_state_sha':readj(source/'training_backend.json')['initial_model_state_sha']})
print(json.dumps(readj(out/'RESULT.json')))
