"""CPU FP32 stable NMS with raw anchor lineage, FP64 normalized-score entropy."""
import math,time
import numpy as np
from core import require
def score(raw,nc,timeout=60.):
    raw=np.asarray(raw,dtype=np.float32);require(raw.ndim==2 and raw.shape[0]==4+nc,'Unsupported raw head shape')
    require(np.isfinite(raw).all() and nc>1,'Nonfinite raw output')
    scores=raw[4:].T.copy();require(((scores>=0)&(scores<=1)).all(),'Scores outside sigmoid range')
    conf=scores.max(axis=1);cls=scores.argmax(axis=1);eligible=np.flatnonzero(conf>np.float32(.25))
    # Geometry is required only for the unchanged confidence-eligible candidates.
    # Keep all raw anchors and coordinates; do not clamp or compact this array.
    boxes=raw[:4].T.copy();require((boxes[eligible,2:]>=0).all(),'Negative eligible box size')
    boxes[:,:2]-=boxes[:,2:]/2;boxes[:,2:]+=boxes[:,:2]
    order=eligible[np.lexsort((eligible,-conf[eligible]))][:30000];retained=[];start=time.monotonic()
    while len(order) and len(retained)<300:
        require(time.monotonic()-start<=timeout,'NMS timeout')
        i=int(order[0]);retained.append(i);rest=order[1:]
        wh=np.maximum(np.float32(0),np.minimum(boxes[i,2:],boxes[rest,2:])-np.maximum(boxes[i,:2],boxes[rest,:2]));intersection=wh[:,0]*wh[:,1]
        area=(boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1]);rwh=boxes[rest,2:]-boxes[rest,:2];union=area+rwh[:,0]*rwh[:,1]-intersection
        iou=np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)
        order=rest[~((cls[rest]==cls[i])&(iou>np.float32(.7)))]
    kept=np.array(retained,dtype=np.int64);entropies=[]
    for i in kept:
        s=scores[i].astype(np.float64);den=math.fsum(s);require(den>0,'Entropy zero denominator');q=s/den
        h=-math.fsum(float(x)*math.log(float(x)) for x in q if x>0)/math.log(nc)
        require(math.isfinite(h) and -1e-14<=h<=1+1e-14,'Entropy nonfinite/range');entropies.append(min(1.,max(0.,h)))
    uncertainty=math.fsum(entropies)/len(entropies) if entropies else 0.
    return {'uncertainty':uncertainty,'anchor_indices':retained,'boxes_xyxy':boxes[kept].tolist(),'confidence':conf[kept].tolist(),'class':cls[kept].tolist(),'raw_class_scores':scores[kept].tolist(),'box_entropy':entropies,'detections':len(retained),'max_det_reached':len(retained)==300}
