import collections

def ious(a,b):
 import numpy as np
 a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
 lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:]);wh=np.maximum(hi-lo,0);inter=wh[:,:,0]*wh[:,:,1]
 aa=np.maximum(a[:,2]-a[:,0],0)*np.maximum(a[:,3]-a[:,1],0);bb=np.maximum(b[:,2]-b[:,0],0)*np.maximum(b[:,3]-b[:,1],0)
 return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-12)

def failures(gt,preds,names):
 # Operational one-to-one confidence-greedy matching at IoU .50, conf .25.
 import numpy as np
 counts={c:collections.Counter() for c in range(len(names))};allrows=[]
 for sid,gg in gt.items():
  raw=preds.get(sid,[]);pp=sorted([v for v in raw if v['conf']>=.25],key=lambda v:-v['conf'])
  mat=ious([g['box'] for g in gg],[v['box'] for v in pp]);low=[v for v in raw if v['conf']<.25];lm=ious([g['box'] for g in gg],[v['box'] for v in low]);used=set();pairs={}
  for k,v in enumerate(pp):
   candidates=[g for g in range(len(gg)) if g not in used and gg[g]['cls']==v['cls'] and mat[g,k]>=.5]
   if candidates:
    g=max(candidates,key=lambda a:mat[a,k]);used.add(g);pairs[g]=k
  for g,v in enumerate(gg):
   c=v['cls'];counts[c]['GT_boxes']+=1
   if g in pairs:
    overlap=float(mat[g,pairs[g]]);kind='TP75' if overlap>=.75 else 'TP50_only';counts[c]['TP50']+=1;counts[c]['TP75']+=int(overlap>=.75)
   else:
    wrong=any(mat[g,k]>=.5 and q['cls']!=c for k,q in enumerate(pp));loc=any(.1<=mat[g,k]<.5 and q['cls']==c for k,q in enumerate(pp));weak=any(lm[g,k]>=.5 and q['cls']==c for k,q in enumerate(low))
    kind='FN_wrong_class_overlap' if wrong else 'FN_localization_0.1_to_0.5' if loc else 'FN_low_confidence' if weak else 'FN_no_matching_detection'
    counts[c][kind]+=1;overlap=float(mat[g].max()) if pp else 0.
   allrows.append(dict(id=sid,kind=kind,gt_class=names[c],gt_index=g,max_or_matched_iou=overlap))
  matched_preds=set(pairs.values())
  for k,v in enumerate(pp):
   if k in matched_preds:continue
   c=v['cls'];same=[g for g in range(len(gg)) if gg[g]['cls']==c];other=[g for g in range(len(gg)) if gg[g]['cls']!=c]
   kind='FP_duplicate_or_competing' if any(mat[g,k]>=.5 for g in same) else 'FP_wrong_class_overlap' if any(mat[g,k]>=.5 for g in other) else 'FP_localization_0.1_to_0.5' if any(mat[g,k]>=.1 for g in same) else 'FP_no_GT_overlap'
   counts[c][kind]+=1;allrows.append(dict(id=sid,kind=kind,gt_class=names[c],gt_index=-1,max_or_matched_iou=float(mat[:,k].max()) if gg else 0.))
 return counts,allrows
