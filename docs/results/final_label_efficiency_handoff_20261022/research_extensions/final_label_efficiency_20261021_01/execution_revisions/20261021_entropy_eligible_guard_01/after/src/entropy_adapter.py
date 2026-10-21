"""YOLO26 raw one2one adapter; keep the existing normalized class-score entropy."""
import hashlib
import numpy as np
import torch
from entropy import score

def xyxy_to_xywh(raw):
    raw=raw.copy();xyxy=raw[:4].copy()
    eligible=raw[4:].max(axis=0)>np.float32(.25)
    assert np.all(xyxy[2:,eligible]>=xyxy[:2,eligible]),'Inverted confidence-eligible XYXY box'
    raw[:2]=(xyxy[:2]+xyxy[2:])/np.float32(2)
    raw[2:4]=xyxy[2:]-xyxy[:2]
    return raw

def score_one2one(model, tensor, nc, *, raw_capture=None):
    head=model.model[-1]
    assert head.end2end and head.nc==nc and not head.export
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        output=model(tensor.float())
        assert isinstance(output,tuple) and 'one2one' in output[1]
        branch=output[1]['one2one']
        decoded=head._inference(branch)
        assert decoded.shape==(1,4+nc,8400)
        assert torch.equal(decoded[:,4:],branch['scores'].sigmoid())
        assert torch.isfinite(decoded).all()
        raw=decoded[0].float().cpu().numpy().copy()
    # Capture the original XYXY coordinates and raw anchor order before validation.
    if raw_capture is not None:raw_capture(raw.copy())
    invalid=np.flatnonzero((raw[2]<raw[0])|(raw[3]<raw[1]))
    confidence=raw[4:].max(axis=0)
    geometry={'raw_xyxy_sha':hashlib.sha256(raw.tobytes()).hexdigest(),
              'raw_anchor_count':int(raw.shape[1]),'invalid_raw_box_count':len(invalid),
              'invalid_eligible_box_count':int((confidence[invalid]>np.float32(.25)).sum()),
              'invalid_raw_boxes':[{'anchor_index':int(i),'xyxy':raw[:4,i].tolist(),
                   'class_scores':raw[4:,i].tolist(),'confidence':float(confidence[i]),
                   'confidence_eligible':bool(confidence[i]>np.float32(.25))} for i in invalid]}
    # end2end=True decodes XYXY; arithmetic conversion preserves even unused boxes.
    raw=xyxy_to_xywh(raw)
    result=score(raw,nc)
    result.update(raw_sha=hashlib.sha256(raw.tobytes()).hexdigest(),head='one2one_before_topk',
                  shape=list(raw.shape),coordinate_conversion='xyxy_to_xywh',raw_score_lineage=True,
                  geometry_diagnostics=geometry)
    for cls,confidence,scores in zip(result['class'],result['confidence'],result['raw_class_scores']):
        assert cls==int(np.argmax(scores)) and confidence==max(scores)
    return result

def image_tensor(path):
    import cv2
    from ultralytics.data.augment import LetterBox
    image=cv2.imread(str(path));assert image is not None
    image=LetterBox(new_shape=(640,640),auto=False,scale_fill=False,scaleup=True,center=True,stride=32)(image=image)
    return torch.from_numpy(np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1))).unsqueeze(0).cuda().float()/255
