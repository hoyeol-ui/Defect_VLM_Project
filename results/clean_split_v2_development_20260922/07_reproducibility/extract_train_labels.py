"""Extract the ACTIVE annotation-patch train labels, never the base-package labels."""
from pathlib import Path
import hashlib
import json
import tarfile

b=Path.cwd()/'9월'/'9월 21일'
analysis=b/'analysis_20260922'
out=analysis/'final_thesis_analysis'/'07_reproducibility'/'train_labels'
normalization=json.loads((analysis/'evidence/final_main_runpod_execution_v2_20260919/annotation_overlays/ANNOTATION_PATCH_v2_0_2/TRAIN_DEV_NORMALIZATION.json').read_text(encoding='utf8'))
expected={(r['dataset'],r['id']):r['after_label_sha'] for r in normalization['records'] if r['partition']=='train'}
seen=set()
with tarfile.open(b/'FINAL_RESULTS_ONLY_20260922.tar.gz','r|gz') as t:
 for m in t:
  if m.isfile() and '/annotation_overlays/ANNOTATION_PATCH_v2_0_2/labels/' in m.name and '/train/' in m.name and m.name.endswith('.txt'):
   parts=Path(m.name).parts
   dataset=parts[-3]
   sid=Path(parts[-1]).stem
   content=t.extractfile(m).read()
   assert hashlib.sha256(content).hexdigest()==expected[(dataset,sid)]
   p=out/dataset/parts[-1]
   p.parent.mkdir(parents=True,exist_ok=True)
   p.write_bytes(content)
   seen.add((dataset,sid))
assert seen==set(expected)
print('active patched train labels',len(seen))
