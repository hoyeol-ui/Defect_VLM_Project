"""Prepare one explicitly requested full-train diagnostic; no ML imports or GPU calls."""
from pathlib import Path
import ast,csv,json,hashlib,shutil
P=Path(__file__).resolve().parent;BASE=P.parents[1]
DINO=BASE/'research_extensions/yolo26s_sgd_20261017_01'
RANDOM=BASE/'research_extensions/random_dino_26s_640_PREPARED_20261017_01'
HIST=BASE/'output/final_main_results_retrieval_20260922/metadata_verified/final_main_v2_20260922'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def put(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
 for name in ('src','config','assets','evidence','runs'):(P/name).mkdir(exist_ok=False)
 index=read(HIST/'inputs/GC10-DET/train_images.json');ids=sorted(r['id'] for r in index)
 assert len(ids)==len(set(ids))==1604
 dino_ids=read(DINO/'datasets/GC10-DET/loader_ids.json');dev=dino_ids['dev']
 assert len(dev)==343 and set(dino_ids['train'])<=set(ids) and not set(ids)&set(dev)
 labelroot=BASE/'output/annotation_patch_v2_0_2_20260919/payload/annotation_overlays/ANNOTATION_PATCH_v2_0_2/labels/GC10-DET/train'
 audit=BASE/'Defect_AL_Local_Recovery_Kit_20261015/Defect_AL_Local_Recovery_Kit_20261015/out/audit_20261015_01/validation/FILE_HASHES.csv'
 with audit.open(encoding='utf-8-sig',newline='') as f:known={str(Path(r['ActualPath']).resolve()).lower():r['ExpectedSHA256'] for r in csv.DictReader(f)}
 ds=P/'datasets/GC10-DET';receipts=[]
 for typ in ('images','labels'):(ds/typ/'train').mkdir(parents=True,exist_ok=True)
 for row in index:
  sid=row['id'];image=BASE/'output/final_main_runpod_execution_v2_20260919'/row['image'];label=labelroot/(sid+'.txt')
  for src,dst,expected in [(image,ds/'images/train'/image.name,row['image_sha']),(label,ds/'labels/train'/label.name,known[str(label.resolve()).lower()])]:
   assert sha(src)==expected,str(src);shutil.copyfile(src,dst);assert sha(dst)==expected
   receipts.append(dict(source=str(src),copy=str(dst),sha256=expected))
 for typ in ('images','labels'):shutil.copytree(DINO/'datasets/GC10-DET'/typ/'dev',ds/typ/'dev')
 shutil.copyfile(DINO/'datasets/GC10-DET/names.json',ds/'names.json')
 put(ds/'loader_ids.json',dict(train=ids,dev=dev))
 put(ds/'selection_manifest.json',dict(dataset='GC10-DET',method='FULL_TRAIN_DIAGNOSTIC',seed=None,round=None,
  selected_ids=ids,cumulative_ids=ids,query_count=1604,not_active_learning_budget=True,
  split_sha=read(HIST/'config/protocol.json')['datasets']['GC10-DET']['split_sha'],
  source_train_index=str(HIST/'inputs/GC10-DET/train_images.json')))
 for src in (RANDOM/'src').glob('*.py'):shutil.copyfile(src,P/'src'/src.name)
 for name in ('bootstrap.py','windows_job.py'):shutil.copyfile(RANDOM/name,P/name)
 shutil.copyfile(DINO/'config/common_args.json',P/'config/common_args.json')
 shutil.copyfile(DINO/'assets/yolo26s.pt',P/'assets/yolo26s.pt')
 w=(P/'src/worker.py').read_text(encoding='utf-8')
 assert w.count("'acquisition_seed':2026091901")==1
 w=w.replace("'acquisition_seed':2026091901","'acquisition_seed':None,'diagnostic_full_train':True,'not_active_learning_budget':True")
 needle="'partition':'development','technical_only'";assert w.count(needle)==1
 w=w.replace(needle,"'speed_ms_per_image':{k:float(v) for k,v in metrics.speed.items()},'partition':'development','technical_only'")
 (P/'src/worker.py').write_text(w,encoding='utf-8')
 for name in ('core.py','strict_loop.py','trainer_adapter.py','modern_trainer.py','amp_runtime.py','amp_compat.py'):assert sha(P/'src'/name)==sha(DINO/'src'/name)
 assert sha(P/'config/common_args.json')==sha(DINO/'config/common_args.json')
 files={f.relative_to(P).as_posix():sha(f) for d in ('datasets','assets') for f in (P/d).rglob('*') if f.is_file()}
 old=read(DINO/'evidence/INPUT_SEAL.json')['files']
 for key,expected in old.items():
  if key.startswith('datasets/GC10-DET/') and ('/images/' in key or '/labels/' in key):assert files[key]==expected,key
 put(P/'evidence/INPUT_SEAL.json',dict(files=files,downloaded_weights={},test_present=False))
 put(P/'evidence/COPY_RECEIPTS.json',receipts)
 shutil.copyfile(BASE/'research_extensions/yolo26s_sgd_300_20261017_01/failure_core.py',P/'failure_core.py')
 put(P/'PREPARATION.json',dict(status='READY',authorization='Latest user request: same-recipe full1604 train and same dev343 diagnostic, exactly once.',
  train_n=1604,dev_n=343,learner_seed=2026091900,acquisition_seed=None,not_active_learning_budget=True,
  source_train_index=str(HIST/'inputs/GC10-DET/train_images.json'),annotation_version='ANNOTATION_PATCH_v2_0_2',
  existing_DINO561_and_dev343_image_label_hashes_match=True,common_args_bytes_identical=True,numerical_training_sources_identical=True,
  pretrained_sha=sha(P/'assets/yolo26s.pt'),pretrained_matches_DINO=True,DINO_best_used_as_initial_weight=False,
  source_hashes={f.name:sha(f) for f in (P/'src').glob('*.py')},config_sha=sha(P/'config/common_args.json'),
  expected_train_batches_per_epoch=201,baseline_train_batches_per_epoch=71,epochs=100,
  scope='Fixed epoch full-train diagnostic, not dataset ceiling or a label-efficiency claim; extra examples also imply more optimizer updates.',
  estimated_minutes=[90,110],test_evaluation=False,additional_training_runs=0))
 for f in list((P/'src').glob('*.py'))+[P/'bootstrap.py',P/'windows_job.py']:ast.parse(f.read_text(encoding='utf-8'),filename=str(f))
 print(json.dumps(dict(status='READY',train=1604,dev=343,same_recipe=True,same_pretrained=True,test=False),ensure_ascii=False))
if __name__=='__main__':main()
