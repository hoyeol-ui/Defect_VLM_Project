"""Read-only standard-library verification of the published saved results.

No CUDA, ML library, inference, training, network request or filesystem write.
"""
from pathlib import Path
import csv,hashlib,itertools,json,math,statistics as st
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def close(a,b,msg):require(math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11),msg)
def stats(v):return {'mean':st.mean(v),'sample_sd':st.stdev(v),'paired_values':v}
def check_stats(expected,v,msg):
 for k,x in stats(v).items():
  if isinstance(x,list):
   require(len(expected[k])==len(x),msg)
   for a,b in zip(expected[k],x):close(a,b,msg+' values')
  else:close(expected[k],x,msg+' '+k)
def area(xs,ys):return sum((b-a)*(u+v)/2 for a,b,u,v in zip(xs,xs[1:],ys,ys[1:]))/(xs[-1]-xs[0])
def signflip(v):
 target=abs(st.mean(v))
 return sum(abs(st.mean(a*s for a,s in zip(v,z)))>=target-1e-15 for z in itertools.product((-1,1),repeat=len(v)))/(2**len(v))

catalog={}
for line in (ROOT/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
 digest,name=line.split('  ',1);p=ROOT/name
 require(p.resolve().is_relative_to(ROOT.resolve()),'Manifest path leaves export')
 require(p.is_file() and sha(p)==digest,'File integrity: '+name);catalog[name]=digest
actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts}
require(actual==set(catalog),'Missing or uncatalogued public files')
P=load('results/PHYSICAL_MODEL_METRICS.json');L=load('results/LOGICAL_190_ROWS.json');S=load('results/TEST_SUMMARY.json');A=load('results/POSTHOC_ANALYSIS.json');V=load('PROVENANCE.json')
require(len(P)==170 and len(L)==190,'Cardinality')
require(catalog['results/TEST_SUMMARY.json']==V['source_summary_sha256']==A['source_summary_sha'],'Original summary identity')
require(catalog['evidence/MODEL_HASH_AUDIT.json']==V['source_model_audit_sha256'],'Original training audit identity')
by_id={p['model']['cell_id']:p for p in P};require(len(by_id)==170,'Unique physical models')
lookup={(p['model']['dataset'],p['model']['method'],p['model']['budget_percent'],p['model']['pair']):p for p in P}
audit=load('evidence/MODEL_HASH_AUDIT.json');require(audit['checked']==170 and len(audit['models'])==170,'Training artifact audit count')
require({x['cell_id'] for x in audit['models']}==set(by_id),'Training audit cells')
for x in audit['models']:
 m=by_id[x['cell_id']]['model']
 require(x['best_sha']==m['best_sha'] and x['last_sha']==m['last_sha'],'Best/last model hash')
training=load('evidence/TRAINING_COMPLETIONS_170.json');require(len(training)==170,'Training completion count')
for c in training:
 m=by_id[c['cell_id']]['model'];require(c['status']=='PASS' and c['batch']==32 and c['completed_epochs']==c['schedule_epochs']==200 and c['seed']==m['learner_seed'],'Completed training settings')
receipts=load('evidence/TEST_RECEIPTS_170.json');require(len(receipts)==170,'Test receipt count')
for r in receipts:
 cell=r['cell_id'];p=by_id[cell];record=load(r['published_path'])
 require(catalog[r['published_path']]==r['published_metrics_sha256'],'Published metric receipt')
 require(record['status']=='PASS' and record['cell_id']==cell and record['partition']=='test','Fixed test status')
 require(record['overall']==p['metrics'] and record['class_metrics']==p['per_class'],'Physical record consistency')
 require(record['model_sha']==r['model_sha256']==p['model']['best_sha'],'Tested model identity')
 require(record['images_seen']=={'GC10-DET':343,'NEU-DET':270}[p['model']['dataset']],'Fixed test image count')
 close(st.mean(c['AP50_95'] for c in p['per_class']),p['metrics']['metrics/mAP50-95(B)'],'Class AP mean')
 for c in p['per_class']:
  require(len(c['AP_by_IoU'])==10,'Per-class IoU array')
  close(st.mean(c['AP_by_IoU']),c['AP50_95'],'Per-class AP across IoU')
  close(c['AP_by_IoU'][0],c['AP50'],'AP50 entry')
for row in L:
 p=by_id[row['physical_cell']]
 require(p['metrics']==row['metrics'],'Logical/physical score identity')
 require(row['pair']==p['model']['pair'] and row['dataset']==p['model']['dataset'] and row['budget_percent']==p['model']['budget_percent'],'Logical/physical pairing')
for summary in S['summaries']:
 rows=[r for r in L if all(r[k]==summary[k] for k in ['dataset','method','budget_percent'])]
 require(len(rows)==5,'Summary seed count')
 for k,expected in summary['metrics_percent'].items():check_stats(expected,[100*r['metrics'][k] for r in sorted(rows,key=lambda x:x['pair'])],'Mean and SD')
methods=['Random','DINOv2_S14_F518_FF','Entropy_NCS_MEAN_V2'];datasets=['GC10-DET','NEU-DET'];budgets=[10,15,20,25,30,35]
metric='metrics/mAP50-95(B)';curves={}
for ds in datasets:
 for method in methods:
  values=[]
  for pair in range(1,6):
   points=[lookup[(ds,'SHARED_R0' if b==10 else method,b,pair)] for b in budgets]
   values.append(area([p['model']['train_images'] for p in points],[100*p['metrics'][metric] for p in points]))
  curves[(ds,method)]=values
  expected=next(x for x in S['nAULC'] if x['dataset']==ds and x['method']==method)
  check_stats(expected,values,'nAULC')
  ratios=[100*lookup[(ds,method,35,p)]['metrics'][metric]/lookup[(ds,'Full',100,p)]['metrics'][metric] for p in range(1,6)]
  full=A['Full35'][ds+'|'+method]['attainment']
  close(st.mean(ratios),full['mean'],'Paired Full mean');close(st.stdev(ratios),full['sd'],'Paired Full SD')
comparisons=[]
for c in S['paired_comparisons']:
 other=c['comparison'].split(' minus ')[1]
 differences=[a-b for a,b in zip(curves[(c['dataset'],methods[1])],curves[(c['dataset'],other)])]
 check_stats(c['nAULC_difference_percentage_points'],differences,'Paired effects')
 p=signflip(differences);close(p,c['exact_two_sided_sign_flip_p'],'Exact p');comparisons.append((p,c))
running=0
for i,(p,c) in enumerate(sorted(comparisons,key=lambda x:x[0])):
 running=max(running,(4-i)*p);close(min(1,running),c['Holm4_adjusted_p'],'Holm four-comparison family')
for filename,n in [('METRICS_170.csv',170),('CLASS_METRICS_170.csv',1360)]:
 with (ROOT/'results'/filename).open(encoding='utf-8',newline='') as f:require(len(list(csv.DictReader(f)))==n,'CSV rows')
require(catalog['manuscript/03_JKICS_Revised_Manuscript_170models_20261027.docx']==V['manuscript_sha256'],'Verified manuscript identity')
print(json.dumps({'status':'PASS','files_verified':len(catalog),'physical_models':170,'logical_rows':190,'training_artifact_hashes':340,'test_receipts':170,'statistics_recomputed':'means, sample SD, actual-count nAULC, four exact sign-flip tests and Holm, paired Full ratios','new_inference':False},indent=2))
