"""Verify this manuscript revision from saved data; no network, models or writes."""
from pathlib import Path
from zipfile import ZipFile
import hashlib,json,math,re,statistics as st
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[2]
SEALED=REPO/'docs/results/runpod_170_final_20261027'
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,message):
 if not ok:raise RuntimeError(message)
def close(a,b):require(math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-10),'Derived numeric mismatch')
def area(xs,ys):return sum((xs[i+1]-xs[i])*(ys[i+1]+ys[i])/2 for i in range(5))/(xs[-1]-xs[0])
catalog={}
for line in (ROOT/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
 digest,name=line.split('  ',1);path=ROOT/name
 require(path.resolve().is_relative_to(ROOT),'Manifest path escapes revision')
 require(path.is_file() and sha(path)==digest,'Revision file integrity: '+name);catalog[name]=digest
actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts}
require(actual==set(catalog),'Missing or uncatalogued revision file')
V=load(ROOT/'PROVENANCE.json');Q=load(ROOT/'VERIFICATION.json');D=load(ROOT/'DERIVED_ANALYSIS.json')
require(sha(SEALED/'SHA256SUMS')==V['sealed_result_manifest_sha256'],'Original manifest changed')
require(sha(SEALED/'results/TEST_SUMMARY.json')==V['source_test_summary_sha256']==Q['test_summary_sha256'],'Original final summary changed')
require(sha(SEALED/'results/POSTHOC_ANALYSIS.json')==V['source_posthoc_analysis_sha256'],'Saved analysis changed')
man=ROOT/V['manuscript'];previous=ROOT/V['previous_manuscript']
require(sha(man)==V['manuscript_sha256']==Q['output_sha256'],'Published DOCX differs from verified local revision')
require(sha(previous)==V['previous_manuscript_sha256']==Q['source_sha256'],'Previous manuscript identity')
P=load(SEALED/'results/PHYSICAL_MODEL_METRICS.json');S=load(SEALED/'results/TEST_SUMMARY.json')
require(len(P)==170 and S['physical_models']==170 and S['logical_rows']==190,'Model cardinality')
index={(x['model']['dataset'],x['model']['method'],x['model']['pair'],x['model']['round']):x for x in P}
M=['Random','DINOv2_S14_F518_FF','Entropy_NCS_MEAN_V2'];DS=['GC10-DET','NEU-DET']
def rec(ds,m,p,r):return index[(ds,'SHARED_R0' if r==0 else m,p,r)]
mins={};fulls={};class_deltas={}
for ds in DS:
 for m in M:
  vals=[];full=[]
  for pair in range(1,6):
   rs=[rec(ds,m,pair,r) for r in range(6)]
   vals.append(area([x['model']['train_images'] for x in rs],[100*min(c['AP50_95'] for c in x['per_class']) for x in rs]))
   full.append(100*rs[-1]['metrics']['metrics/mAP50-95(B)']/index[(ds,'Full',pair,None)]['metrics']['metrics/mAP50-95(B)'])
  mins[ds+'|'+m]=vals;fulls[ds+'|'+m]=full
  require(sum(x>=95 for x in full)==D['Full95_counts'][ds+'|'+m],'Full95 count')
 differences=[d-e for d,e in zip(mins[ds+'|'+M[1]],mins[ds+'|'+M[2]])]
 expected=D['minimum_class_DINO_minus_Entropy'][ds]
 close(st.mean(differences),expected['mean']);close(st.stdev(differences),expected['sd'])
 for x,y in zip(differences,expected['paired_values']):close(x,y)
 names=[c['name'] for c in rec(ds,M[0],1,5)['per_class']]
 for name in names:
  values=[]
  for pair in range(1,6):
   r=next(c for c in rec(ds,M[0],pair,5)['per_class'] if c['name']==name)['AP50_95']
   d=next(c for c in rec(ds,M[1],pair,5)['per_class'] if c['name']==name)['AP50_95']
   values.append(100*(d-r))
  class_deltas[ds+'|'+name]=values
  close(st.mean(values)/len(names),D['class_decomposition'][ds][name])
 overall=st.mean(100*(rec(ds,M[1],p,5)['metrics']['metrics/mAP50-95(B)']-rec(ds,M[0],p,5)['metrics']['metrics/mAP50-95(B)']) for p in range(1,6))
 close(sum(D['class_decomposition'][ds].values()),overall)
require(len(class_deltas)==16,'All 16 classes')

NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
with ZipFile(man) as z,ZipFile(previous) as old:
 root=ET.fromstring(z.read('word/document.xml'));r0=ET.fromstring(old.read('word/document.xml'))
 equations=lambda r:[ET.tostring(x) for x in r.findall('.//m:oMath',NS)]
 require(equations(root)==equations(r0) and len(equations(root))==4,'Four original native equations')
 media=[n for n in old.namelist() if n.startswith('word/media/')]
 require(len(media)==5 and all(z.read(n)==old.read(n) for n in media),'Five original result figures')
 require(z.read('word/media/image6.png')==(ROOT/'figures/Fig1_Study_Architecture.png').read_bytes(),'Architecture image identity')
 require(len(root.findall('.//w:drawing',NS))==6,'Six manuscript figures')
 comments=ET.fromstring(z.read('word/comments.xml'))
 require(len(comments.findall('w:comment',NS))==3,'Three editorial comments')
 md=(ROOT/man.with_suffix('.md').name).read_text(encoding='utf-8')
 require(md.count('$$')==8,'Four web equations')
 for i in range(1,7):require(md.count(f'![Figure {i}](')==1,'Web figure order')
 tables=root.find('w:body',NS).findall('w:tbl',NS)
 require(len(tables)==10,'Nine numerical tables plus front matter')
 for i in [4,7,9]:
  for row in tables[i].findall('w:tr',NS):
   cells=[]
   for cell in row.findall('w:tc',NS):
    celltext=[]
    for p in cell.findall('w:p',NS):
     celltext.append(''.join('\n' if e.tag=='{'+NS['w']+'}br' else (e.text or '') for e in p.iter() if e.tag in ['{'+NS['w']+'}t','{'+NS['w']+'}br']))
    cells.append('\n'.join(celltext).replace('\n','<br>').replace('|','\\|'))
   require('| '+' | '.join(cells)+' |' in md,'Web table matches DOCX table')
 require(len(tables[7].findall('w:tr',NS))==17,'All-class manuscript table')

readme=(REPO/'README.md').read_text(encoding='utf-8')
for ds in DS:
 for m,label in zip(M,['Random','DINOv2-FF','Entropy']):
  overall=next(x for x in S['nAULC'] if x['dataset']==ds and x['method']==m)
  ap=next(x for x in S['summaries'] if x['dataset']==ds and x['method']==m and x['budget_percent']==35)['metrics_percent']['metrics/mAP50-95(B)']
  v=mins[ds+'|'+m]
  row=f"|{ds}|{label}|{overall['mean']:.2f} ± {overall['sample_sd']:.2f}|{st.mean(v):.2f} ± {st.stdev(v):.2f}|{ap['mean']:.2f} ± {ap['sample_sd']:.2f}|"
  require(row in readme,'README primary row')
for ds in DS:
 full=fulls[ds+'|'+M[1]]
 require(f"|{ds}|{st.mean(full):.2f} ± {st.stdev(full):.2f}|{min(full):.2f}–{max(full):.2f}|{sum(v>=95 for v in full)}/5|" in readme,'README Full row')
require(Q['all_pages_visually_checked'] and Q['pages']==9 and not Q['new_training_or_inference'],'Saved visual/numeric QA')
print(json.dumps({'status':'PASS','catalogued_files':len(catalog),'physical_models':170,'classes':16,'figures':6,'native_equations':4,'docx_sha256':sha(man),'readonly_saved_data_verification':True},indent=2))
