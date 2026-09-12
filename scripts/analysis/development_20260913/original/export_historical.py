"""Read five exact local git blobs. No checkout, network, or source mutations."""
from pathlib import Path
import subprocess,json,hashlib
R=Path(__file__).resolve().parent; repo=R.parent/'Defect_VLM_Project'; dest=R/'historical_geometry';dest.mkdir(exist_ok=True)
branch='research/entropy-a00-audit-20260908';commit=subprocess.check_output(['git','-C',str(repo),'rev-parse',branch]).decode().strip()
root='docs/reviews/dino_class_geometry_posthoc_20260909/'
files=['results/class_compactness.csv','results/selection_vs_geometry.csv','results/spearman_by_round.csv','results/analysis_integrity.json','analysis_config.json']
rows=[]
for rel in files:
    spec=commit+':'+root+rel
    b=subprocess.check_output(['git','-C',str(repo),'show',spec]);p=dest/Path(rel).name;p.write_bytes(b)
    rows.append({'path':root+rel,'file':p.name,'sha256':hashlib.sha256(b).hexdigest(),'git_blob':subprocess.check_output(['git','-C',str(repo),'rev-parse',spec]).decode().strip()})
(dest/'SOURCE_PROVENANCE.json').write_text(json.dumps({'branch':branch,'commit':commit,'files':rows},indent=2))
print('Exported',len(rows),'read-only historical blobs at',commit)
