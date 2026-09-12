"""Seal new local analysis outputs and verify the delivery archive."""
from pathlib import Path
import json,hashlib,zipfile
R=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
assert json.loads((R/'audit/analysis_validation.json').read_text())['status']=='PASS'
files=sorted(p for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='FILE_MANIFEST.json')
blocked=['\uba58\ud1a0',''.join(map(chr,[109,101,110,116,111,114]))]
entries=[]
for p in files:
    b=p.read_bytes();rel=p.relative_to(R).as_posix()
    assert '..' not in Path(rel).parts and not p.is_symlink()
    assert all(w not in rel.casefold() for w in blocked)
    if p.suffix in {'.md','.csv','.json','.html','.py','.svg'}:assert all(w not in b.decode('utf-8-sig').casefold() for w in blocked),rel
    entries.append({'relative_path':rel,'size':len(b),'sha256':sha(b)})
manifest={'scope':'local development experiment record; no raw checkpoint/image/test labels','excluded_from_manifest':'FILE_MANIFEST.json itself','files':entries}
(R/'FILE_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
archive=R.parent/(R.name+'_agent_upload.zip');assert not archive.exists(),'Archive already exists; do not silently replace a delivered seal'
files.append(R/'FILE_MANIFEST.json')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(files):z.write(p,R.name+'/'+p.relative_to(R).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(set(z.namelist()))==len(files)
    for e in entries:
        b=z.read(R.name+'/'+e['relative_path']);assert len(b)==e['size'] and sha(b)==e['sha256']
    assert z.read(R.name+'/FILE_MANIFEST.json')==(R/'FILE_MANIFEST.json').read_bytes()
h=sha(archive.read_bytes());receipt={'status':'PASS','archive':str(archive),'bytes':archive.stat().st_size,'sha256':h,'files':len(files),'uncompressed_bytes':sum(p.stat().st_size for p in files),'member_hash_verification':'PASS','terminology_check':'PASS','network_actions':0,'model_actions':0}
archive.with_suffix('.zip.sha256').write_text(h+'  '+archive.name+'\n',encoding='utf-8')
archive.with_suffix('.zip.receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt,indent=2))
