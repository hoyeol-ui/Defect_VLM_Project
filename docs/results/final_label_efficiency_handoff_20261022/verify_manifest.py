"""Verify the handoff using Python standard library only."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
rows=json.loads((root/'FILE_MANIFEST.json').read_text(encoding='utf-8'))['files']
bad=[]
for r in rows:
 p=root/r['path']
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:
  bad.append(r['path'])
if bad:
 print('FAIL',*bad,sep='\n');sys.exit(1)
print('PASS:',len(rows),'files; original handoff bytes verified. No research execution.')
