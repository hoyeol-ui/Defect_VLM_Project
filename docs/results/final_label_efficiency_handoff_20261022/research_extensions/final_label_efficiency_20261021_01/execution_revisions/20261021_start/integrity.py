from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def verify():
    manifest=json.loads((P/'evidence/execution_freeze.json').read_text(encoding='utf-8'))
    for name,h in manifest['files'].items():assert sha(P/name)==h,'Prepared input/source changed: '+name
    for name,h in manifest['runtime_files'].items():assert sha(Path(name))==h,'Installed numerical source changed: '+name
    return manifest
