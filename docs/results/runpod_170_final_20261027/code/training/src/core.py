"""Isolated200epoch campaign IO; no original experiment is edited."""
import hashlib,json,os,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
STATE=Path(os.environ['MODERN_YOLO_RUN']).resolve()
if not STATE.is_relative_to(WORK/'smoke') and not STATE.is_relative_to(WORK/'runs'):raise RuntimeError('Unsafe artifact directory')
class Stop(RuntimeError):pass
def require(ok,message):
    if not ok:raise Stop(message)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def readj(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,obj,exclusive=True):
    p=Path(p).resolve();require(p.is_relative_to(STATE),'Output outside current job');p.parent.mkdir(parents=True,exist_ok=True)
    if exclusive:
        with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
    else:
        tmp=p.with_name(p.name+'.tmp')
        with tmp.open('w',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        os.replace(tmp,p)
