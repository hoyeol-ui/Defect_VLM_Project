"""Research A isolated IO adapter; does not import the original research core."""
import csv,hashlib,json,os,sys,time,socket,contextlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
STATE=Path(os.environ['MODERN_YOLO_RUN']).resolve()
if not STATE.is_relative_to(WORK/'smoke') and not STATE.is_relative_to(WORK/'runs'):raise RuntimeError('Unsafe output root')
class Stop(RuntimeError):pass
def require(condition,message):
 if not condition:raise Stop(message)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def readj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def put(p,x,exclusive=True):
 p=Path(p).resolve();require(p.is_relative_to(STATE),'Write outside current run '+str(p));p.parent.mkdir(parents=True,exist_ok=True)
 if exclusive:
  with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 else:
  tmp=p.with_name(p.name+'.tmp');
  with tmp.open('w',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
  os.replace(tmp,p)
def patch_identity():return sha(ROOT/'amp_runtime.py')
