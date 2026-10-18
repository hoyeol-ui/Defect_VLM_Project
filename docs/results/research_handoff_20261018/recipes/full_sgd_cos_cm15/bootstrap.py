"""Set all model/cache paths inside the isolated workspace before ML imports."""
from pathlib import Path
import os,sys
P=Path(__file__).resolve().parent
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['GC10-DET','NEU-DET'],required=True);ap.add_argument('--model',choices=['yolov8n','yolo11n','yolo12n','yolo26n','yolo26s'],required=True);ap.add_argument('--phase',choices=['smoke','train'],required=True);ap.add_argument('--run-id',required=True);ap.add_argument('--controller-pid',type=int,required=True)
 a=ap.parse_args();out=P/('smoke' if a.phase=='smoke' else 'runs')/a.run_id
 assert out.resolve().parent==P/('smoke' if a.phase=='smoke' else 'runs')
 import time,json
 deadline=time.monotonic()+20
 while not (out/'JOB_READY.json').exists():
  if time.monotonic()>deadline:raise RuntimeError('No owned Windows Job permit')
  time.sleep(.05)
 permit=json.loads((out/'JOB_READY.json').read_text())
 assert permit['worker_pid']==os.getpid() and permit['controller_pid']==os.getppid()==a.controller_pid
 os.environ.update(MODERN_YOLO_RUN=str(out),YOLO_CONFIG_DIR=str(out/'settings'),MPLCONFIGDIR=str(out/'mpl'),TORCH_HOME=str(P/'assets/torch_cache'),HF_HOME=str(P/'assets/hf_cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',WANDB_MODE='disabled',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true')
 sys.dont_write_bytecode=True;sys.path.insert(0,str(P/'src'))
 if a.model.startswith('yolo26'):
  from amp_compat import install
  install()
 from worker import run
 run(a)
if __name__=='__main__':main()
