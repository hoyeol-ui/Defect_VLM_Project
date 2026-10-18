"""Detached owner of exactly one authorized training and its normal final dev evaluation."""
from pathlib import Path
import os,sys,json,csv,hashlib,subprocess,traceback,collections
P=Path(__file__).resolve().parent;B=P.parents[1]
sys.path.insert(0,str(P))
PY=B/'Defect_VLM_Project/.python311/python.exe'
BASELINE=B/'research_extensions/full_data_26s_sgd_640_20261017_01/runs/GC10_FULL1604_yolo26s_SGD_e100_b8_i640'
RUN_ID='GC10_FULL1604_yolo26s_SGD_Cosine_CM20_e100_b8_i640'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def put(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def table(p,rows):
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

from compare_results import summarize

def main():
    prep=read(P/'PREPARATION.json');lock=B/'research_extensions/GPU_TRAINING_ACTIVE.lock'
    assert all(sha(P/'src'/n)==h for n,h in prep['source_hashes'].items())
    assert sha(P/'config/common_args.json')==prep['config_sha']
    assert not (P/'RUN_ONCE.json').exists(),'Already attempted; no retry'
    with lock.open('x',encoding='utf-8') as f:json.dump(dict(controller_pid=os.getpid(),workspace=str(P),run_id=RUN_ID),f)
    proc=None;job=None;out=P/'runs'/RUN_ID
    try:
        with (P/'RUN_ONCE.json').open('x',encoding='utf-8') as f:json.dump(dict(authorization=prep['authorization'],controller_pid=os.getpid()),f)
        out.mkdir(parents=True,exist_ok=False)
        from windows_job import WindowsJob
        job=WindowsJob()
        cmd=[str(PY),'-I','-B',str(P/'bootstrap.py'),'--dataset','GC10-DET','--model','yolo26s','--phase','train','--run-id',RUN_ID,'--controller-pid',str(os.getpid())]
        with (out/'console.log').open('x',encoding='utf-8') as log:
            proc=subprocess.Popen(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1'})
            job.assign(proc);put(out/'JOB_READY.json',dict(controller_pid=os.getpid(),worker_pid=proc.pid))
            put(P/'STATUS.json',dict(status='TRAINING',controller_pid=os.getpid(),worker_pid=proc.pid,output=str(out),epochs=100,train_n=1604,automatic_followup=False))
            code=proc.wait()
        assert code==0,f'Worker failed ({code}); no retry or batch fallback'
        summarize(out)
        put(P/'STATUS.json',dict(status='COMPLETE',output=str(out),report=str(P/'SGD_COSINE_CM10_VS_CM20_KO.md'),additional_experiments=False))
    except BaseException:
        put(P/'FAILURE.json',dict(error=traceback.format_exc(),automatic_retry=False))
        put(P/'STATUS.json',dict(status='FAILED',output=str(out),automatic_retry=False));raise
    finally:
        if job:job.close()
        if proc and proc.poll() is None:proc.kill();proc.wait(timeout=30)
        if lock.exists() and read(lock).get('controller_pid')==os.getpid():lock.unlink()

if __name__=='__main__':main()
