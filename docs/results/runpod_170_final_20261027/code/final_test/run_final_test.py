"""New170 final-test controller. No resume, retraining, fallback or Pod actions."""
import argparse
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *

def gpu_readiness():
    # Reuse the calibrated collector/classifier unchanged, save new evidence here.
    sys.path.insert(0,str(O))
    from gpu_guard_revision_20261023_02 import gpu_guard as guard
    calibrated=guard.calibration();snapshots=[]
    for i in range(2):
        snapshot=guard.collector.collect();snapshot['classification']=guard.classify(snapshot,calibrated);snapshots.append(snapshot)
        if not i:time.sleep(1)
    stable=guard.identity(snapshots[0])==guard.identity(snapshots[1])
    allowed=stable and all(x['classification']['decision']=='ALLOW' for x in snapshots)
    out={'time_utc':now(),'decision':'ALLOW' if allowed else 'BLOCK','process_and_module_lists_stable':stable,'calibration':calibrated,'snapshots':snapshots,'Windows_admin':bool(ctypes.windll.shell32.IsUserAnAdmin()),'CUDA_context_created':False,'test_inference_performed':False}
    path=E/('GPU_PREFLIGHT_'+str(time.time_ns())+'.json');put(path,out)
    reasons=sorted({r for s in snapshots for r in s['classification']['reasons']})
    print('LOCAL_GPU_PREFLIGHT',out['decision'],path,flush=True)
    if reasons:print('BLOCK_REASONS',json.dumps(reasons),flush=True)
    require(allowed,'GPU readiness unknown/blocked. No inference or execution reservation. Evidence: '+str(path))
    return out,path

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',action='store_true');ap.add_argument('--gpu-check-only',action='store_true');args=ap.parse_args()
    require(Path(sys.executable).resolve()==PYTHON.resolve(),'Use the fixed local Python executable')
    if args.gpu_check_only:
        gpu_readiness();return
    require((E/'LOCAL_PREFLIGHT_PASS.json').exists(),'Verified local backup/package not ready')
    verify_seal();models=verify_models();verify_inputs()
    require(shutil.disk_usage(ROOT).free>=5*1024**3,'Less than5GiB free for evaluation')
    require(not RUN.exists() and not (ROOT/'EXECUTION_RESERVED.json').exists(),'Existing scientific test attempt preserved; no automatic restart')
    if not args.run:
        print('CPU_PACKAGE_PREFLIGHT_PASS; no inference',flush=True);return
    auth=read(E/'EXECUTION_AUTHORIZATION.json')
    require(auth['pod_stop_authorized'] is False and 'Prepare and run fixed final test locally' in auth['authorized'],'No matching local test authorization')
    require(not GPU_LOCK.exists() and not MUTEX.exists(),'Existing local GPU/evaluation lock; do not clear it')
    readiness,path=gpu_readiness()
    owner={'controller_pid':os.getpid(),'created_utc':now(),'purpose':'FIXED_170_MODEL_FINAL_TEST','root':str(ROOT)}
    gpu_owned=mutex_owned=False;attempt=False
    try:
        put(MUTEX,owner);mutex_owned=True
        put(GPU_LOCK,owner);gpu_owned=True
        put(ROOT/'EXECUTION_RESERVED.json',{**owner,'automatic_retry':False,'package_lock_sha':sha(E/'TEST_PACKAGE_LOCK.json'),'authorization_sha':sha(E/'EXECUTION_AUTHORIZATION.json'),'GPU_preflight':str(path)});attempt=True
        RUN.mkdir(exist_ok=False)
        sys.path.insert(0,str(BASE/'research_extensions/full_data_26s_sgd_cosine_640_20261018_01'))
        from windows_job import WindowsJob
        for i,m in enumerate(models,1):
            out=RUN/'models'/m['cell_id'];out.mkdir(parents=True,exist_ok=False)
            os.environ.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED=str(m['learner_seed']))
            job=WindowsJob()
            try:
                with (out/'console.log').open('xb') as log:
                    child=subprocess.Popen([str(PYTHON),'-I','-X','utf8','-B',str(ROOT/'tools/worker.py'),'--cell',m['cell_id'],'--parent',str(os.getpid())],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    job.assign(child)
                    put(out/'PERMIT.json',{'controller_pid':os.getpid(),'worker_pid':child.pid,'cell_id':m['cell_id'],'output':str(out),'package_lock_sha':sha(E/'TEST_PACKAGE_LOCK.json'),'execution_authorization_sha':sha(E/'EXECUTION_AUTHORIZATION.json')})
                    code=child.wait()
                require(code==0,'Test worker failed: '+m['cell_id']+'; see '+str(out/'console.log'))
                result=read(out/'metrics.json')
                require(result['status']=='PASS' and result['cell_id']==m['cell_id'] and result['model_sha']==m['best_sha'],'Worker result mismatch')
                put(out/'RECEIPT.json',{'metrics_sha':sha(out/'metrics.json'),'cell_id':m['cell_id'],'time_utc':now()})
                print('FINAL_TEST_COMPLETE',i,'/170',m['cell_id'],flush=True)
            finally:job.close()
        verify_models();verify_inputs();verify_seal()
        code=subprocess.call([str(PYTHON),'-X','utf8','-B',str(ROOT/'tools/aggregate.py')])
        require(code==0,'Aggregation failed; preserve raw metrics')
        put(E/'FINAL_TEST_COMPLETE.json',{'status':'170_FIXED_MODELS_TESTED','time_utc':now(),'physical_models':170,'logical_rows':190,'test_images_per_dataset':{'GC10-DET':343,'NEU-DET':270},'summary_sha':sha(ROOT/'results/TEST_SUMMARY.json'),'package_lock_sha':sha(E/'TEST_PACKAGE_LOCK.json'),'Pod_stopped':False,'automatic_retry':False})
        print('ALL170_LOCAL_TEST_COMPLETE. RunPod remains running.',flush=True)
    except BaseException:
        if attempt and not (E/'FINAL_TEST_STOPPED.json').exists():put(E/'FINAL_TEST_STOPPED.json',{'time_utc':now(),'traceback':traceback.format_exc(),'automatic_retry':False,'Pod_stopped':False})
        raise
    finally:
        # Only release markers created by this local controller, never server locks.
        if gpu_owned and read(GPU_LOCK).get('controller_pid')==os.getpid():GPU_LOCK.unlink()
        if mutex_owned and read(MUTEX).get('controller_pid')==os.getpid():MUTEX.unlink()

if __name__=='__main__':
    try:main()
    except BaseException:
        if '--run' in sys.argv and not (ROOT/'EXECUTION_RESERVED.json').exists() and not (E/'FINAL_TEST_NOT_STARTED.json').exists():
            put(E/'FINAL_TEST_NOT_STARTED.json',{'time_utc':now(),'traceback':traceback.format_exc(),'inference_started':False,'automatic_retry':False,'Pod_stopped':False})
        raise
