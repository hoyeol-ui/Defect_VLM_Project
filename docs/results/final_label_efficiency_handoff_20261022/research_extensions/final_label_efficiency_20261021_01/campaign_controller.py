"""Sequential, fail-stop owner of the 101 approved new scientific training cells."""
from pathlib import Path
import sys,os,json,csv,time,subprocess,traceback,shutil,datetime,ctypes,statistics,argparse
P=Path(__file__).resolve().parent;B=P.parents[1];C=P/'campaign'
PY=B/'Defect_VLM_Project/.python311/python.exe'
sys.path.insert(0,str(P))
from campaign_lib import read,stamp,kst,cell_id,load_matrix,ordered_new,eta
from integrity import verify,sha

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def event(kind,**data):
    with (C/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(dict(time_UTC=stamp(),time_KST=kst(),event=kind,**data),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def save_ledger(rows):
    path=C/'execution_ledger.csv';temp=path.with_suffix('.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def progress(row):
    path=P/'runs'/row['cell_id'];epoch=0;elapsed=None
    try:
        epoch=int(read(path/'progress.json')['epoch'])
        start=read(path/'LIFECYCLE_START.json')['time']
        elapsed=max(0.,time.time()-start)
    except FileNotFoundError:pass
    return epoch,elapsed
def status(rows,state,current=None,action=None):
    epoch=0;elapsed=None
    if current and action=='TRAIN':epoch,elapsed=progress(current)
    estimate=eta(rows,current['cell_id'] if current else None,elapsed,epoch)
    acquisitions=[]
    for f in (P/'acquisition_diagnostics').glob('*/*/R*_diagnostics.json'):
        r=read(f)
        if r['gate']=='PASS':acquisitions.append(r)
    durations=[r['seconds'] for r in acquisitions]
    seconds_each=statistics.median(durations) if durations else 90.
    acq_hours=max(0,30-len(acquisitions))*seconds_each/3600
    total_hours=estimate['remaining_training_hours']+acq_hours
    doc={'state':state,'updated_UTC':stamp(),'updated_KST':kst(),'controller_pid':os.getpid(),
        'new_training_target':101,'new_succeeded':sum(r['state']=='SUCCEEDED' for r in rows),
        'reused_models':sum(r['state']=='REUSED' for r in rows),'failed':sum(r['state']=='FAILED' for r in rows),
        'pending':sum(r['state']=='PENDING' for r in rows),'running':sum(r['state']=='RUNNING' for r in rows),
        'current_cell':current['cell_id'] if current else None,'action':action,'current_epoch':epoch,
        'current_training_elapsed_seconds':elapsed,'remaining_training_estimate':estimate,
        'acquisitions_passed':len(acquisitions),'acquisition_target':30,'remaining_acquisition_hours_estimate':acq_hours,
        'remaining_training_plus_acquisition_hours':total_hours,
        'estimated_models_complete_KST':(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))+datetime.timedelta(hours=total_hours)).isoformat(timespec='seconds'),
        'estimate_excludes':'Final test evaluation, interruptions and user stop; early ETA provisional until complete cells are measured.',
        'free_disk_GiB':shutil.disk_usage(P).free/2**30,'test_inference_performed':False,
        'protocol_sha':sha(P/'config/protocol.json'),'execution_freeze_sha':sha(P/'evidence/execution_freeze.json')}
    if current and action=='ACQUIRE':
        cfg=read(P/'config/protocol.json');seed=cfg['seed_pairs'][int(current['pair'])-1]['acquisition']
        path=P/'acquisition_diagnostics'/current['dataset']/str(seed)/f"R{current['acquire_round']}_progress.json"
        if path.exists():doc['acquisition_progress']=read(path)
    atomic(C/'status.json',doc)
    if current:
        current['epochs_completed']=epoch if action=='TRAIN' else current['epochs_completed']
        current['updated_KST']=doc['updated_KST']
    save_ledger(rows)
    lines=['# 최종 실험 실행 현황','',f"갱신: {doc['updated_KST']}",f"상태: {state}",'',
        f"- 신규 학습 완료 {doc['new_succeeded']}/101, 기존 Full 재사용 {doc['reused_models']}, 실패 {doc['failed']}, 대기 {doc['pending']}",
        f"- 현재: {doc['current_cell'] or '없음'}, 작업 {action or '-'}, epoch {epoch}/100",
        f"- 추정 잔여 학습 {estimate['remaining_training_hours']:.2f} GPU시간 + Entropy 획득 {acq_hours:.2f}시간",
        f"- 예상 모델 완료: {doc['estimated_models_complete_KST']}",
        '- 최종 테스트는 실행하지 않음. ETA는 실제 완료 셀과 진행 epoch로 갱신하며 중단 시간·최종 평가를 제외함.',
        f"- Entropy 획득 검증: {len(acquisitions)}/30; 예측 없음·동점·ID 순위 기록은 acquisition_diagnostics에 저장.",'',
        '| 데이터 | 반복 | 방법 | 예산 | 상태 | epoch | 학습시간(분) |','|---|---:|---|---:|---|---:|---:|']
    for r in rows:
        minutes=f"{float(r['training_seconds'])/60:.2f}" if r['training_seconds'] else ''
        lines.append(f"|{r['dataset']}|{r['pair']}|{r['method']}|{r['budget_percent']}%|{r['state']}|{r['epochs_completed']}|{minutes}|")
    temp=C/'STATUS.tmp';temp.write_text('\n'.join(lines)+'\n',encoding='utf-8');os.replace(temp,C/'STATUS_KO.md')
    return doc

def owned_process(cmd,log_path,rows,row,action):
    verify()
    assert not (B/'research_extensions/GPU_TRAINING_ACTIVE.lock').exists(),'Another GPU operation owns the lock'
    pending=sum(r['state'] in ['PENDING','RUNNING'] for r in rows)
    assert shutil.disk_usage(P).free>pending*161_000_000+8*2**30,'Insufficient remaining disk budget'
    sys.path.insert(0,str(B/'research_extensions/full_data_26s_sgd_cosine_640_20261018_01'))
    from windows_job import WindowsJob
    job=WindowsJob();proc=None
    try:
        with log_path.open('x',encoding='utf-8') as log:
            proc=subprocess.Popen(cmd,cwd=P,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1'},creationflags=subprocess.CREATE_NO_WINDOW)
            job.assign(proc)
            event('PROCESS_STARTED',cell=row['cell_id'],action=action,pid=proc.pid,command=cmd)
            while True:
                try:code=proc.wait(timeout=15);break
                except subprocess.TimeoutExpired:status(rows,'RUNNING',row,action)
            if code!=0:raise RuntimeError(f'{action} failed exit={code}; log={log_path}')
    finally:
        if job:job.close()
        if proc and proc.poll() is None:proc.kill();proc.wait(timeout=30)

def acquire(rows,row,round_):
    cfg=read(P/'config/protocol.json');seed=cfg['seed_pairs'][int(row['pair'])-1]['acquisition']
    target=P/'selections'/row['dataset']/str(seed)/'Entropy'/f'R{round_}.json'
    if target.exists():
        m=read(target);report=Path(m['diagnostics_path'])
        assert sha(report)==m['diagnostics_sha'] and read(report)['gate']=='PASS'
        return
    row['acquire_round']=round_;status(rows,'RUNNING',row,'ACQUIRE')
    cmd=[str(PY),'-I','-B',str(P/'acquire.py'),'--dataset',row['dataset'],'--pair',row['pair'],'--round',str(round_)]
    owned_process(cmd,C/'logs'/f"{row['dataset']}_Entropy_pair{row['pair']}_R{round_}_acquire.log",rows,row,'ACQUIRE')
    m=read(target);report=Path(m['diagnostics_path'])
    assert sha(report)==m['diagnostics_sha']
    r=read(report);assert r['gate']=='PASS' and r['ranking_validated'] and r['selection_within_remaining_pool']
    event('ACQUISITION_VALIDATED',dataset=row['dataset'],pair=row['pair'],round=round_,
        no_prediction_fraction=r['no_prediction_fraction'],tied_candidate_fraction=r['candidates_in_exact_ties_fraction'],
        cutoff_tie_truncated_by_ID=r['cutoff_tie_truncated_by_ID'],selected_by_cutoff_ID_tie_fraction=r['selected_by_cutoff_ID_tie_fraction'],report=str(report))

def initialize():
    matrix=load_matrix();rows=[]
    for r in matrix:
        rows.append(dict(r,cell_id=cell_id(r),state='REUSED' if r['status']=='REUSE_TRAINED_CHECKPOINT' else 'PENDING',
            started_KST='',ended_KST='',updated_KST=kst(),epochs_completed=100 if r['status']=='REUSE_TRAINED_CHECKPOINT' else 0,
            training_seconds='',total_cell_seconds='',best_sha='',best_epoch='',error='',acquire_round=''))
    reused=next(r for r in rows if r['state']=='REUSED')
    old=B/reused['reuse_path'];reused['best_sha']=sha(old/'train/weights/best.pt');reused['best_epoch']=read(old/'best_selection.json')['epoch']
    return rows

def resume_rows():
    """Resume the same matrix, without rerunning successful or recovered cells."""
    gate=read(C/'recovery/20261021_entropy_eligible_guard_01/RECOVERY_PASS.json')
    assert gate['status']=='PASS' and gate['training_repeated'] is False and gate['test_inference'] is False
    assert gate['protocol_sha']==sha(P/'config/protocol.json')
    assert gate['execution_freeze_sha']==sha(P/'evidence/execution_freeze.json')
    with (C/'execution_ledger.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    original=load_matrix();assert len(rows)==len(original)==102
    lookup={r['cell_id']:r for r in rows};assert len(lookup)==102
    for planned in original:
        r=lookup[cell_id(planned)]
        for key,value in planned.items():assert str(r[key])==str(value),(key,r['cell_id'])
        assert r['state'] in ['SUCCEEDED','REUSED','PENDING'],'Unresolved failed/running cell; no automatic retry'
        if r['state']=='SUCCEEDED':
            root=P/'runs'/r['cell_id'];result=read(root/'RESULT.json')
            assert result['status']=='PASS' and result['epochs']==100 and result['technical_only'] is False and result['test_evaluated'] is False
            assert result['best_sha']==r['best_sha']==sha(root/'train/weights/best.pt')
            assert result['last_sha']==sha(root/'train/weights/last.pt')
        elif r['state']=='REUSED':assert sha(B/r['reuse_path']/'train/weights/best.pt')==r['best_sha']
        else:
            assert not (P/'runs'/r['cell_id']).exists(),'Existing incomplete run must be investigated, never retrained automatically'
            assert not (C/'logs'/(r['cell_id']+'.log')).exists(),'Preserve existing log; no automatic retry'
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',action='store_true',required=True);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    C.mkdir(exist_ok=True);(C/'logs').mkdir(exist_ok=True)
    assert read(P/'evidence/execution_start_preflight.json')['status']=='PASS'
    verify();lock=C/'CAMPAIGN_ACTIVE.lock'
    with lock.open('x',encoding='utf-8') as f:json.dump({'controller_pid':os.getpid(),'started_KST':kst()},f)
    rows=[];active=None
    # Keep this long local scientific run from being suspended by automatic system sleep.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        if args.resume:rows=resume_rows()
        else:
            assert not (C/'execution_ledger.csv').exists(),'Existing campaign ledger; explicit validated resume required'
            rows=initialize()
        lookup={r['cell_id']:r for r in rows};save_ledger(rows)
        event('CAMPAIGN_RESUMED' if args.resume else 'CAMPAIGN_STARTED',target_new_training=101,
            already_succeeded=sum(r['state']=='SUCCEEDED' for r in rows),protocol_sha=sha(P/'config/protocol.json'),
            execution_freeze_sha=sha(P/'evidence/execution_freeze.json'))
        for planned in ordered_new(load_matrix()):
            row=lookup[cell_id(planned)];active=row
            if row['state']=='SUCCEEDED':
                # A completed shared10 parent may still need its first acquisition.
                if row['method']=='SHARED_R0':acquire(rows,row,1)
                continue
            if row['method']=='Entropy':acquire(rows,row,int(row['round']))
            row.update(state='RUNNING',started_KST=kst());status(rows,'RUNNING',row,'TRAIN')
            event('TRAINING_STARTED',cell=row['cell_id'],dataset=row['dataset'],train_n=row['train_n'],learner_seed=row['learner_seed'])
            cmd=[str(PY),'-I','-B',str(P/'cell.py'),'--dataset',row['dataset'],'--method','Random' if row['method']=='SHARED_R0' else row['method'],
                '--pair',row['pair'],'--round',row['round'] or '0']
            before=time.perf_counter();owned_process(cmd,C/'logs'/(row['cell_id']+'.log'),rows,row,'TRAIN')
            run=P/'runs'/row['cell_id'];result=read(run/'RESULT.json')
            assert result['status']=='PASS' and result['epochs']==100 and result['technical_only'] is False and result['test_evaluated'] is False
            assert sha(run/'train/weights/best.pt')==result['best_sha'] and sha(run/'train/weights/last.pt')==result['last_sha']
            row.update(state='SUCCEEDED',ended_KST=kst(),epochs_completed=100,training_seconds=result['training_seconds'],
                total_cell_seconds=time.perf_counter()-before,best_sha=result['best_sha'],best_epoch=read(run/'best_selection.json')['epoch'])
            event('TRAINING_SUCCEEDED',cell=row['cell_id'],training_seconds=row['training_seconds'],total_cell_seconds=row['total_cell_seconds'],best_sha=row['best_sha'])
            save_ledger(rows)
            if row['method']=='SHARED_R0':acquire(rows,row,1)
            status(rows,'RUNNING')
        active=None
        assert sum(r['state']=='SUCCEEDED' for r in rows)==101 and sum(r['state']=='REUSED' for r in rows)==1
        checkpoints=[]
        for r in rows:
            source=B/r['reuse_path'] if r['state']=='REUSED' else P/'runs'/r['cell_id']
            path=source/'train/weights/best.pt';assert sha(path)==r['best_sha']
            checkpoints.append({'cell_id':r['cell_id'],'path':str(path),'sha':r['best_sha']})
        atomic(C/'MODELS_READY_FOR_SEPARATE_TEST.json',{'date_KST':kst(),'models':checkpoints,
            'protocol_sha':sha(P/'config/protocol.json'),'execution_freeze_sha':sha(P/'evidence/execution_freeze.json'),
            'test_inference_performed':False,'next_stage':'Prepare a separate frozen evaluation manifest; no test is started by this controller.'})
        event('ALL_MODELS_COMPLETE_TEST_NOT_RUN',new_completed=101,reused=1)
        status(rows,'MODELS_COMPLETE_TEST_NOT_RUN')
    except BaseException:
        error=traceback.format_exc()
        if active and active['state']!='SUCCEEDED':active.update(state='FAILED',ended_KST=kst(),error=error[-1600:])
        atomic(C/'failure.json',{'time_KST':kst(),'cell':active['cell_id'] if active else None,'error':error,
            'automatic_retry':False,'scientific_settings_changed':False})
        event('CAMPAIGN_STOPPED_ON_FAILURE',cell=active['cell_id'] if active else None,error=error)
        if rows:status(rows,'FAILED_STOPPED',active,'INVESTIGATE')
        (C/'FAILURE_REPORT_KO.md').write_text('# 정규 실험 중단 보고\n\n'+f'시간: {kst()}\n\n'+
            '과학적 설정을 변경하거나 자동 재시도하지 않았습니다. 실패 로그·가중치·실행 원장을 보존했습니다.\n\n```text\n'+error+'\n```\n',encoding='utf-8')
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        if lock.exists() and read(lock).get('controller_pid')==os.getpid():lock.unlink()

if __name__=='__main__':main()
