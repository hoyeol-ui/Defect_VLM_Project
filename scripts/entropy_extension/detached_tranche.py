"""Independent Windows continuation of the same authorized official tranche.

Only scheduling/telemetry changes. Frozen workers, acquisition and round gates stay intact.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'src'), str(Path(__file__).resolve().parent)]
from defect_al.entropy import file_sha256
from defect_al.entropy_extension import ExtensionGuardError, read_json, write_json_new, checked_file, now, query_id, cell_id
from defect_al.entropy_official import OfficialExtension
from official import run_job


def remaining_job_action(row):
    if row['state'] == 'COMPLETE':
        return 'VERIFY_AND_SKIP'
    if row['state'] in ('RUNNING', 'FAILED') or row['attempt'] != 0:
        raise ExtensionGuardError('ambiguous/incomplete previous attempt; no automatic recovery or retry')
    if row['state'] not in ('PENDING', 'READY', 'BLOCKED_MANIFEST'):
        raise ExtensionGuardError('unexpected job state')
    return 'RUN'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control-seal', type=Path, required=True)
    p.add_argument('--control-seal-sha256', required=True)
    p.add_argument('--redirect-stdio', action='store_true')
    args = p.parse_args()
    checked_file(args.control_seal, args.control_seal_sha256)
    control = read_json(args.control_seal)
    if args.redirect_stdio:
        sys.stdout = Path(control['stdout_path']).open('x', encoding='utf-8', buffering=1)
        sys.stderr = Path(control['stderr_path']).open('x', encoding='utf-8', buffering=1)
    for entry in control['files']:
        checked_file(entry['path'], entry['sha256'])
    # No job starts until the launcher verifies the independent Windows broker parent.
    startup_path = Path(control['startup_approval_path'])
    deadline = time.monotonic() + 30
    while not startup_path.exists() and time.monotonic() < deadline:
        time.sleep(0.2)
    if not startup_path.exists():
        raise ExtensionGuardError('independent-process startup approval missing')
    startup = read_json(startup_path)
    if startup['pid'] != os.getpid() or startup['independent_of_codex'] is not True:
        raise ExtensionGuardError('process is not independently detached')
    ext = OfficialExtension(REPO / 'entropy_extension_v1')
    if ext.official_sha != control['official_freeze_sha256']:
        raise ExtensionGuardError('official freeze mismatch')
    authorization = Path(control['authorization_path'])
    checked_file(authorization, control['authorization_sha256'])
    auth = ext.verify_authorization(authorization, 'GC10-DET_a00_Entropy_A_R1')
    if auth['authorized_trajectories'] != ['GC10-DET_a00', 'NEU-DET_a00']:
        raise ExtensionGuardError('exact two-trajectory authorization required')
    logdir = ext.root / 'runtime/official/detached_tranche_a00'
    logdir.mkdir(parents=True, exist_ok=True)
    state_path = logdir / 'process_state.json'
    if state_path.exists():
        raise ExtensionGuardError('existing detached launch; explicit reconciliation required')
    # An exclusive, non-inheritable file lock prevents duplicate detached schedulers.
    import msvcrt
    mutex = (logdir / 'scheduler.lock').open('a+b')
    mutex.seek(0); mutex.write(b'1'); mutex.flush(); mutex.seek(0)
    msvcrt.locking(mutex.fileno(), msvcrt.LK_NBLCK, 1)
    original_started = read_json(ext.root / 'runtime/official/tranche_a00/started.json')['timestamp']
    original_start = datetime.fromisoformat(original_started).timestamp()
    state = {'status':'RUNNING', 'pid':os.getpid(), 'parent_pid':os.getppid(), 'started_at':now(),
        'original_tranche_started_at':original_started, 'official_freeze_sha256':ext.official_sha,
        'authorization_sha256':file_sha256(authorization),'execution_control_seal_sha256':args.control_seal_sha256,
        'scope':auth['authorized_trajectories'],'max_training_cells':20,'max_acquisitions':10,
        'integration_reused':False,'locked_test_accesses':0,'current_job':None,'completed_cells':2,'completed_queries':1,
        'stdout_path':control['stdout_path'],'stderr_path':control['stderr_path']}

    def persist(**updates):
        state.update(updates); state['updated_at'] = now()
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
        temporary.replace(state_path)

    stop = threading.Event()
    peak = [0]
    errors = []
    def monitor_gpu():
        with (logdir / 'gpu_samples.jsonl').open('x',encoding='utf-8') as stream:
            while not stop.is_set():
                try:
                    result = subprocess.run(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    result.check_returncode()
                    timestamp,memory,utilization=[s.strip() for s in result.stdout.strip().splitlines()[0].split(',')]
                    peak[0]=max(peak[0],int(memory))
                    stream.write(json.dumps({'timestamp':timestamp,'memory_used_mib':int(memory),
                                             'utilization_percent':int(utilization)})+'\n');stream.flush()
                except Exception as error:errors.append(str(error))
                stop.wait(2)

    monitor=threading.Thread(target=monitor_gpu,daemon=True)
    persist(); monitor.start()
    print(f'DETACHED_OFFICIAL_START pid={os.getpid()} same freeze and authorization; no scope expansion',flush=True)
    try:
        with ext.connection() as c:
            rows=[dict(r) for r in c.execute('SELECT * FROM jobs')]
        if len(rows)!=300 or any(r['state'] in ('RUNNING','FAILED') for r in rows):
            raise ExtensionGuardError('registry must be at a sealed idle boundary')
        if any(r['attempt'] or r['artifact'] for r in rows if '_a00_' not in r['node_id']):
            raise ExtensionGuardError('outside-scope official attempt exists')
        ext.validate_gate('GC10-DET_a00',1)
        ext.historical_integrity()
        estimate=read_json(ext.release_root/'runtime_estimate.json')
        for trajectory in auth['authorized_trajectories']:
            dataset=trajectory.rsplit('_a',1)[0]
            for rr in range(1,6):
                if ext.gate_path(trajectory,rr).exists():
                    ext.validate_gate(trajectory,rr)
                    print(f'DETACHED_VERIFIED_EXISTING_ROUND {trajectory} R{rr}',flush=True)
                    continue
                round_estimate=next(r['conservative_seconds'] for r in estimate['rounds'] if r['dataset']==dataset and r['round']==rr)
                if time.time()-original_start+round_estimate>8*3600:
                    raise ExtensionGuardError('next round would exceed eight-hour wall-time budget')
                jobs=[('acquire',query_id(trajectory,rr))]+[('train-cell',cell_id(trajectory,rr,s)) for s in ('A','B')]
                for command,node in jobs:
                    ext.verify_authorization(authorization,node)
                    if remaining_job_action(ext.state(node))=='VERIFY_AND_SKIP':
                        ext.completed(node)
                        continue
                    persist(current_job=node,phase=command)
                    run_job(ext,authorization,command,node,logdir)
                    with ext.connection() as c:
                        done=[r['node_id'] for r in c.execute("SELECT node_id FROM jobs WHERE state='COMPLETE'")]
                    persist(completed_cells=sum('_Entropy_' in n for n in done),
                            completed_queries=sum('_Q_' in n for n in done),sampled_peak_vram_mib=peak[0])
                persist(current_job=query_id(trajectory,rr),phase='audit-round')
                run_job(ext,authorization,'audit-round',query_id(trajectory,rr),logdir)
        ext.historical_integrity()
        from prepare import historical_records
        historical,_,last_count=historical_records()
        if len(historical)!=440 or last_count!=440:
            raise ExtensionGuardError('historical integrity failure')
        gates=[ext.validate_gate(t,r) for t in auth['authorized_trajectories'] for r in range(1,6)]
        with ext.connection() as c:
            rows=[dict(r) for r in c.execute('SELECT * FROM jobs')]
        if (sum(r['state']=='COMPLETE' and '_Entropy_' in r['node_id'] for r in rows)!=20
                or sum(r['state']=='COMPLETE' and '_Q_' in r['node_id'] for r in rows)!=10
                or any(r['attempt'] or r['artifact'] for r in rows if '_a00_' not in r['node_id'])):
            raise ExtensionGuardError('final scope/count mismatch')
        summary={'status':'PASS','finished_at':now(),'official_freeze_sha256':ext.official_sha,
            'authorization_sha256':file_sha256(authorization),'completed_training_cells':20,'completed_acquisitions':10,
            'remaining_training_cells':180,'remaining_acquisitions':90,'round_gates':gates,
            'historical_440_unchanged':True,'locked_test_accesses':0,'integration_reused':False,
            'wall_seconds_since_original_launch':time.time()-original_start,'sampled_peak_vram_mib':peak[0],
            'gpu_monitor_errors':errors,'interim_descriptive_results':'not opened; ready for a later user-requested audit'}
        write_json_new(logdir/'completion_audit.json',summary)
        persist(status='COMPLETE',phase='SEALED',current_job=None,completed_cells=20,completed_queries=10,
                completed_at=now(),sampled_peak_vram_mib=peak[0],completion_audit_path=str(logdir/'completion_audit.json'))
        print('DETACHED_OFFICIAL_COMPLETE 20/20 cells; 10/10 queries; 10/10 round gates; no further work launched',flush=True)
    except BaseException as error:
        persist(status='BLOCKED',phase='STOPPED',error=str(error),sampled_peak_vram_mib=peak[0])
        print(f'DETACHED_OFFICIAL_BLOCKED {error}',file=sys.stderr,flush=True)
        raise
    finally:
        stop.set();monitor.join(timeout=12)
        mutex.seek(0);msvcrt.locking(mutex.fileno(),msvcrt.LK_UNLCK,1);mutex.close()


if __name__=='__main__':main()
