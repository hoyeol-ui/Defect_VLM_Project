"""One-shot independent Windows launch; no long polling or active Codex dependency."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
from defect_al.entropy import file_sha256
from defect_al.entropy_extension import read_json, write_json_new, now
from defect_al.entropy_official import OfficialExtension


def main():
    ext=OfficialExtension(REPO/'entropy_extension_v1')
    auth=ext.root/'authorizations/official_tranche_a00.json'
    ext.verify_authorization(auth,'GC10-DET_a00_Entropy_A_R1')
    ext.validate_gate('GC10-DET_a00',1)
    with ext.connection() as c:rows=[dict(r) for r in c.execute('SELECT * FROM jobs')]
    assert len(rows)==300 and sum(r['state']=='COMPLETE' for r in rows)==3
    assert not any(r['state'] in ('RUNNING','FAILED') for r in rows)
    assert sum(r['attempt'] for r in rows)==3
    logdir=ext.root/'runtime/official/detached_tranche_a00'
    logdir.mkdir(parents=True,exist_ok=False)
    stdout=logdir/'stdout.log';stderr=logdir/'stderr.log'
    files=[Path(__file__).resolve(),REPO/'scripts/entropy_extension/detached_tranche.py',
           REPO/'tests/test_entropy_detached.py',REPO/'reports/entropy_official_tranche_a00_20260908/safe_handoff.json']
    control={'stage':'USER_AUTHORIZED_EXECUTION_CONTROL_ONLY','created_at':now(),
        'official_freeze_sha256':ext.official_sha,'authorization_path':str(auth),'authorization_sha256':file_sha256(auth),
        'reason':'user requests Codex-independent Windows runner after sealed R1 B; experimental protocol unchanged',
        'remaining_training_cells':18,'remaining_queries':9,'max_total_training_cells':20,'max_total_queries':10,
        'stdout_path':str(stdout),'stderr_path':str(stderr),'startup_approval_path':str(logdir/'independent_startup.json'),
        'files':[{'path':str(p),'sha256':file_sha256(p)} for p in files]}
    seal=logdir/'execution_control_seal.json';write_json_new(seal,control)
    command=[ext.protocol['environment']['python_executable'],str(REPO/'scripts/entropy_extension/detached_tranche.py'),
             '--control-seal',str(seal),'--control-seal-sha256',file_sha256(seal),'--redirect-stdio']
    # WMI creates the process under its Windows service broker, outside the Codex process tree.
    commandline=subprocess.list2cmdline(command).replace("'", "''")
    cwd=str(ext.root).replace("'", "''")
    powershell=("$ErrorActionPreference='Stop'; "
        "$startup=New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0;CreateFlags=[uint32]8}; "
        f"$result=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{CommandLine='{commandline}';CurrentDirectory='{cwd}';ProcessStartupInformation=$startup}}; "
        "$result | Select-Object ProcessId,ReturnValue | ConvertTo-Json -Compress")
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',powershell],
        capture_output=True,text=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW,timeout=20)
    created=json.loads(result.stdout)
    if created['ReturnValue']!=0:
        raise RuntimeError(f'WMI creation failed: {created}')
    import psutil
    process=psutil.Process(created['ProcessId'])
    broker=psutil.Process(process.ppid())
    if broker.name().lower()!='wmiprvse.exe':
        process.terminate();process.wait(timeout=10)
        raise RuntimeError('unexpected Windows broker parent; no startup approval issued')
    launch={'status':'LAUNCHED','pid':process.pid,'process_create_time':process.create_time(),
        'started_at':datetime.fromtimestamp(process.create_time(),timezone.utc).isoformat(),
        'official_freeze_sha256':ext.official_sha,'authorization_sha256':file_sha256(auth),
        'execution_control_seal_sha256':file_sha256(seal),'command':command,'launch_method':'WMI Win32_Process.Create',
        'broker_pid':broker.pid,'broker_name':broker.name(),'process_owner':process.username(),
        'console_detached':True,'stdout_path':str(stdout),'stderr_path':str(stderr),
        'process_state_path':str(logdir/'process_state.json'),'codex_foreground_dependency':False}
    write_json_new(logdir/'launch_state.json',launch)
    write_json_new(logdir/'independent_startup.json',{'pid':process.pid,'independent_of_codex':True,
        'broker_pid':broker.pid,'broker_name':broker.name(),'verified_at':now()})
    # One bounded startup check only. The launcher exits and does not monitor epochs or completion.
    time.sleep(2)
    if not process.is_running():
        raise RuntimeError(f'detached runner exited during startup; see {stderr}')
    print(json.dumps(launch,indent=2))


if __name__=='__main__':main()
