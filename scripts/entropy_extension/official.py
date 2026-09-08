"""Bounded official tranche launcher. No full-run or test-evaluation command exists."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from defect_al.entropy import file_sha256
from defect_al.entropy_extension import ExtensionGuardError, cell_id, query_id, read_json, write_json_new, now
from defect_al.entropy_official import OfficialExtension


def run_job(ext, authorization, command, node, logdir):
    path = logdir / f'{node}_{command}.log'
    if path.exists():
        raise ExtensionGuardError('existing launch log; explicit recovery required')
    args = [ext.protocol['environment']['python_executable'], str(Path(__file__).resolve()), command,
            '--authorization', str(authorization), '--node', node]
    started = time.perf_counter()
    with path.open('x', encoding='utf-8') as log:
        child = subprocess.Popen(args, cwd=ext.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding='utf-8', errors='replace', bufsize=1,
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        for line in child.stdout:
            log.write(line)
            log.flush()
            if line.startswith(('ENTROPY_', 'OFFICIAL_')):
                print(line.rstrip(), flush=True)
        code = child.wait()
    write_json_new(logdir / f'{node}_{command}_runtime.json', {
        'command': command, 'node': node, 'returncode': code, 'wall_seconds': time.perf_counter() - started,
        'finished_at': now(), 'log_path': str(path), 'log_sha256': file_sha256(path)})
    if code:
        raise ExtensionGuardError(f'job failed: {node}; see {path}')


def launch(ext, authorization, approved_hash):
    checked_auth = ext.verify_authorization(authorization, 'GC10-DET_a00_Entropy_A_R1')
    if approved_hash != ext.official_sha:
        raise ExtensionGuardError('exact official freeze opt-in required')
    if checked_auth['authorized_trajectories'] != ['GC10-DET_a00', 'NEU-DET_a00']:
        raise ExtensionGuardError('this launcher permits exactly GC10 a00 and NEU a00')
    estimate = read_json(ext.release_root / 'runtime_estimate.json')
    if estimate['conservative_tranche_hours'] > 6:
        raise ExtensionGuardError('estimated total exceeds six hours')
    # First execution must start from a completely untouched official registry.
    with ext.connection() as c:
        rows = [dict(r) for r in c.execute('SELECT * FROM jobs')]
    if len(rows) != 300 or any(r['attempt'] != 0 or r['artifact'] for r in rows):
        raise ExtensionGuardError('official attempts must be zero before first tranche; explicit recovery required')
    ext.historical_integrity()
    logdir = ext.root / 'runtime/official/tranche_a00'
    logdir.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    write_json_new(logdir / 'started.json', {'timestamp': now(), 'official_freeze_sha256': ext.official_sha,
        'authorization_sha256': file_sha256(authorization), 'planned_cells': 20, 'planned_queries': 10,
        'integration_reused': False, 'max_hours': 8})
    print('OFFICIAL_START exact scope: GC10 a00 + NEU a00; 20 cells; 10 queries', flush=True)
    try:
        for trajectory in checked_auth['authorized_trajectories']:
            dataset = trajectory.rsplit('_a', 1)[0]
            for rr in range(1, 6):
                estimate_round = next(r['conservative_seconds'] for r in estimate['rounds'] if r['dataset'] == dataset and r['round'] == rr)
                if time.perf_counter() - start + estimate_round > 8 * 3600:
                    raise ExtensionGuardError('insufficient time remaining before eight-hour boundary; no next round started')
                qid = query_id(trajectory, rr)
                run_job(ext, authorization, 'acquire', qid, logdir)
                for slot in ('A', 'B'):
                    run_job(ext, authorization, 'train-cell', cell_id(trajectory, rr, slot), logdir)
                run_job(ext, authorization, 'audit-round', qid, logdir)
        ext.historical_integrity()
        write_json_new(logdir / 'complete.json', {'status': 'PASS', 'finished_at': now(),
            'wall_seconds': time.perf_counter() - start, 'official_freeze_sha256': ext.official_sha,
            'authorization_sha256': file_sha256(authorization), 'training_cells_completed': 20,
            'acquisitions_completed': 10, 'locked_test_accesses': 0})
        print('OFFICIAL_TRANCHE_COMPLETE 20/20 cells; 10/10 acquisitions; all round gates PASS', flush=True)
    except BaseException as error:
        write_json_new(logdir / 'blocked.json', {'status': 'BLOCKED', 'timestamp': now(),
            'wall_seconds': time.perf_counter() - start, 'reason': str(error)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['status', 'launch-tranche', 'acquire', 'train-cell', 'audit-round'])
    p.add_argument('--authorization', type=Path)
    p.add_argument('--node')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--official-freeze-sha256')
    args = p.parse_args()
    ext = OfficialExtension(REPO / 'entropy_extension_v1')
    if args.command == 'status':
        with ext.connection() as c:
            print(json.dumps([dict(r) for r in c.execute('SELECT state,COUNT(*) AS n,SUM(attempt) AS attempts FROM jobs GROUP BY state')]))
        return
    if not args.authorization:
        raise ExtensionGuardError('official tranche authorization required')
    if args.command == 'launch-tranche':
        if not args.execute:
            raise ExtensionGuardError('explicit --execute required; no jobs started')
        launch(ext, args.authorization, args.official_freeze_sha256)
        return
    if not args.node:
        raise ExtensionGuardError('node required')
    ext.verify_authorization(args.authorization, args.node)
    node = ext.node(args.node)
    if args.command == 'acquire':
        if node['kind'] != 'acquisition':
            raise ExtensionGuardError('query node required')
        ext.run_authorized_acquire(node['dataset'], node['acquisition_index'], node['round'], args.authorization)
        print(f'OFFICIAL_ACQUISITION_COMPLETE {args.node}', flush=True)
    elif args.command == 'train-cell':
        from defect_al.entropy_official_training import run_cell
        run_cell(ext.root, 'official', args.node, args.authorization)
    else:
        gate = ext.audit_round(node['trajectory'], node['round'])
        print(f"OFFICIAL_ROUND_GATE {node['trajectory']} R{node['round']} {gate['status']}", flush=True)


if __name__ == '__main__':
    main()
