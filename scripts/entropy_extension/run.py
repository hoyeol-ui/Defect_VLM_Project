"""Entropy extension commands. Integration execution requires explicit opt-in.

No locked-test operation exists. Every training cell runs in a fresh process.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
from defect_al.entropy import file_sha256
from defect_al.entropy_extension import (Extension, ExtensionGuardError, cell_id,
                                        now, trajectory_id, write_json_new)


def spawn(ext, args):
    subprocess.run([ext.protocol['environment']['python_executable'], str(Path(__file__).resolve()), *args],
                   cwd=ext.root, check=True)


def execute_trajectory(ext, dataset, index, through_round, authorization):
    tid = trajectory_id(dataset, index)
    if through_round not in range(1, 6):
        raise ExtensionGuardError('through-round must be 1..5')
    for round_idx in range(1, through_round + 1):
        spawn(ext, ['acquire', '--mode', ext.mode, '--dataset', dataset, '--index', str(index),
                    '--round', str(round_idx), '--authorization', str(authorization)])
        for slot in ('A', 'B'):
            spawn(ext, ['train-cell', '--mode', ext.mode, '--cell', cell_id(tid, round_idx, slot),
                        '--authorization', str(authorization)])


def verify_integration(ext):
    a = ext.completed('GC10-DET_a00_Entropy_A_R1')
    b = ext.completed('GC10-DET_a00_Entropy_B_R1')
    if (a['manifest_sha256'] != b['manifest_sha256'] or
            any(r['epochs_completed'] != 100 or r['cold_start'] is not True or
                r['checkpoint_selection_split'] != 'validation' or r['locked_test_accessed'] is not False for r in [a,b])):
        raise ExtensionGuardError('real integration lifecycle mismatch')
    next_parent = ext.policy_checkpoint('GC10-DET_a00', 2)
    if next_parent['sha256'] != a['best_sha256'] or next_parent['node_id'] != a['node_id']:
        raise ExtensionGuardError('next round policy is not R1 A best')
    result = {'status': 'PASS', 'freeze_sha256': ext.freeze_sha, 'timestamp': now(),
              'verified_completed_cells': [a['node_id'], b['node_id']],
              'manifest_sha256': a['manifest_sha256'], 'next_policy_parent': next_parent,
              'r2_acquisition_executed': False, 'locked_test_accessed': False}
    path = ext.root / 'integration_pass.json'
    if not path.exists():
        write_json_new(path, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['status','acquire','prepare-integration','integration','train-cell','run-trajectory','run-official','retry-failed'])
    p.add_argument('--mode', choices=['integration','official'], default='integration')
    p.add_argument('--dataset', choices=['GC10-DET','NEU-DET'], default='GC10-DET')
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--round', type=int, default=1)
    p.add_argument('--cell')
    p.add_argument('--authorization', type=Path)
    p.add_argument('--execute-training', action='store_true')
    p.add_argument('--approve-gc10-a00-r1', action='store_true')
    p.add_argument('--freeze-sha256')
    args = p.parse_args()
    ext = Extension(REPO / 'entropy_extension_v1', args.mode)
    ext.initialize_registry()
    if args.command == 'status':
        with ext.connection() as c:
            counts = [dict(r) for r in c.execute('SELECT state,COUNT(*) AS n FROM jobs GROUP BY state')]
        print(json.dumps({'namespace':ext.mode,'freeze_sha256':ext.freeze_sha,'states':counts}))
    elif args.command == 'acquire':
        if ext.mode == 'integration':
            result = ext.acquire(args.dataset,args.index,args.round)
        else:
            if not args.authorization: raise ExtensionGuardError('authorization required')
            result = ext.run_authorized_acquire(args.dataset,args.index,args.round,args.authorization)
        print(json.dumps({'acquisition':result['node_id'],'manifest_sha256':result['manifest_sha256']}))
    elif args.command == 'prepare-integration':
        if ext.mode != 'integration': raise ExtensionGuardError('integration namespace required')
        ext.acquire('GC10-DET',0,1)
        specs = [ext.prepare_training(f'GC10-DET_a00_Entropy_{slot}_R1') for slot in ('A','B')]
        if specs[0]['manifest_sha256'] != specs[1]['manifest_sha256']:
            raise ExtensionGuardError('A/B must share one manifest')
        print(json.dumps({'status':'READY_AWAITING_USER_TRAINING_DECISION','cells':[s['node_id'] for s in specs],
                          'manifest_sha256':specs[0]['manifest_sha256'],'training_executed':False}))
    elif args.command == 'train-cell':
        if not args.authorization or not args.cell: raise ExtensionGuardError('cell and authorization required')
        from defect_al.entropy_training import run_cell
        run_cell(ext.root,ext.mode,args.cell,args.authorization)
    elif args.command == 'retry-failed':
        if not args.cell: raise ExtensionGuardError('node ID required')
        ext.retry_failed(args.cell,args.authorization)
        print('Requeued same frozen node for an explicit cold-start retry; no training started')
    elif args.command == 'integration':
        if ext.mode != 'integration' or not args.execute_training or not args.approve_gc10_a00_r1 or args.freeze_sha256 != ext.freeze_sha:
            raise ExtensionGuardError('explicit two-cell opt-in and exact freeze hash required; no training started')
        auth_path = ext.root / 'authorizations/integration.json'
        if not auth_path.exists():
            write_json_new(auth_path, {'status':'AUTHORIZED','mode':'integration','freeze_sha256':ext.freeze_sha,
                           'allowed_cells':['GC10-DET_a00_Entropy_A_R1','GC10-DET_a00_Entropy_B_R1'],
                           'locked_test_access':False,'source':'explicit CLI --approve-gc10-a00-r1 opt-in','timestamp':now()})
        ext.verify_authorization(auth_path,'GC10-DET_a00_Entropy_A_R1')
        execute_trajectory(ext,'GC10-DET',0,1,auth_path)
        print(json.dumps(verify_integration(ext)))
    elif args.command in ('run-trajectory','run-official'):
        if ext.mode != 'official' or not args.execute_training or not args.authorization:
            raise ExtensionGuardError('official scope, explicit execution flag and new authorization required')
        ext.verify_authorization(args.authorization,'GC10-DET_a00_Entropy_A_R1')
        trajectories = [(args.dataset,args.index)] if args.command == 'run-trajectory' else [(d,i) for d in ('GC10-DET','NEU-DET') for i in range(10)]
        for dataset,index in trajectories:
            execute_trajectory(ext,dataset,index,5,args.authorization)


if __name__ == '__main__':
    main()
