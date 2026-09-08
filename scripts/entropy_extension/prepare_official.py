"""Freeze the official protocol and authorize only the requested two trajectories."""
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'src'), str(Path(__file__).resolve().parent)]
from defect_al.entropy import file_sha256, stable_sha256, validate_inference_config
from defect_al.entropy_extension import Extension, ExtensionGuardError, checked_file, read_json, write_json_new, now
from prepare import historical_records


def main():
    ext = Extension(REPO / 'entropy_extension_v1', 'official')
    release = ext.root / 'official_release_v1'
    if release.exists():
        raise ExtensionGuardError('official release already exists; never overwrite')
    expected = '0d3deb6656b9b22ddbb8d63f17ef6721adac4f3120560528c44954f8713d00fb'
    if ext.freeze_sha != expected:
        raise ExtensionGuardError('engine freeze differs from passed integration')
    integ = Extension(ext.root, 'integration')
    audit = read_json(REPO / 'reports/entropy_integration_20260908/integration_audit.json')
    if audit['status'] != 'INTEGRATION PASS':
        raise ExtensionGuardError('real integration PASS required')
    integration = [integ.completed(f'GC10-DET_a00_Entropy_{slot}_R1') for slot in ('A','B')]
    with ext.connection() as c:
        rows = [dict(r) for r in c.execute('SELECT * FROM jobs ORDER BY node_id')]
    assert len(rows) == 300 and all(r['attempt'] == 0 and r['artifact'] is None for r in rows)
    assert len(ext.plan) == 200 and len(ext.queries) == 100 and len({n['trajectory'] for n in ext.plan}) == 20
    assert all('_A_' in q['depends_on'][0] and '_B_' not in q['depends_on'][0] for q in ext.queries)
    for subdir in ['acquisitions/official','outcomes_private/official','training_inputs/official']:
        assert not (ext.root / subdir).exists()
    config = read_json(ext.root / 'training_config.json')
    assert config['epochs'] == 100 and config['batch'] == 8 and config['imgsz'] == 640
    assert config['optimizer'] == 'SGD' and config['resume'] is False and config['val'] is True
    assert config['model'] == ext.protocol['pretrained']['path']
    checked_file(ext.protocol['pretrained']['path'], ext.protocol['pretrained']['sha256'])
    validate_inference_config(ext.protocol['inference'])
    for node in ext.plan:
        ref = ext.r0[f"{node['trajectory']}_SHARED_{node['slot']}_R0"]
        assert node['training_seed'] == ref['training_seed']
        assert node['labeled_count'] == ext.protocol['budgets'][node['dataset']]['cumulative_counts'][node['round']]
        for key in ['manifest','best','training_args']:
            checked_file(ref[key + '_path'], ref[key + '_sha256'])
    for path, digest in read_json(ext.root / 'historical_guard.json').items():
        checked_file(REPO / path, digest)
    public, history, last_count = historical_records()
    metadata = []
    protected_roots = [REPO/'confirmatory_v1_1/final_analysis', REPO/'confirmatory_v1_1/interim_exploratory',
                       REPO/'docs/design/full_data_reference_20260908']
    for root in protected_roots:
        for path in sorted(root.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                stat = path.stat()
                metadata.append({'path':str(path),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns})
    # Only runtime fields from historical training metadata enter the estimate.
    raw = [{'dataset':r['dataset'],'round':r['round'],'cell_id':r['cell_id'],
            'train_seconds':r['training_metadata']['train_runtime_seconds'],
            'validation_seconds':r['training_metadata']['development_eval_runtime_seconds']}
           for r in history if r['round'] > 0]
    base = statistics.median(r['train_seconds'] for r in raw if r['dataset']=='GC10-DET' and r['round']==1)
    factor = statistics.mean(r['train_runtime_seconds'] for r in integration) / base
    observed_overhead = audit['execution']['total_wall_seconds'] - sum(r['train_runtime_seconds']+r['validation_runtime_seconds'] for r in integration)
    rounds = []
    for ds in ['GC10-DET','NEU-DET']:
        smoke = read_json(REPO / f'reports/entropy_preflight_20260908/{ds}/repeat_1/summary.json')
        budget = ext.protocol['budgets'][ds]
        for rr in range(1,6):
            group = [r for r in raw if r['dataset']==ds and r['round']==rr]
            query_s = smoke['duration_seconds'] * (budget['acquisition_universe_n'] - budget['cumulative_counts'][rr-1]) / smoke['stats']['unlabeled_pool_size'] + 15
            gate_s = 15
            overhead = max(20, observed_overhead)
            central = 2*(statistics.median(r['train_seconds'] for r in group)*factor + statistics.median(r['validation_seconds'] for r in group)) + overhead + query_s + gate_s
            conservative = 2*(max(r['train_seconds'] for r in group)*factor + max(r['validation_seconds'] for r in group)) + 2*overhead + 1.5*query_s + 2*gate_s
            rounds.append({'dataset':ds,'round':rr,'train_count':budget['cumulative_counts'][rr],
                'historical_runtime_n':len(group),'estimated_seconds':central,'conservative_seconds':conservative,
                'acquisition_estimate_seconds':query_s,'round_gate_estimate_seconds':gate_s})
    estimate = {'method':'dataset/round historical medians scaled by GC10 integration training ratio; validation separate; process, query and gate overhead included',
        'neu_scaling_assumption':'same 1.072x local runtime correction; no NEU integration timing available',
        'observed_gc10_training_ratio':factor,'integration_two_cell_wall_seconds':audit['execution']['total_wall_seconds'],
        'integration_observed_process_overhead_seconds':observed_overhead,
        'rounds':rounds,'central_tranche_hours':sum(r['estimated_seconds'] for r in rounds)/3600,
        'conservative_tranche_hours':sum(r['conservative_seconds'] for r in rounds)/3600,
        'full_200_central_hours':sum(r['estimated_seconds'] for r in rounds)*10/3600,
        'source_runtime_measurements':raw,'performance_values_used':False,'gpu':'NVIDIA GeForce RTX 3070'}
    if estimate['conservative_tranche_hours'] > 6:
        print(json.dumps(estimate));raise ExtensionGuardError('estimated total exceeds six hours; not authorized')
    protocol = {'id':'entropy-official-v1','stage':'OFFICIAL_PROTOCOL_FROZEN','created_at':now(),
        'engine_freeze_sha256':ext.freeze_sha,'method_protocol_path':str(ext.root/'protocol.json'),
        'method_protocol_sha256':file_sha256(ext.root/'protocol.json'), 'integration_reuse':False,
        'all_official_cells_fresh_after_authorization':True,'planned_training_cells':200,'planned_acquisitions':100,
        'trajectories':20,'policy_learner':'A','learner_B_policy_participant':False,'manifest_sharing':'A/B identical cumulative manifest',
        'cold_start_pretrained':ext.protocol['pretrained'],'training_config_sha256':file_sha256(ext.root/'training_config.json'),
        'rounds':[1,2,3,4,5],'entropy':{'p':'post-NMS boxes.conf','equation':'-p ln(p) - (1-p) ln(1-p)',
            'aggregation':'SUM','no_detection':0.0,'conf':0.25,'iou':0.70,'max_det':300,'class_aware_nms':True,
            'ranking':'score descending, canonical image ID ascending'},
        'limitations':{'smoke_entropy_detection_count_spearman':{'GC10-DET':0.9466,'NEU-DET':0.9738},
            'interpretation':'strong dependence on number of retained detections; no threshold/aggregation/seed/budget tuning'},
        'interim_policy':'development descriptive only after complete tranche integrity audit; NOT FINAL - 1/10 acquisition trajectories; no CI/significance/superiority claim; no protocol changes from interim outcomes',
        'locked_test_access':'FORBIDDEN','locked_test_endpoint':None,'reference_runs_in_scope':False,
        'historical_R0':'hash-verified read-only reuse','historical_440_mutation':'FORBIDDEN',
        'official_artifacts_root':str(ext.root),'namespace':'official',
        'round_gate':'both learner artifacts, selection, data, policy-parent and historical integrity PASS before next round',
        'worker_delta':'scoped authorization/official provenance and CUDA memory counters only; training semantics identical to integration worker',
        'statistical_unit':ext.protocol['statistical_unit'],'learner_reduction':ext.protocol['learner_reduction']}
    release.mkdir()
    write_json_new(release/'protocol.json',protocol)
    write_json_new(release/'runtime_estimate.json',estimate)
    write_json_new(release/'protected_metadata.json',metadata)
    write_json_new(release/'initial_registry_snapshot.json',{'freeze_sha256':ext.freeze_sha,'rows':rows,
                    'database_sha256':file_sha256(ext.db),'official_attempts':0})
    write_json_new(release/'preflight.json',{'status':'PASS','timestamp':now(),'planned_cells':200,'queries':100,
        'trajectories':20,'official_attempts':0,'integration_artifacts_in_registry':0,
        'R0_read_only_references_verified':40,'historical_best_last_manifest_count':440,
        'seeds_budgets_config_hashes':'PASS','locked_test_endpoint':'DISABLED','B_policy_edges':0,
        'execution_time_gate':'PASS','reference_runs_preserved':True})
    static = list(release.glob('*.json')) + [ext.root/'freeze.json', ext.root/'integration_pass.json',
        REPO/'reports/entropy_integration_20260908/integration_audit.json',
        ext.root/'yolo26n.pt']
    static += [REPO/p for p in ['src/defect_al/entropy_official.py','src/defect_al/entropy_official_training.py',
        'scripts/entropy_extension/official.py','scripts/entropy_extension/prepare_official.py','tests/test_entropy_official.py']]
    write_json_new(release/'freeze.json',{'stage':'OFFICIAL_PROTOCOL_FROZEN','created_at':now(),
        'engine_freeze_sha256':ext.freeze_sha,'files':[{'path':str(p),'sha256':file_sha256(p)} for p in static]})
    official_sha = file_sha256(release/'freeze.json')
    tids = ['GC10-DET_a00','NEU-DET_a00']
    auth = {'status':'AUTHORIZED','mode':'official','created_at':now(),'source':'explicit user attached tranche instructions 2026-09-08; estimated total below six hours',
        'freeze_sha256':ext.freeze_sha,'official_freeze_sha256':official_sha,'locked_test_access':False,
        'integration_reuse':False,'authorized_trajectories':tids,'max_wall_hours':8,'automatic_scope_expansion':False,
        'allowed_cells':[n['node_id'] for n in ext.plan if n['trajectory'] in tids],
        'allowed_queries':[n['node_id'] for n in ext.queries if n['trajectory'] in tids],
        'integration_pass_path':str(ext.root/'integration_pass.json'),
        'integration_pass_sha256':file_sha256(ext.root/'integration_pass.json')}
    auth_path = ext.root/'authorizations/official_tranche_a00.json'
    write_json_new(auth_path,auth)
    from defect_al.entropy_official import OfficialExtension
    official = OfficialExtension(ext.root)
    for node in auth['allowed_cells'] + auth['allowed_queries']:
        official.verify_authorization(auth_path,node)
    official.historical_integrity()
    command = f'& "{ext.protocol["environment"]["python_executable"]}" "{REPO / "scripts/entropy_extension/official.py"}" launch-tranche --execute --authorization "{auth_path}" --official-freeze-sha256 {official_sha}'
    script = REPO/'docs/design/run_entropy_official_tranche_a00_20260908.ps1'
    with script.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write("$ErrorActionPreference = 'Stop'\n"+command+"\nif ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n")
    print(json.dumps({'status':'OFFICIAL_TRANCHE_READY','official_freeze_sha256':official_sha,
        'authorization_sha256':file_sha256(auth_path),'planned_cells':200,'authorized_cells':20,'authorized_queries':10,
        'central_hours':estimate['central_tranche_hours'],'conservative_hours':estimate['conservative_tranche_hours'],
        'round_minutes':[{'dataset':r['dataset'],'round':r['round'],'minutes':r['estimated_seconds']/60} for r in rounds],
        'command':command},indent=2))


if __name__ == '__main__':
    main()
