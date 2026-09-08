"""Post-run audit and descriptive development analysis. No training entrypoint."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts/entropy_extension')]
from defect_al.entropy import file_sha256, stable_sha256
from defect_al.entropy_extension import checked_file, read_json, write_json_new, now
from defect_al.entropy_official import OfficialExtension
from prepare import historical_records

LABEL = 'NOT FINAL — 1/10 acquisition trajectories'
ROLE = 'INTERIM DESCRIPTIVE ONLY'


def main():
    import numpy as np
    import yaml
    ext = OfficialExtension(REPO/'entropy_extension_v1')
    runtime = ext.root/'runtime/official'
    detached = runtime/'detached_tranche_a00'
    auth_path = ext.root/'authorizations/official_tranche_a00.json'
    expected_auth = '1cf0b12328878f222be2a3824e945f21e51620240f16723db51b15ef959e3936'
    assert ext.official_sha == 'e40842787a93d2e06304f8da7eea5ea93f2125c7bd15982bbce6610903505e6a'
    checked_file(auth_path, expected_auth)
    auth = ext.verify_authorization(auth_path, 'GC10-DET_a00_Entropy_A_R1')
    state, completion = read_json(detached/'process_state.json'), read_json(detached/'completion_audit.json')
    assert state['status']=='COMPLETE' and completion['status']=='PASS'
    assert completion['completed_training_cells']==20 and completion['completed_acquisitions']==10
    for record in [state,completion,read_json(detached/'launch_state.json')]:
        assert record['official_freeze_sha256']==ext.official_sha and record['authorization_sha256']==expected_auth
    checked_file(detached/'execution_control_seal.json',state['execution_control_seal_sha256'])
    for item in read_json(detached/'execution_control_seal.json')['files']:
        checked_file(item['path'],item['sha256'])
    db_before = file_sha256(ext.db)
    with ext.connection() as c:
        registry = [dict(r) for r in c.execute('SELECT * FROM jobs ORDER BY node_id')]
    allowed = set(auth['allowed_cells']+auth['allowed_queries'])
    assert len(registry)==300 and len(allowed)==30
    assert {r['node_id'] for r in registry if r['state']=='COMPLETE'}==allowed
    assert all(r['attempt']==1 for r in registry if r['node_id'] in allowed)
    assert all(r['attempt']==0 and r['artifact'] is None for r in registry if r['node_id'] not in allowed)
    assert len(auth['allowed_cells'])==20 and len(auth['allowed_queries'])==10
    gates, cells, acquisition_results = [], [], []
    for ds in ('GC10-DET','NEU-DET'):
        for rr in range(1,6):
            path = ext.gate_path(f'{ds}_a00',rr)
            assert path.is_file()
            gate_hash = file_sha256(path)
            gate = ext.audit_round(f'{ds}_a00',rr)
            checked_file(path,gate_hash)
            assert gate['status']=='PASS'
            assert all(gate[key]==0 for key in ['locked_test_accesses','leakage','duplicates','reselections','learner_B_policy_edges'])
            q = ext.completed(f'{ds}_a00_Q_R{rr}')
            assert q['inference_config']==ext.protocol['inference']
            assert q['inference_config_sha256']==stable_sha256(ext.protocol['inference'])
            scores=read_json(next(f['path'] for f in q['files'] if f['path'].endswith('scores.json')))
            assert scores['hashes']==q['score_hashes']
            assert len(scores['ranking'])==len(set(scores['ranking']))
            assert scores['selected_ids']==scores['ranking'][:q['query_size']]
            for slot in ('A','B'):
                result=ext.completed(f'{ds}_a00_Entropy_{slot}_R{rr}')
                assert result['official_authorization_sha256']==expected_auth
                assert result['authorization_sha256']==expected_auth
                assert result['manifest_sha256']==q['manifest_sha256']
                spec=read_json(ext.root/'training_inputs/official'/q['node_id']/f'{slot}_spec.json')
                actual=yaml.safe_load((Path(result['best_path']).parents[1]/'args.yaml').read_text())
                for key,value in spec['training_config'].items():
                    assert str(actual[key])==str(value) if key=='device' else actual[key]==value, key
                cells.append(result)
            assert q['official_authorization_sha256']==expected_auth
            gates.append(gate);acquisition_results.append(q)
            print(f'POST_RUN_GATE_VERIFIED {ds} R{rr}',flush=True)
    protected, metadata = ext.historical_integrity()
    _, historical, last_count = historical_records()
    assert len(historical)==last_count==440
    assert not any('_B_' in parent for q in ext.queries for parent in q['depends_on'])
    warnings={}
    for base in [runtime/'tranche_a00',detached,ext.root/'logs_private/official']:
        for log in base.glob('*.log'):
            for line in log.read_text(encoding='utf-8',errors='replace').splitlines():
                if any(k in line for k in ['WARNING','Traceback','CUDA error','OutOfMemory']):
                    line=line.strip();warnings[line]=warnings.get(line,0)+1
    expected_warning="WARNING 'half' is deprecated and will be removed in the future. Use 'quantize' instead."
    assert set(warnings)<={expected_warning}
    handoff=read_json(REPO/'reports/entropy_official_tranche_a00_20260908/safe_handoff.json')
    assert handoff['intentional_scheduler_stop'] and not handoff['training_process_terminated']
    checked_file(handoff['round_gate_path'],handoff['round_gate_sha256'])
    audit={'status':'PASS WITH WARNINGS' if warnings else 'PASS','timestamp':now(),
        'official_freeze_sha256':ext.official_sha,'authorization_sha256':expected_auth,
        'training_cells_complete':20,'acquisitions_complete':10,'round_gates_pass':10,
        'historical_protected_hashes_unchanged':protected,'historical_best_last_manifest_cells_verified':440,
        'historical_analysis_metadata_unchanged':metadata,'locked_test_accesses':0,
        'locked_test_evidence':'frozen source/authorization, sealed train+development-only input IDs, args and completed records; no OS syscall tracing',
        'same_round_AB_manifest':True,'all_query_parents_previous_A_best':True,'B_policy_edges':0,
        'duplicates':0,'reselections':0,'leakage':0,'integration_reused':False,
        'freeze_authorization_hashes_match':True,'warnings':warnings,
        'foreground_exit_15':'intentional sealed-boundary handoff; not a failed training attempt',
        'official_registry_sha256_before':db_before,'remaining_training_cells':180,'remaining_acquisitions':90,
        'cells':cells,'gates':gates}
    if (REPORT/'post_run_audit.json').exists():
        previous_audit=read_json(REPORT/'post_run_audit.json')
        for key in ['official_freeze_sha256','authorization_sha256','official_registry_sha256_before','cells','gates']:
            assert previous_audit[key]==audit[key]
    else:
        write_json_new(REPORT/'post_run_audit.json',audit)
    print('POST_RUN_AUDIT_PASS development-only analysis begins',flush=True)

    # Work on independent copies so even the framework's label cache cannot write into official inputs.
    data_yamls={}
    for ds in ('GC10-DET','NEU-DET'):
        dataset=ext.dataset(ds);dest=REPORT/'development_inputs'/ds
        for image_id in dataset['validation_ids']:
            for kind,suffix in [('images','.jpg'),('labels','.txt')]:
                source=dataset[kind][image_id];checked_file(source['path'],source['sha256'])
                target=dest/kind/f'{image_id}{suffix}';target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():shutil.copyfile(source['path'],target)
                checked_file(target,source['sha256'])
        images=sorted((dest/'images').glob('*.jpg'))
        assert {p.stem for p in images}==set(dataset['validation_ids'])
        image_list=dest/'development.txt';image_list.write_text('\n'.join(p.as_posix() for p in images)+'\n',encoding='utf-8')
        cfg={'path':dest.as_posix(),'train':image_list.as_posix(),'val':image_list.as_posix(),'names':dataset['names']}
        # The API requires a train key; this script contains no training call.
        data_yamls[ds]=dest/'development_only.yaml'
        data_yamls[ds].write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    from defect_al.entropy_training import verify_environment
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    verify_environment(ext.protocol)
    raw=[];started=time.perf_counter();reused_metrics=False
    for result in cells:
        cid=result['node_id'];ds=result['dataset'];yaml_path=data_yamls[ds]
        checked_file(result['best_path'],result['best_sha256'])
        metric_path=REPORT/'development_metrics'/f'{cid}.json'
        if metric_path.exists():
            row=read_json(metric_path)
            assert row['checkpoint_sha256']==result['best_sha256']
            assert row['development_yaml_sha256']==file_sha256(yaml_path)
            assert (row['dataset'],row['round'],row['slot'])==(ds,result['round'],result['slot'])
            raw.append(row);reused_metrics=True
            continue
        init_seeds(seed=result['training_seed'],deterministic=True)
        with (REPORT/f'{cid}_development.log').open('x',encoding='utf-8') as log, redirect_stdout(log),redirect_stderr(log):
            metric=YOLO(result['best_path'],verbose=False).val(data=str(yaml_path),split='val',imgsz=640,
                batch=8,device=0,workers=4,conf=0.001,iou=0.7,max_det=300,plots=False,save_json=False,
                verbose=False,project=str(REPORT/'validation_outputs'),name=cid)
        row={'dataset':ds,'method':'Entropy','round':result['round'],'slot':result['slot'],
            'map50_95':float(metric.box.map),'source':'development revalidation of completed official best.pt',
            'checkpoint_sha256':result['best_sha256'],'checkpoint_path':result['best_path'],
            'development_yaml_sha256':file_sha256(yaml_path),'development_n':len(ext.dataset(ds)['validation_ids'])}
        raw.append(row);write_json_new(REPORT/'development_metrics'/f'{cid}.json',row)
        print(f'DEVELOPMENT_VALIDATED {cid}',flush=True)
    eval_seconds=None if reused_metrics else time.perf_counter()-started
    for row in historical:
        if row['acquisition_index']!=0:continue
        path=Path(row['best_pt_path']).parents[1]/'OUTCOME_PRIVATE_DEVELOPMENT_METRICS.json'
        metric=read_json(path);assert metric['metric_source']=='development'
        # Also verify the archived validation membership behind the comparison metrics.
        actual=yaml.safe_load((Path(row['best_pt_path']).parents[1]/'args.yaml').read_text())
        archived_yaml=yaml.safe_load(Path(actual['data']).read_text())
        val_path=Path(archived_yaml['val'])
        if not val_path.is_absolute():
            archive_root=Path(archived_yaml['path'])
            assert archive_root.is_absolute()
            val_path=archive_root/val_path
        assert val_path.is_absolute() and val_path.suffix=='.txt'
        ids={Path(p).stem for p in val_path.read_text().splitlines()}
        assert ids==set(ext.dataset(row['dataset'])['validation_ids'])
        methods=['Entropy','Random','DINO'] if row['round']==0 else [row['cell_id'].split('_')[2]]
        for method in methods:
            raw.append({'dataset':row['dataset'],'method':method,'round':row['round'],'slot':row['training_seed_slot'],
                'map50_95':float(metric['map50_95']),'source':'archived development-only post-training best.pt evaluation',
                'checkpoint_sha256':row['best_pt_sha256'],'checkpoint_path':row['best_pt_path'],
                'metric_path':str(path),'metric_sha256':file_sha256(path),'development_n':len(ids)})
    assert len(raw)==72
    curves=[];aucs=[]
    for ds in ('GC10-DET','NEU-DET'):
        budget=ext.protocol['budgets'][ds]
        x=np.asarray(budget['cumulative_counts'],dtype=float)/budget['acquisition_universe_n']
        for method in ('Entropy','Random','DINO'):
            means=[];slot_curves={s:[] for s in ('A','B')}
            for rr in range(6):
                group=[r for r in raw if r['dataset']==ds and r['method']==method and r['round']==rr]
                assert len(group)==2 and {r['slot'] for r in group}=={'A','B'}
                for r in group:slot_curves[r['slot']].append(r['map50_95'])
                mean=statistics.mean(r['map50_95'] for r in group);means.append(mean)
                curves.append({'dataset':ds,'method':method,'round':rr,'labeled_n':budget['cumulative_counts'][rr],
                    'budget_fraction':float(x[rr]),'AB_mean_map50_95':mean,'role':ROLE,'label':LABEL})
            auc=float(np.trapezoid(means,x)/(x[-1]-x[0]))
            manual=sum((x[i+1]-x[i])*(means[i]+means[i+1])/2 for i in range(5))/(x[-1]-x[0])
            learner_mean=statistics.mean(float(np.trapezoid(v,x)/(x[-1]-x[0])) for v in slot_curves.values())
            assert abs(auc-manual)<1e-12 and abs(auc-learner_mean)<1e-12
            aucs.append({'dataset':ds,'method':method,'provisional_nAULC':auc,'role':ROLE,'label':LABEL})
    write_json_new(REPORT/'interim_raw_development.json',raw)
    write_json_new(REPORT/'interim_descriptive_results.json',{'role':ROLE,'label':LABEL,'curves':curves,'nAULC':aucs,
        'formula':'trapezoidal A/B mean development mAP50-95 over exact labeled fractions, divided by R5-R0 fraction span',
        'development_revalidation_seconds':eval_seconds,
        'development_timing_note':'exact aggregate validation timer was not saved before relative-path comparison assembly resumed; existing 20 validated metric records reused' if reused_metrics else 'measured wall time',
        'statistics':'descriptive only; no CI or hypothesis tests',
        'protocol_changes':False,'new_training_calls':0,'locked_test_accesses':0})

    # Combine foreground and independent job records. B1 has no subprocess runtime record
    # because the scheduler was intentionally suspended; account for its measured cell times.
    operations=[]
    for base in (runtime/'tranche_a00',detached):
        for path in base.glob('*_runtime.json'):
            item=read_json(path);assert item['returncode']==0
            checked_file(item['log_path'],item['log_sha256']);operations.append(item)
    assert len(operations)==38
    round_times=[]
    for ds in ('GC10-DET','NEU-DET'):
        for rr in range(1,6):
            group=[r for r in cells if r['dataset']==ds and r['round']==rr]
            measured=sum(r['train_runtime_seconds']+r['validation_runtime_seconds'] for r in group)
            qid=f'{ds}_a00_Q_R{rr}'
            query=next(o['wall_seconds'] for o in operations if o['node']==qid and o['command']=='acquire')
            overheads=[o['wall_seconds']-r['train_runtime_seconds']-r['validation_runtime_seconds'] for r in cells
                       for o in operations if o['node']==r['node_id'] and o['command']=='train-cell']
            process_overhead=sum(next((o['wall_seconds']-r['train_runtime_seconds']-r['validation_runtime_seconds']
                for o in operations if o['node']==r['node_id'] and o['command']=='train-cell'),statistics.median(overheads)) for r in group)
            gate_seconds=next((o['wall_seconds'] for o in operations if o['node']==qid and o['command']=='audit-round'),
                             statistics.median(o['wall_seconds'] for o in operations if o['command']=='audit-round'))
            total=measured+query+process_overhead+gate_seconds
            round_times.append({'dataset':ds,'round':rr,'train_and_validation_seconds':measured,
                'acquisition_seconds':query,'process_overhead_seconds':process_overhead,'gate_seconds':gate_seconds,
                'estimated_active_wall_seconds':total,'missing_overhead_imputed':ds=='GC10-DET' and rr==1})
    totals={ds:sum(r['estimated_active_wall_seconds'] for r in round_times if r['dataset']==ds) for ds in ('GC10-DET','NEU-DET')}
    remaining=9*sum(totals.values())/3600
    runtime_report={'rounds':round_times,'dataset_active_seconds':totals,
        'total_elapsed_wall_seconds_including_handoff':completion['wall_seconds_since_original_launch'],
        'estimated_active_seconds':sum(totals.values()),'remaining_training_cells':180,'remaining_acquisitions':90,
        'remaining_hours_central':remaining,'planning_hours_range':[remaining*.9,remaining*1.2],
        'basis':'same local RTX3070; 9 remaining trajectories/dataset multiplied by measured a00 round runtimes; handoff idle time excluded',
        'imputation':'GC10 R1 B process overhead and manual R1 gate runtime missing after intentional scheduler suspension; use observed medians',
        'range_is_confidence_interval':False,'range_note':'operational planning margin only; image mix, load and thermal changes can alter duration',
        'sampled_peak_vram_mib':max(completion['sampled_peak_vram_mib'],read_json(REPO/'reports/entropy_official_tranche_a00_20260908/execution_summary.json')['sampled_device_peak_vram_mib'])}
    write_json_new(REPORT/'runtime_and_remaining_work.json',runtime_report)
    checked_file(ext.db,db_before);ext.historical_integrity()
    write_json_new(REPORT/'after_analysis_integrity.json',{'status':'PASS','official_registry_sha256':file_sha256(ext.db),
        'registry_bytes_unchanged':True,'new_training_calls':0,'locked_test_accesses':0,'historical_protected_hashes_unchanged':protected})
    print(json.dumps({'audit':audit['status'],'nAULC':aucs,'runtime':runtime_report,'curves':curves},indent=2))


if __name__=='__main__':main()
