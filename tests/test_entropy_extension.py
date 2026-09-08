"""Dependency, seal, authorization and shared-trajectory integration tests (no GPU)."""
import json
from pathlib import Path

import pytest

from defect_al.entropy import PINNED_INFERENCE, file_sha256, entropy_acquisition
from defect_al.entropy_extension import (Extension, ExtensionGuardError, build_plan, cell_id,
    query_id, trajectory_id, require_training_authorization, deny_locked_test, write_json_new)


@pytest.fixture
def extension(tmp_path, monkeypatch):
    root = tmp_path / 'extension'
    root.mkdir()
    seeds = [{'dataset':ds,'acquisition_index':str(i),'acquisition_seed':str(100+i),
              'training_seed_a':str(1000+i),'training_seed_b':str(2000+i)}
             for ds in ['GC10-DET','NEU-DET'] for i in range(10)]
    budgets = {ds:{'query_count_per_round':1,'cumulative_counts':[1,2,3,4,5,6]} for ds in ['GC10-DET','NEU-DET']}
    protocol = {'policy_learner':'A','locked_test_access':'FORBIDDEN','seeds':seeds,'budgets':budgets,
                'inference':{**PINNED_INFERENCE,'software_version':{'fixture':'1'},'source_sha256':{'fixture':'1'}},
                'pretrained':{'path':str(tmp_path/'pretrained.pt'),'sha256':''}}
    (tmp_path/'pretrained.pt').write_bytes(b'pretrained')
    protocol['pretrained']['sha256']=file_sha256(tmp_path/'pretrained.pt')
    refs={}
    for ds in ['GC10-DET','NEU-DET']:
        ids=[f'{i:064x}' for i in range(1,9)]
        images,labels={},{}
        for i in ids:
            im=tmp_path/ds/f'{i}.jpg';im.parent.mkdir(exist_ok=True);im.write_bytes(b'image fixture')
            label=im.with_suffix('.txt');label.write_bytes(b'label fixture')
            images[i]={'path':str(im),'sha256':file_sha256(im)}
            labels[i]={'path':str(label),'sha256':file_sha256(label)}
        write_json_new(root/'datasets'/f'{ds}.json',{'train_ids':ids[:-1],'validation_ids':ids[-1:],
                       'images':images,'labels':labels,'names':{0:'fixture'}})
        for i in range(10):
            tid=trajectory_id(ds,i)
            manifest=tmp_path/f'{tid}_R0.txt';manifest.write_text(ids[0]+'\n')
            for slot in ['A','B']:
                cp=tmp_path/f'{tid}_{slot}'/'best.pt';cp.parent.mkdir();cp.write_bytes(slot.encode())
                refs[f'{tid}_SHARED_{slot}_R0']={'trajectory':tid,'slot':slot,'round':0,
                    'manifest_path':str(manifest),'manifest_sha256':file_sha256(manifest),
                    'best_path':str(cp),'best_sha256':file_sha256(cp)}
    cells,queries=build_plan(seeds,budgets)
    for name,payload in [('protocol',protocol),('planned_cells',cells),('planned_acquisitions',queries),
                         ('historical_r0',refs),('training_config',{'epochs':100,'resume':False,'val':True,
                          'model':str(tmp_path/'pretrained.pt')})]:
        write_json_new(root/f'{name}.json',payload)
    files=[{'path':str(p),'sha256':file_sha256(p)} for p in root.rglob('*.json')]
    write_json_new(root/'freeze.json',{'files':files})
    def fake_score(*args,**kwargs):
        kwargs['_predict_fn']=lambda cp,rows,cfg: ((r['image_id'],[0.5]) for r in rows)
        return entropy_acquisition(*args,**kwargs)
    monkeypatch.setattr('defect_al.entropy_extension.entropy_acquisition',fake_score)
    ext=Extension(root,'official');ext.initialize_registry()
    return ext


def fake_train_complete(ext,tid,round_idx,slot):
    cid=cell_id(tid,round_idx,slot)
    spec=ext.prepare_training(cid)
    attempt=ext._claim(cid,'READY')
    output=ext.root/'synthetic_results'/cid/f'attempt{attempt}'
    output.mkdir(parents=True)
    best=output/'best.pt';best.write_bytes(f'{cid} synthetic'.encode())
    payload={'node_id':cid,'freeze_sha256':ext.freeze_sha,'trajectory':tid,'round':round_idx,'slot':slot,
             'best_path':str(best),'best_sha256':file_sha256(best),'epochs_completed':100,'cold_start':True,
             'manifest_sha256':spec['manifest_sha256'],
             'files':[{'path':str(best),'sha256':file_sha256(best)}]}
    write_json_new(output/'result.json',payload);ext._complete(cid,output/'result.json')
    return payload


def test_exact_plan_cardinality_and_dependencies(extension):
    assert len(extension.plan)==200 and len(extension.queries)==100
    assert len({c['trajectory'] for c in extension.plan})==20
    assert all(c['round'] in range(1,6) for c in extension.plan)
    assert all('_A_' in q['depends_on'][0] and '_B_' not in q['depends_on'][0] for q in extension.queries)
    for q in extension.queries:
        cells=[c for c in extension.plan if c['depends_on']==[q['node_id']]]
        assert {c['slot'] for c in cells}=={'A','B'}
        assert len({c['labeled_count'] for c in cells})==1


def test_r2_before_r1_a_fails(extension):
    with pytest.raises(ExtensionGuardError,match='not COMPLETE'):
        extension.policy_checkpoint('GC10-DET_a00',2)


def test_unselected_training_cell_cannot_start(extension):
    with pytest.raises(ExtensionGuardError,match='not COMPLETE'):
        extension.prepare_training('GC10-DET_a00_Entropy_A_R1')


def test_b_completion_cannot_unblock_next_query(extension):
    extension._acquire('GC10-DET_a00_Q_R1')
    fake_train_complete(extension,'GC10-DET_a00',1,'B')
    with pytest.raises(ExtensionGuardError,match='not COMPLETE'):
        extension.policy_checkpoint('GC10-DET_a00',2)


def test_a_alone_supplies_next_policy_and_ab_share_manifest(extension):
    extension._acquire('GC10-DET_a00_Q_R1')
    a=extension.prepare_training('GC10-DET_a00_Entropy_A_R1')
    b=extension.prepare_training('GC10-DET_a00_Entropy_B_R1')
    assert a['manifest_sha256']==b['manifest_sha256']
    assert a['data_yaml_sha256']==b['data_yaml_sha256']
    assert a['training_config']['seed'] != b['training_config']['seed']
    result=fake_train_complete(extension,'GC10-DET_a00',1,'A')
    assert extension.policy_checkpoint('GC10-DET_a00',2)['sha256']==result['best_sha256']
    assert extension.state('GC10-DET_a00_Entropy_B_R1')['state']=='READY'


def test_all_twenty_five_round_chains_use_only_a(extension):
    for ds in ['GC10-DET','NEU-DET']:
        for i in range(10):
            tid=trajectory_id(ds,i)
            for r in range(1,6):
                q=extension._acquire(query_id(tid,r))
                assert q['policy_checkpoint']['slot']=='A'
                assert q['policy_checkpoint']['round']==r-1
                a=fake_train_complete(extension,tid,r,'A')
                b=fake_train_complete(extension,tid,r,'B')
                assert a['manifest_sha256']==b['manifest_sha256']
    with extension.connection() as c:
        assert c.execute("SELECT COUNT(*) FROM jobs WHERE state='COMPLETE'").fetchone()[0]==300


def test_read_only_r0_hash_guard(extension):
    r=extension.r0['GC10-DET_a00_SHARED_A_R0'];Path(r['best_path']).write_bytes(b'drift')
    with pytest.raises(ExtensionGuardError,match='seal mismatch'):
        extension.policy_checkpoint('GC10-DET_a00',1)


def test_shared_manifest_tamper_detected(extension):
    result=extension._acquire('GC10-DET_a00_Q_R1')
    Path(result['manifest_path']).write_text('bad')
    with pytest.raises(ExtensionGuardError,match='seal mismatch'):
        extension.prepare_training('GC10-DET_a00_Entropy_A_R1')


def test_checkpoint_tamper_blocks_next_round(extension):
    extension._acquire('GC10-DET_a00_Q_R1')
    result=fake_train_complete(extension,'GC10-DET_a00',1,'A')
    Path(result['best_path']).write_bytes(b'drift')
    with pytest.raises(ExtensionGuardError,match='seal mismatch'):
        extension.policy_checkpoint('GC10-DET_a00',2)


def test_frozen_config_tamper_detected(extension):
    (extension.root/'training_config.json').write_text('{}')
    with pytest.raises(ExtensionGuardError,match='seal mismatch'):
        Extension(extension.root)


@pytest.mark.parametrize('node',['NEU-DET_a00_Q_R1','GC10-DET_a01_Q_R1','GC10-DET_a00_Q_R2'])
def test_integration_cannot_expand_scope(extension,node):
    ext=Extension(extension.root,'integration');ext.initialize_registry()
    with pytest.raises(ExtensionGuardError,match='outside integration'):
        ext.node(node)


def test_missing_or_wrong_auth_fails_before_training():
    with pytest.raises(ExtensionGuardError):
        require_training_authorization({},'hash','integration','GC10-DET_a00_Entropy_A_R1')
    with pytest.raises(ExtensionGuardError):
        require_training_authorization({'status':'AUTHORIZED','freeze_sha256':'wrong'},'hash','integration','x')


def test_integration_auth_cannot_cover_extra_cells():
    auth={'status':'AUTHORIZED','freeze_sha256':'h','mode':'integration','locked_test_access':False,
          'allowed_cells':['GC10-DET_a00_Entropy_A_R1','GC10-DET_a00_Entropy_B_R1','NEU-DET_a00_Entropy_A_R1']}
    with pytest.raises(ExtensionGuardError,match='exactly'):
        require_training_authorization(auth,'h','integration','GC10-DET_a00_Entropy_A_R1')


def test_test_access_is_unconditionally_denied():
    with pytest.raises(ExtensionGuardError,match='forbidden'):
        deny_locked_test(authorization={'status':'AUTHORIZED'})


def test_double_claim_prevented(extension):
    extension._claim('GC10-DET_a00_Q_R1','PENDING')
    with pytest.raises(ExtensionGuardError):
        extension._claim('GC10-DET_a00_Q_R1','PENDING')


def test_retry_preserves_attempts_and_cannot_reset_running(extension):
    ext=Extension(extension.root,'integration');ext.initialize_registry();qid='GC10-DET_a00_Q_R1'
    ext._claim(qid,'PENDING')
    with pytest.raises(ExtensionGuardError): ext.retry_failed(qid)
    ext.fail(qid);ext.retry_failed(qid)
    assert ext._claim(qid,'PENDING')==2
    ext.fail(qid);ext.retry_failed(qid)
    assert ext._claim(qid,'PENDING')==3
    ext.fail(qid)
    with pytest.raises(ExtensionGuardError): ext.retry_failed(qid)


def test_label_bytes_are_copied_not_hardlinked(extension):
    extension._acquire('GC10-DET_a00_Q_R1')
    spec=extension.prepare_training('GC10-DET_a00_Entropy_A_R1')
    data=extension.dataset('GC10-DET')
    import yaml
    conf=yaml.safe_load(Path(spec['data_yaml_path']).read_text())
    assert 'test' not in conf
    copied=next((Path(conf['path'])/'labels').glob('*.txt'))
    source=Path(data['labels'][copied.stem]['path'])
    before=source.read_bytes();copied.write_bytes(b'local cache mutation')
    assert source.read_bytes()==before
    with pytest.raises(ExtensionGuardError,match='seal mismatch'):
        extension.prepare_training('GC10-DET_a00_Entropy_B_R1')


def test_worker_authorization_and_cold_start_lifecycle_without_real_training(extension,monkeypatch):
    import sys
    from types import SimpleNamespace
    import yaml
    from defect_al.entropy_training import run_cell
    ext=Extension(extension.root,'integration');ext.initialize_registry()
    ext.acquire('GC10-DET',0,1)
    auth=ext.root/'test_authorization.json'
    write_json_new(auth,{'status':'AUTHORIZED','mode':'integration','freeze_sha256':ext.freeze_sha,
                        'locked_test_access':False,'allowed_cells':['GC10-DET_a00_Entropy_A_R1','GC10-DET_a00_Entropy_B_R1']})
    initializations=[]; train_calls=[]; val_splits=[]
    class FakeYOLO:
        def __init__(self,path,verbose=False):initializations.append(path)
        def add_callback(self,*args):pass
        def train(self,**args):
            train_calls.append(args)
            output=Path(args['project'])/args['name'];(output/'weights').mkdir(parents=True)
            (output/'weights/best.pt').write_bytes(b'synthetic validation best')
            (output/'weights/last.pt').write_bytes(b'synthetic last')
            (output/'args.yaml').write_text(yaml.safe_dump(args))
            self.trainer=SimpleNamespace(save_dir=output,epoch=99)
        def val(self,**args):val_splits.append(args['split'])
    monkeypatch.setattr('defect_al.entropy_training.verify_environment',lambda _: {'synthetic':True})
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(set_device=lambda _:None,empty_cache=lambda:None)))
    a=run_cell(ext.root,'integration','GC10-DET_a00_Entropy_A_R1',auth)
    b=run_cell(ext.root,'integration','GC10-DET_a00_Entropy_B_R1',auth)
    assert initializations[0]==initializations[2]==ext.protocol['pretrained']['path']
    assert train_calls[0]['seed']!=train_calls[1]['seed']
    assert all(c['resume'] is False for c in train_calls)
    assert a['manifest_sha256']==b['manifest_sha256']
    assert a['training_config_sha256']!=b['training_config_sha256']
    assert val_splits==['val','val']
    assert ext.policy_checkpoint('GC10-DET_a00',2)['sha256']==a['best_sha256']


def test_official_training_rejects_fake_pass_without_real_cells(extension):
    passed=extension.root/'fake_pass.json'
    write_json_new(passed,{'status':'PASS','freeze_sha256':extension.freeze_sha})
    auth=extension.root/'fake_official_authorization.json'
    write_json_new(auth,{'status':'AUTHORIZED','mode':'official','freeze_sha256':extension.freeze_sha,
                        'locked_test_access':False,'allowed_cells':[c['node_id'] for c in extension.plan],
                        'integration_pass_path':str(passed),'integration_pass_sha256':file_sha256(passed)})
    ext=Extension(extension.root,'integration');ext.initialize_registry()
    with pytest.raises(ExtensionGuardError,match='not COMPLETE'):
        extension.verify_authorization(auth,'GC10-DET_a00_Entropy_A_R1')
