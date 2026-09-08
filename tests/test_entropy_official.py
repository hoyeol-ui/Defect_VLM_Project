"""Official tranche guards and unchanged training semantics. No training runs."""
import copy
from pathlib import Path
import pytest

from defect_al.entropy_extension import ExtensionGuardError, build_plan
from defect_al.entropy_official import OfficialExtension, require_scope, verify_checkpoint_selection


@pytest.fixture
def scope():
    seeds = [{'dataset': ds, 'acquisition_index': i, 'acquisition_seed': i,
              'training_seed_a': i+1, 'training_seed_b': i+2}
             for ds in ('GC10-DET','NEU-DET') for i in range(10)]
    cells, queries = build_plan(seeds, {ds: {'query_count_per_round':1,'cumulative_counts':[1,2,3,4,5,6]}
                                      for ds in ('GC10-DET','NEU-DET')})
    tids = ['GC10-DET_a00','NEU-DET_a00']
    auth = {'status':'AUTHORIZED','mode':'official','official_freeze_sha256':'o','freeze_sha256':'e',
            'locked_test_access':False,'integration_reuse':False,'authorized_trajectories':tids,
            'allowed_cells':[c['node_id'] for c in cells if c['trajectory'] in tids],
            'allowed_queries':[q['node_id'] for q in queries if q['trajectory'] in tids]}
    return auth, cells, queries


def test_exact_twenty_cells_ten_queries(scope):
    auth,cells,queries=scope
    assert len(auth['allowed_cells'])==20 and len(auth['allowed_queries'])==10
    for node in auth['allowed_cells']+auth['allowed_queries']:
        require_scope(auth,'o','e',cells,queries,node)


@pytest.mark.parametrize('key,value', [('mode','integration'),('status','CANDIDATE'),('locked_test_access',True),
    ('integration_reuse',True),('official_freeze_sha256','wrong'),('freeze_sha256','wrong')])
def test_reject_wrong_seals_and_reuse(scope,key,value):
    auth,cells,queries=scope;auth[key]=value
    with pytest.raises(ExtensionGuardError): require_scope(auth,'o','e',cells,queries,'GC10-DET_a00_Entropy_A_R1')


def test_no_other_index(scope):
    auth,cells,queries=scope
    with pytest.raises(ExtensionGuardError,match='outside'):
        require_scope(auth,'o','e',cells,queries,'GC10-DET_a01_Entropy_A_R1')


@pytest.mark.parametrize('key',['allowed_cells','allowed_queries'])
def test_incomplete_or_duplicate_scope_rejected(scope,key):
    auth,cells,queries=scope;auth[key].pop()
    with pytest.raises(ExtensionGuardError): require_scope(auth,'o','e',cells,queries,'GC10-DET_a00_Entropy_A_R1')


def test_best_must_be_validation_max():
    history={'epoch':list(range(1,101)),'metrics/mAP50-95(B)':[0.2]*99+[0.1]}
    best={'train_results':history,'train_metrics':{'fitness':0.2}}
    last={'train_results':history,'train_metrics':{'fitness':0.1}}
    verify_checkpoint_selection(best,last)
    bad=copy.deepcopy(best);bad['train_metrics']['fitness']=0.1
    with pytest.raises(ExtensionGuardError):verify_checkpoint_selection(bad,last)


def test_official_worker_only_changes_guards_and_memory_observer():
    source=Path(__file__).resolve().parents[1]/'src/defect_al'
    original=(source/'entropy_training.py').read_text()
    official=(source/'entropy_official_training.py').read_text()
    expected=original.replace('from .entropy_extension import Extension, ExtensionGuardError, checked_file, inside, now, write_json_new',
        'from .entropy_extension import ExtensionGuardError, checked_file, inside, now, write_json_new\nfrom .entropy_official import OfficialExtension as Extension')
    expected=expected.replace('torch.cuda.empty_cache()', 'torch.cuda.empty_cache()\n            torch.cuda.reset_peak_memory_stats(0)')
    expected=expected.replace("'environment': environment, 'outcome_values_exposed': False,", "'environment': environment, 'outcome_values_exposed': False,\n                      'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(0),\n                      'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(0),")
    assert official==expected


def test_next_round_requires_both_learner_round_gate(tmp_path):
    ext=OfficialExtension.__new__(OfficialExtension)
    ext.root=tmp_path
    with pytest.raises(ExtensionGuardError,match='gate absent'):
        ext.validate_gate('GC10-DET_a00',1)


def test_integration_result_cannot_be_imported(tmp_path,monkeypatch):
    from defect_al.entropy_extension import Extension
    ext=OfficialExtension.__new__(OfficialExtension)
    ext.root=tmp_path;ext.official_sha='o'
    cid='GC10-DET_a00_Entropy_A_R1'
    monkeypatch.setattr(Extension,'completed',lambda *_:{'official_freeze_sha256':'o'})
    monkeypatch.setattr(ext,'node',lambda *_:{'kind':'training'})
    monkeypatch.setattr(ext,'state',lambda *_:{'artifact':str(tmp_path/'outcomes_private/integration'/cid/'cell_result.json')})
    with pytest.raises(ExtensionGuardError,match='integration namespace'):
        ext.completed(cid)
