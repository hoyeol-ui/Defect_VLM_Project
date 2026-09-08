"""The independent runner never silently repeats or resumes a training attempt."""
import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/entropy_extension/detached_tranche.py'
spec=importlib.util.spec_from_file_location('entropy_detached_test',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_completed_cell_is_only_verified():
    assert module.remaining_job_action({'state':'COMPLETE','attempt':1})=='VERIFY_AND_SKIP'


@pytest.mark.parametrize('state,attempt',[('RUNNING',1),('FAILED',1),('READY',1),('OTHER',0)])
def test_ambiguous_attempt_is_never_retried(state,attempt):
    with pytest.raises(module.ExtensionGuardError):module.remaining_job_action({'state':state,'attempt':attempt})


@pytest.mark.parametrize('state',['PENDING','READY','BLOCKED_MANIFEST'])
def test_only_unattempted_jobs_can_run(state):
    assert module.remaining_job_action({'state':state,'attempt':0})=='RUN'
