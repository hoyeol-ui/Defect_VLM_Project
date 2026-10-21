"""Final read-only input/runtime audit plus an explicit execution-code revision seal."""
from pathlib import Path
import sys,os,json,hashlib,shutil,platform,subprocess,importlib.metadata as meta,datetime
P=Path(__file__).resolve().parent;B=P.parents[1];sys.path.insert(0,str(P))
from audit import read,sha,put
from campaign_lib import load_matrix,ordered_new,cell_id,kst

def main():
    assert not (P/'campaign/CAMPAIGN_ACTIVE.lock').exists()
    assert not (B/'research_extensions/GPU_TRAINING_ACTIVE.lock').exists()
    archive=P/'execution_revisions/20261021_start'
    old=read(archive/'execution_freeze.json')
    changed={name:{'before':h,'after':sha(P/name)} for name,h in old['files'].items() if sha(P/name)!=h}
    assert set(changed)=={'cell.py','acquire.py'},changed
    for name,h in old['runtime_files'].items():assert sha(Path(name))==h,('Numerical runtime drift',name)
    assert read(P/'evidence/execution_checks.json')['status']=='PASS'
    assert read(P/'evidence/cpu_checks.json')['status']=='PASS'
    files_checked=0;sample_checked=0;datasets={}
    for ds in ['GC10-DET','NEU-DET']:
        file=P/'inputs'/ds/'index.json'
        assert sha(file)==old['files'][file.relative_to(P).as_posix()]
        index=read(file);groups={part:{r['id'] for r in index[part]} for part in ['train','dev']}
        split=B/'output/dataset_phase1b_v2_20260919'/ds
        allgroups={part:set((split/(part+'_sample_ids.txt')).read_text().splitlines()) for part in ['train','dev','test']}
        for part in ['train','dev']:
            assert len(groups[part])==len(index[part]) and groups[part]==allgroups[part]
            for r in index[part]:
                for key in ['image','label']:
                    assert sha(r[key])==r[key+'_sha'],(ds,part,r['id'],key)
                    files_checked+=1
                sample_checked+=1
        assert not (groups['train']&groups['dev']) and not ((groups['train']|groups['dev'])&allgroups['test'])
        datasets[ds]={part:len(ids) for part,ids in allgroups.items()}
    cfg=read(P/'config/protocol.json');args=read(P/'config/common_args.json')
    assert args['epochs']==100 and args['batch']==8 and args['imgsz']==640 and args['optimizer']=='SGD'
    assert args['cos_lr'] is True and args['close_mosaic']==10 and args['amp'] is True
    assert cfg['seed_pairs']==[{'acquisition':2026091901,'learner':2026091900},{'acquisition':2026091902,'learner':2026091908},{'acquisition':2026091903,'learner':2026091909}]
    for name in ['config/protocol.json','config/common_args.json']:
        assert sha(P/name)==old['files'][name]
    full=read(P/'evidence/full_reuse_check.json');run=B/full['source']
    assert all(full['checks'].values()) and sha(run/'train/weights/best.pt')==full['best_sha']
    assert read(run/'RESULT.json')['epochs']==100
    env=read(run/'ENVIRONMENT.json')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch,ultralytics
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(2026091900,deterministic=True)
    live={'python':platform.python_version(),'torch':torch.__version__,'torchvision':meta.version('torchvision'),
        'ultralytics':ultralytics.__version__,'numpy':meta.version('numpy'),'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0),'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()}
    assert live==env,{'current':live,'full':env}
    assert torch.cuda.device_count()==1
    for name,v in old['versions'].items():assert meta.version(name)==v,('version drift',name)
    backend={'matmul_TF32':torch.backends.cuda.matmul.allow_tf32,'cudnn_TF32':torch.backends.cudnn.allow_tf32,
        'cudnn_benchmark':torch.backends.cudnn.benchmark,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
    recorded=read(run/'training_backend.json')
    assert all(recorded[k]==v for k,v in backend.items()),(recorded,backend)
    ordered=ordered_new(load_matrix());free=shutil.disk_usage(P).free;minimum=101*161_000_000+8*2**30
    assert free>minimum
    # Only execution bookkeeping changed. Preserve the original seal and all scientific anchors.
    newfiles=dict(old['files'])
    for name in changed:newfiles[name]=sha(P/name)
    for f in P.glob('*.py'):newfiles[f.name]=sha(f)
    new={'files':newfiles,'runtime_files':old['runtime_files'],'versions':old['versions'],
        'previous_freeze_sha':sha(archive/'execution_freeze.json'),'revision_date_KST':kst(),
        'science_anchors_unchanged':{name:old['files'][name] for name in old['files'] if name.startswith(('config/','inputs/','selections/','src/'))},
        'revision':'Execution telemetry, sequential fail-stop controller and diagnostics only; no training/selection math or scientific settings changed.'}
    put(P/'evidence/execution_freeze.json',new)
    (archive/'CHANGE_LOG_KO.md').write_text('# 정규 실행 준비 구현 변경\n\n'+
        '- 학습 완료 시각·초기 환경·training_seconds를 별도 저장하도록 cell.py 기록 보강. 학습 kwargs와 수치 소스는 동일.\n'+
        '- acquire.py에 예측 없음·정확 동점·cutoff ID 규칙·ID 순위 기술통계와 재현 검사를 추가. 점수·후보 정리·선택 규칙 불변.\n'+
        '- 순차101셀 실행 원장·실측 ETA·GPU 소유 프로세스·실패 즉시 중단 제어 추가. 전체 동점이면 원인 보고 후 중단하며 대체 selector를 적용하지 않음.\n'+
        '- config, 입력 index, 기존72개 선택 manifest, src 수치 계산 파일 및 설치 패키지 해시 불변. 변경 전 소스·봉인 보존.\n',encoding='utf-8')
    report={'status':'PASS','checked_KST':kst(),'datasets':datasets,'input_samples_checked':sample_checked,'input_files_sha_checked':files_checked,
        'live_runtime_equals_reused_full_runtime':live,'backend_matches_full':backend,'all_scientific_anchors_unchanged':True,
        'GC10_full_reuse_best_sha':full['best_sha'],'GC10_full_reuse_checked':True,
        'free_disk_GiB':free/2**30,'minimum_remaining_disk_GiB':minimum/2**30,
        'GPU_lock_absent':True,'campaign_lock_absent':True,'new_training_cells':len(ordered),
        'first_cells':[cell_id(r) for r in ordered[:11]],'test_images_labels_or_inference_accessed':False,
        'execution_code_changes':changed,'new_execution_freeze_sha':sha(P/'evidence/execution_freeze.json')}
    put(P/'evidence/execution_start_preflight.json',report)
    from integrity import verify
    verify();print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
