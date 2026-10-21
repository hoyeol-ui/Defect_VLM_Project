"""Read-only campaign snapshot and manuscript handoff; no ML imports or training."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, sys, json, csv, io, hashlib, subprocess, re

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
B = Path('C:/Users/user/Desktop/vlm')
REPO = B / 'Defect_VLM_Project'
P = B / 'research_extensions/final_label_efficiency_20261021_01'
OUT_NORMAL = REPO / 'docs/results/final_label_efficiency_handoff_20261022'
# The preserved revision tree has legitimate paths beyond the legacy Windows 260-char limit.
OUT = Path('\\\\?\\' + str(OUT_NORMAL))
BUILD = B / 'output/macbook_handoff_20261022'
BRANCH = 'codex/final-label-efficiency-handoff-20261022'
KST = timezone(timedelta(hours=9))
registry = []

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def git(*args):
    return subprocess.run(['git', *args], cwd=REPO, check=True, capture_output=True).stdout

def put(rel, data, origin=None, scope='fixed source or document'):
    if isinstance(data, str):
        data = data.encode('utf-8')
    dst = OUT / rel
    assert not dst.exists(), dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    registry.append(dict(path=Path(rel).as_posix(), sha256=sha(data), bytes=len(data),
                         source=origin, scope=scope))

def copy(src, rel=None, scope='fixed source or document'):
    src = Path(src)
    if rel is None:
        rel = src.relative_to(B)
    put(rel, src.read_bytes(), str(src), scope)

def write_json(rel, value, scope='handoff metadata'):
    put(rel, json.dumps(value, ensure_ascii=False, indent=2) + '\n', scope=scope)

def tree_text(folder):
    for root, dirs, files in os.walk(folder, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in {'__pycache__', 'raw', 'weights', 'mpl',
                    'view', 'Ultralytics', 'settings', 'sealed_dev', '.git'} and not d.endswith('_geometry_raw'))
        for name in sorted(files):
            f = Path(root) / name
            if f.suffix.lower() in {'.py', '.md', '.csv', '.json', '.jsonl', '.yaml', '.yml', '.txt', '.log'}:
                yield f

def main():
    assert not OUT.exists(), OUT
    OUT.mkdir(parents=True)
    started = datetime.now(KST).isoformat(timespec='seconds')
    base = git('rev-parse', 'HEAD').decode().strip()
    status_before = git('status', '--porcelain=v1', '--untracked-files=no')
    write_json('LOCAL_REPOSITORY_STATE.json', dict(base_commit=base,
        working_branch=git('branch', '--show-current').decode().strip(),
        tracked_changes_at_start=status_before.decode('utf-8', errors='replace'),
        policy='Original checkout, branch, index, and historical working changes remain untouched.'))

    # Verify the frozen training inputs/code and runtime without importing numerical libraries.
    freeze = read(P / 'evidence/execution_freeze.json')
    checks = []
    for rel, expected in freeze['files'].items():
        f = P / rel
        assert sha(f.read_bytes()) == expected, ('campaign drift', rel)
        checks.append(dict(kind='campaign', source=rel, sha256=expected))
    for name, expected in freeze['runtime_files'].items():
        f = Path(name)
        assert sha(f.read_bytes()) == expected, ('runtime drift', name)
        checks.append(dict(kind='runtime', source=name, sha256=expected))
    write_json('FROZEN_SOURCE_VERIFICATION.json', dict(status='PASS', checked_at_KST=started,
        files=checks, numerical_libraries_imported=False, training_started=False,
        final_test_inference=False))

    # Capture status+ledger+events inside one unchanged controller heartbeat window.
    for attempt in range(10):
        status_raw = (P / 'campaign/status.json').read_bytes()
        ledger_raw = (P / 'campaign/execution_ledger.csv').read_bytes()
        events_raw = (P / 'campaign/events.jsonl').read_bytes()
        status_ko = (P / 'campaign/STATUS_KO.md').read_bytes()
        if status_raw == (P / 'campaign/status.json').read_bytes():
            break
    else:
        raise RuntimeError('Could not capture a stable controller heartbeat window')
    status = json.loads(status_raw.decode('utf-8-sig'))
    rows = list(csv.DictReader(io.StringIO(ledger_raw.decode('utf-8-sig'))))
    assert len(rows) == 102
    assert sum(r['state'] == 'SUCCEEDED' for r in rows) == status['new_succeeded']
    assert sum(r['state'] == 'RUNNING' for r in rows) == status['running']
    assert sum(r['state'] == 'REUSED' for r in rows) == status['reused_models']
    assert not status['test_inference_performed']
    for name, data in [('status.json', status_raw), ('execution_ledger.csv', ledger_raw),
                       ('events.jsonl', events_raw), ('STATUS_KO.md', status_ko)]:
        put(f'research_extensions/{P.name}/campaign/{name}', data, str(P / 'campaign' / name),
            'immutable snapshot; not a live monitor')

    for src in sorted((B / 'docs').glob('20261021_*')):
        if src.is_file():
            copy(src)
    for src in sorted((B / 'docs/figures').glob('*')):
        if src.is_file():
            copy(src)
    for folder in ['journal_manuscript_jkics/manuscript', 'journal_manuscript_jkics/figures',
                   'journal_manuscript_jkics/submission_check', 'thesis_manuscript']:
        for src in tree_text(B / folder):
            copy(src)
        # Images, editable vectors and the authored Word file are preserved byte-for-byte.
        for src in sorted((B / folder).glob('*')):
            if src.is_file() and src.suffix.lower() in {'.docx', '.png', '.svg'}:
                copy(src)
    copy(B / 'journal_manuscript_jkics/README.md')
    work = B / 'journal_manuscript_jkics/work'
    for name in ['build_manuscript.py', 'inspect_reference.py', 'final-style-evidence.json',
                 'package-qa.json', 'template-style-evidence.json', 'template_structure.json',
                 'official_submission_guidelines.pdf', 'official_submission_guidelines.txt',
                 'official_submission_template.docx']:
        if (work / name).exists():
            copy(work / name)
    attached = Path('C:/Users/user/Downloads/논문투고양식.docx')
    assert sha(attached.read_bytes()) == sha((work / 'official_submission_template.docx').read_bytes())

    # Frozen recipe, input membership/hash manifests, selections and implementation revisions.
    for src in sorted(P.glob('*')):
        if src.is_file():
            copy(src)
    for folder in ['src', 'config', 'inputs', 'evidence', 'selections', 'execution_revisions']:
        for src in tree_text(P / folder):
            copy(src)
    deps = B / 'research_extensions/full_data_26s_sgd_cosine_640_20261018_01'
    copy(deps / 'windows_job.py')
    for name in ['config/common_args.json', 'README_KO.md', 'controller.py']:
        if (deps / name).is_file():
            copy(deps / name, Path('dependency_reference') / deps.name / name)

    # Recovery/aquisition metadata includes per-image raw geometry, but not multi-GB NPZ arrays.
    for src in tree_text(P / 'campaign/recovery'):
        copy(src, scope='archived recovery evidence; historical status retained')
    for src in tree_text(P / 'acquisition_diagnostics'):
        if src.name.endswith('diagnostics.json') or src.suffix.lower() in {'.csv', '.jsonl'}:
            copy(src, scope='acquisition evidence captured after cutoff; completed reports only')

    checkpoints = []
    completed_ids = {r['cell_id'] for r in rows if r['state'] == 'SUCCEEDED'}
    run_names = {'train/results.csv', 'train/args.yaml'}
    for r in rows:
        run = P / 'runs' / r['cell_id']
        if r['state'] == 'SUCCEEDED':
            for src in sorted(run.glob('*')):
                if src.is_file() and src.suffix.lower() in {'.json', '.jsonl', '.yaml', '.log'}:
                    copy(src, scope='completed model evidence; development metrics only')
            for name in sorted(run_names):
                src = run / name
                if src.exists():
                    copy(src, scope='completed model training history')
            for name in ['best.pt', 'last.pt']:
                f = run / 'train/weights' / name
                checkpoints.append(dict(cell_id=r['cell_id'], state=r['state'],
                    source=str(f), bytes=f.stat().st_size, sha256=sha(f.read_bytes()),
                    included_in_bundle=False, role=name))
        elif r['state'] == 'REUSED':
            reuse = Path(r['reuse_path'])
            if not reuse.is_absolute():
                reuse = B / reuse
            if reuse.is_dir():
                reuse = reuse / 'train/weights/best.pt'
            if reuse.is_file():
                checkpoints.append(dict(cell_id=r['cell_id'], state=r['state'], source=str(reuse),
                    bytes=reuse.stat().st_size, sha256=sha(reuse.read_bytes()),
                    included_in_bundle=False, role='reused best checkpoint'))
                for name in ['RESULT.json', 'TRAINING_COMPLETE.json', 'best_selection.json',
                             'sealed_dev_metrics.json', 'resolved_args.json', 'loader_ids.json']:
                    src = reuse.parents[2] / name
                    if src.exists():
                        copy(src, f'reused_gc10_full/{name}', 'reused historical model; development only')
            else:
                checkpoints.append(dict(cell_id=r['cell_id'], state=r['state'], source=str(reuse),
                    recorded_best_sha=r.get('best_sha'), included_in_bundle=False,
                    note='See full reuse evidence for source checkpoint location.'))
    write_json('CHECKPOINT_INDEX.json', checkpoints)

    # Preserve the pre-existing working changes without silently adopting historical claims.
    changed = git('diff', '--name-only', '-z', 'HEAD').split(b'\0')
    for raw in changed:
        if not raw:
            continue
        rel = raw.decode('utf-8')
        f = REPO / rel
        if f.is_file():
            copy(f, Path('historical_working_copy') / rel,
                 'pre-existing historical working copy; not the current YOLO26s manuscript')
    for f in sorted(REPO.glob('*')):
        if f.is_file() and f.suffix.lower() in {'.md', '.csv'} and f.stat().st_size < 1000000:
            tracked = subprocess.run(['git', 'ls-files', '--error-unmatch', '--', f.name],
                                     cwd=REPO, capture_output=True).returncode == 0
            if not tracked:
                copy(f, Path('historical_working_copy') / f.name,
                     'historical thesis writing context; not the current protocol')

    acq = []
    for f in sorted((OUT / f'research_extensions/{P.name}/acquisition_diagnostics').rglob('*_diagnostics.json')):
        z = read(f)
        assert z['gate'] == 'PASS' and z['ranking_validated'] and z['selection_within_remaining_pool']
        assert z['geometry_validation']['invalid_eligible_boxes'] == 0
        assert all(x['repeat_equal'] for x in z['repeat_checks'])
        acq.append(dict(dataset=z['dataset'], acquisition_seed=z['acquisition_seed'], round=z['round'],
            eligible_n=z['eligible_n'], selected_n=z['query_n'],
            no_prediction_fraction=z['no_prediction_fraction'],
            exact_tie_fraction=z['candidates_in_exact_ties_fraction'],
            cutoff_ID_fraction=z['selected_by_cutoff_ID_tie_fraction'],
            invalid_eligible_boxes=z['geometry_validation']['invalid_eligible_boxes'],
            selected_mean_ID_percentile=z['selected_mean_global_ID_percentile'],
            ID_KS_D=z['ID_rank_descriptive_KS_D']))
    write_json('ACQUISITION_SUMMARY.json', acq)

    # Record exclusions honestly: this is a manuscript/code handoff, not a portable training image.
    exclusions = []
    for folder in ['campaign/recovery', 'acquisition_diagnostics', 'runs', 'execution_revisions']:
        for src in (P / folder).rglob('*'):
            if src.is_file() and src.suffix.lower() in {'.npz', '.pt', '.pth'}:
                exclusions.append(dict(source=str(src), bytes=src.stat().st_size,
                    reason='Windows retains originals; raw tensor arrays and model weights omitted.'))
    write_json('EXCLUDED_LARGE_ARTIFACTS.json', exclusions)
    snapshot = dict(created_KST=started, controller_status_KST=status['updated_KST'],
        snapshot_only=True, live_training_host='Windows RTX 3070, C:/Users/user/Desktop/vlm',
        controller_pid=status['controller_pid'], current_cell=status['current_cell'],
        current_epoch=status['current_epoch'], new_succeeded=status['new_succeeded'],
        new_training_target=101, reused_models=status['reused_models'], pending=status['pending'],
        running=status['running'], failed=status['failed'],
        remaining_GPU_hours=status['remaining_training_plus_acquisition_hours'],
        estimated_models_complete_KST=status['estimated_models_complete_KST'],
        estimate_excludes=status['estimate_excludes'], test_inference_performed=False,
        protocol_sha=status['protocol_sha'], execution_freeze_sha=status['execution_freeze_sha'],
        github_repository='https://github.com/hoyeol-ui/Defect_VLM_Project',
        handoff_branch=BRANCH, parent_commit=base, acquisition_reports=len(acq),
        completed_models_copied=sorted(completed_ids), package_contains_no_training_environment=True)
    write_json('SNAPSHOT.json', snapshot)

    put('README_FIRST.md', f'''# 맥북 인계 — 2026년 10월 22일

기준: **{status['updated_KST']}**. 이 패키지는 해당 시점의 스냅샷이다. Windows의 정규 GPU 캠페인은 별도로 계속 실행 중이다. 맥북에서 학습·획득·개발 평가·최종 테스트를 시작하지 않는다.

신규 학습 {status['new_succeeded']}/101회 완료, 기존 GC10 Full 1개 재사용, 현재 {status['current_cell']} {status['current_epoch']}/100 epoch. 잔여 학습+획득 추정 **{status['remaining_training_plus_acquisition_hours']:.1f} GPU시간**, 모델 완료 추정 {status['estimated_models_complete_KST']}. 최종 테스트·중단 시간은 제외한 변동 가능한 추정이다.

## 바로 열 파일

1. [맥북 Codex 요청](MACBOOK_CODEX_PROMPT.md): 내용을 새 맥북 Codex 채팅에 붙여넣는다.
2. [최신 학술지 Word](journal_manuscript_jkics/manuscript/JKICS-FrozenDINOv2-LabelEfficiency-20261021.docx), [본문 대조본](journal_manuscript_jkics/manuscript/JKICS-FrozenDINOv2-LabelEfficiency-20261021.md).
3. [최종 프로토콜](docs/20261021_final_experiment_protocol.md), [연구 감사](docs/20261021_research_audit.md), [Entropy 복구](docs/20261021_entropy_recovery_report.md).
4. [실행 현황 스냅샷](research_extensions/{P.name}/campaign/STATUS_KO.md), [실행 원장](research_extensions/{P.name}/campaign/execution_ledger.csv), [획득 진단 요약](ACQUISITION_SUMMARY.json).
5. [학위논문 관리 경계](thesis_manuscript/README.md), [투고 지침 및 게재논문 검토](journal_manuscript_jkics/submission_check/20261021_JKICS_review.md).

## 내용과 버전 해석

- `docs/`: 10월21일 감사·실험 설계·원고·교수님 보고 그림. 과거 시각이 남은 문서는 이력이다. 최신 현황은 SNAPSHOT과 원장을 우선한다.
- `journal_manuscript_jkics/`: 투고용 8쪽 익명 Word·Markdown·영문 그림·공식 양식·작성 코드·QA. 결과·저자·결론 일부는 평가 대기 상태다.
- `research_extensions/`: frozen 실행 소스, 설정, train/dev 경로·해시, 선택 manifest, 완료 셀의 로그·개발 평가, 코드 변경 전후·복구 회귀 근거. 소스를 실행하지 않고 검토한다.
- `historical_working_copy/`: 기존 저장소의 미커밋 문서·스크립트 수정과 작은 학위논문 작성 자료를 별도 보존했다. 현 최종 프로토콜과 혼합하지 않는다.
- GitHub 브랜치의 다른 기존 경로는 10월18일까지의 연구 이력을 포함한다. 옛 문서의 중단 지시·더 작은 모델·이전 성능은 현재 프로토콜을 대체하지 않는다.
- 과거 `FAILURE.json`과 원장의 traceback은 복구 이력으로 보존되어 있다. 해당 GC10 셀은 checkpoint를 재사용하여 SUCCEEDED로 복구됐다. 현 실패 여부는 스냅샷·원장의 state로 판단한다.

## Windows에 남는 자료

원본 이미지·라벨, 모델 best/last 가중치, 전체 raw tensor NPZ, Python/CUDA/렌더러 설치 환경은 포함하지 않는다. [가중치 인덱스](CHECKPOINT_INDEX.json), [대용량 제외 목록](EXCLUDED_LARGE_ARTIFACTS.json)과 소스 freeze에 경로·해시가 있다. 이미지별 낮은 confidence 역전 좌표와 원래 인덱스의 JSONL 진단은 포함한다. 원본은 Windows에서 그대로 보존된다. 이 ZIP은 학습 이주·재시작용이 아니다.

논문 PDF 원문을 통째로 재배포하지 않고 지침 검토 문서의 공식 링크를 제공한다. 이 ZIP은 최종 테스트 결과나 모든 학습 완료를 의미하지 않는다. GitHub 및 ZIP은 자동 갱신되지 않는다.

## 무결성 확인

이 디렉터리에서 `python3 verify_manifest.py`를 실행하면 첨부 파일 해시와 개수를 확인한다. 연구 코드나 ML 패키지를 불러오지 않는다. 수정 전 검증하고, 원고 편집 후에는 원래 manifest와 달라지는 것이 정상이다.

Windows는 전원을 유지하고 절전·최대 절전·재부팅·종료를 피한다. 화면 끄기·잠금은 가능하다. 이 스냅샷으로 실제 프로세스가 지금도 살아 있다고 판단하지 않는다.
''')

    prompt = '''# 맥북 Codex 인계 요청

첨부 ZIP을 풀고 이 연구의 원고·그림·근거 정리 작업을 이어가 주세요. 질문만 하거나 계획만 제시하지 말고, 확인 가능한 문서 작업부터 수행해 주세요.

## 처음 확인할 위치

- ZIP에는 `Defect_AL_MacBook_20261022/project/` 아래에 GitHub 커밋의 전체 소스 스냅샷이 있습니다. 프로젝트 기준 최신 인계 디렉터리는 `docs/results/final_label_efficiency_handoff_20261022/`입니다. 이 디렉터리를 이하 H라고 부릅니다.
- ZIP 없이 저장소에서 받는 경우 `https://github.com/hoyeol-ui/Defect_VLM_Project`의 `codex/final-label-efficiency-handoff-20261022` 브랜치를 사용하세요. 기본 main이나 이전 10월18일 인계 브랜치만 보고 최신 상태로 판단하지 마세요.
- 먼저 H/README_FIRST.md, H/SNAPSHOT.json, H/ACQUISITION_SUMMARY.json을 읽고, `python3 H/verify_manifest.py`로 첨부 무결성을 확인하세요. H는 실제 디렉터리 경로로 바꿔 실행하세요. 파일 안의 Windows 절대 경로는 원본 출처이며 macOS에서 실행할 경로가 아닙니다.

## 진행 중인 Windows 실험과 작업 경계

Windows RTX3070에서 이미 승인한 정규 캠페인을 순차 실행 중입니다. ZIP과 GitHub 기록은 생성 시점 스냅샷이며 실시간 상태가 아닙니다. 이 사본으로 현재 완료·실패 여부를 단정하지 마세요. 맥북에서 Windows 학습을 이주·재개하거나 중복 실행하지 마세요. 컨트롤러, 셀, 획득, 준비/봉인 스크립트, 개발 평가 및 최종 테스트를 실행하지 마세요. 고정 실행 소스를 편집하거나 원격 Windows 파일에 덮어쓰지 마세요. Windows heartbeat의 감시 역할을 이 맥북 사본이 대신한다고 가정하지 마세요.

실험 프로토콜은 GC10-DET/NEU-DET의 고정 F518 분할, 10/15/20/25/30/35% 여섯 라벨 예산, Random/Frozen DINOv2-FF/Entropy, YOLO26s, 100epoch, imgsz640, batch8, nbs64, SGD cosine, close_mosaic10, 세 개 대응 반복입니다. 전체 물리 모델102개 중 검증된 GC10 Full1개를 재사용하고 신규101회를 학습합니다. 100% Fully Supervised도 각 learner seed별로 확보합니다. 신규 예산·알고리즘·추가 실험·새 성능 최적화는 하지 마세요. 정확한 seed·입력 수·평가 규칙은 H/docs/20261021_final_experiment_protocol.md와 H/research_extensions/final_label_efficiency_20261021_01/config/를 기준으로 하세요.

첫 NEU Full과 각 데이터셋 공유10%를 우선하며, 공유10%는 세 전략이 같은 checkpoint를 공유합니다. 이후 각 예산 모델은 고정된 cold start 학습을 사용합니다. 최종 테스트는 전체 모델·설정 확정 뒤 Windows에서 별도 평가 단계로만 수행합니다.

## Entropy 복구 이력

GC10 첫 공유10% 모델은 100epoch 학습을 끝냈지만 학습 후 adapter가 confidence 필터 전 모든 raw8400박스의 좌표를 검사하여 매우 낮은 신뢰도 역전 박스에서 실패했습니다. 기존 confidence>0.25에 해당하는 후보에만 기존 좌표 guard를 적용하도록 entropy_adapter.py와 entropy.py를 수정했습니다. 좌표 clamp, confidence/NMS/Entropy 공식/ID 규칙 변경은 하지 않았습니다. 유효 후보 좌표 오류는 계속 중단 조건입니다.

정상입력1958이미지 old/new 직접 동일, 두 데이터셋 전체2862이미지 eligible-only oracle 동일, 순위·동점·선택 동일성, GC10 기존 checkpoint·로그 불변과 GC10 정식 R1선택81개 일치를 확인했습니다. 기존 best.pt로 미완료 후처리·개발 평가만 복구했고 재학습은 하지 않았습니다. 회귀 게이트 통과 뒤 캠페인이 재개됐습니다. H/docs/20261021_entropy_recovery_report.md와 H/research_extensions/final_label_efficiency_20261021_01/campaign/recovery/를 읽으세요. old FAILURE.json이나 과거 감사 시각을 현재 미복구 실패로 오해하지 마세요.

NEU Full모델을 사용했던 복구 중 진단과, 나중에 정상 완료한 NEU 공유10%의 실제 획득 결과는 서로 다른 것입니다. 실제 획득 진단은 H/ACQUISITION_SUMMARY.json과 해당 acquisition_diagnostics를 사용하세요. ID 순위 통계만으로 클래스·출처 편향이 없다고 주장하지 마세요.

## 이어서 수행할 문서 작업

1. 최신 Word 편집 원본 H/journal_manuscript_jkics/manuscript/JKICS-FrozenDINOv2-LabelEfficiency-20261021.docx와 Markdown 대조본을 읽고, 확정 가능한 서론·관련 연구·연구 방법을 다듬어 주세요. 원본을 보존하고 날짜를 붙인 새 편집본 및 변경 기록을 만드세요. Word 편집 시 문서 스킬이 있으면 사용하고 렌더링으로 레이아웃을 검증하세요. 결과 대기 문구를 실제 결과처럼 바꾸지 마세요.
2. J-KICS 한국통신학회논문지 투고 지침·양식 검토는 H/journal_manuscript_jkics/submission_check/20261021_JKICS_review.md와 work/official_submission_template.docx를 기준으로 하세요. 이미 게재된 USB cascaded U-Net, MG-AD, YOLO scooter 논문3편의 전개·설정 설명·결과 표 비교가 포함되어 있습니다. 필요하면 공식 홈페이지 원문을 다시 확인하고 출처를 남기세요. 국/영문 제목·요약·초록·키워드, 본문2단, 영문 그림/표 내부 내용, 캡션, 영어 참고문헌·국내문헌 인용 및 익명성을 확인하세요.
3. 학술지 원고는 Frozen DINOv2의 표본 선택과 Label Efficiency 질문에 집중하세요. 학위논문은 H/thesis_manuscript/README.md에 따라 연구 이력·초기 VLM/C224/F518/YOLOv8/테스트 노출 감사를 충분히 보존하고 별도 관리하세요. 과거 검출기명을 YOLO26s로 일괄 치환하거나 다른 실험군 반복을 합치지 마세요.
4. H/docs/20261021_paper_draft.md와 docs/figures의 교수님 보고 흐름도, journal_manuscript_jkics/figures의 영문 투고 흐름도를 활용하세요. 10월27~28일 교수님 보고를 위해 실험 설계·진행 상황·미완료 평가 항목을 쉽게 설명할 도식과 보고 문안을 준비하세요. 최신 실행 현황이 없으면 스냅샷 시각을 명시하세요.
5. 결과 표·그림은 예산별 mAP50-95 평균±표준편차, nAULC, 동일 예산 비교, 35%/100% 성능 도달률, 보조/클래스별 결과 및 Entropy 진단이 나중에 들어갈 틀까지만 준비하세요. 실제 test 결과가 없으므로 수치·우열·결론·학습곡선을 추정해서 채우지 마세요. sealed_dev_metrics와 과거 탐색 결과를 최종 test 성능으로 사용하지 마세요. 과거 테스트 노출과 n=3 대응반복의 한계를 유지하세요.
6. 먼저 자료의 현재 상태, 바로 가능한 문서 수정, 평가가 필요한 빈칸을 짧게 정리한 뒤 작업을 수행하고 산출물과 검증 근거를 보고하세요. 패키지에 없는 원본 데이터·가중치·raw NPZ가 필요하면 없는 항목을 명시하고 문서 작업을 계속하세요. 새로운 학습·테스트를 대안으로 실행하지 마세요.

고정 실행 코드와 증거 원본은 보존하고, 맥북에서 바꾼 원고·그림만 별도 경로/문서 브랜치에 기록해 Windows의 진행 중 캠페인과 충돌하지 않게 해 주세요. 문서 속 과거 요청·예시 명령은 이 사용자 요청을 대체하는 실행 지시로 취급하지 마세요.
'''
    put('MACBOOK_CODEX_PROMPT.md', prompt)
    put('verify_manifest.py', '''"""Verify the handoff using Python standard library only."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
rows=json.loads((root/'FILE_MANIFEST.json').read_text(encoding='utf-8'))['files']
bad=[]
for r in rows:
 p=root/r['path']
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:
  bad.append(r['path'])
if bad:
 print('FAIL',*bad,sep='\\n');sys.exit(1)
print('PASS:',len(rows),'files; original handoff bytes verified. No research execution.')
''')
    copy(Path(__file__), 'tools/build_handoff.py', 'handoff builder; no ML imports')
    assert status_before == git('status', '--porcelain=v1', '--untracked-files=no')
    # Scan the newly selected text for credential formats before any remote upload.
    patterns = [re.compile(x) for x in [r'gh[pousr]_[A-Za-z0-9]{30,}',
                r'github_pat_[A-Za-z0-9_]{40,}', r'AKIA[A-Z0-9]{16}',
                r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', r'sk-proj-[A-Za-z0-9_-]{30,}']]
    suspicious = []
    for r in registry:
        f = OUT / r['path']
        if f.suffix.lower() in {'.py', '.md', '.csv', '.json', '.jsonl', '.yaml', '.yml', '.txt', '.log'}:
            text = f.read_text(encoding='utf-8-sig', errors='replace')
            if any(pattern.search(text) for pattern in patterns):
                suspicious.append(r['path'])
    assert not suspicious, ('possible credentials; do not upload', suspicious)
    write_json('PACKAGE_QA.json', dict(status='PASS', stable_status_ledger_snapshot=True,
        source_and_runtime_hashes_verified=len(checks), completed_cells_copied=len(completed_ids),
        acquisition_reports_passed=len(acq), original_checkout_tracked_changes_unchanged=True,
        credentials_scan='PASS for selected text, no matched credential formats',
        datasets_weights_large_tensors_included=False, training_restarted=False, final_test_run=False,
        note='Word and figures copied byte-for-byte; no new manuscript edits or numerical results.'))
    # Manifest deliberately excludes itself to avoid a recursive hash.
    (OUT / 'FILE_MANIFEST.json').write_text(json.dumps(dict(files=registry,
        self_excluded='FILE_MANIFEST.json'), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (BUILD / 'snapshot_location.json').write_text(json.dumps(dict(output=str(OUT),
        branch=BRANCH, base_commit=base, status=status, files=len(registry)+1),
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(OUT), files=len(registry)+1,
        bytes=sum(r['bytes'] for r in registry), status_cutoff=status['updated_KST'],
        completed=status['new_succeeded'], remaining_GPU_hours=status['remaining_training_plus_acquisition_hours']),
        ensure_ascii=False))

if __name__ == '__main__':
    main()
