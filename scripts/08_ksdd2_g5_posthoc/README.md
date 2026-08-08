# KSDD2 G5 skeptical post-hoc audit

이 폴더의 코드는 이미 동결되어 실행 중인 G5 learner를 수정하지 않는다.
60개 run과 frozen gate 판정이 끝난 뒤에만 별도의 설명용 진단을 수행한다.

## 현재 실행을 기다리고 자동 분석하기

기존 G5가 이미 실행 중이므로 **두 번째 trainer를 시작하지 않는다**.

```powershell
cd "C:\Users\user\Desktop\vlm\Defect_VLM_Project"

$python = "C:\Users\user\Desktop\vlm\Defect_VLM_Project\.python311\python.exe"

& $python `
  scripts\08_ksdd2_g5_posthoc\monitor_and_finalize.py
```

이 명령은 다음을 수행한다.

1. 기존 60-run 진행률을 60초마다 표시
2. 45분 동안 학습 로그가 갱신되지 않으면 중단 상태로 판정
3. 60/60 완료 후 frozen `analyze_g5.py` 판정 확인
4. 판정 파일이 없다면 동결된 분석 코드 실행
5. `audit_g5.py`를 실행하여 냉정한 post-hoc 보고서 생성

## 출력

```text
C:\Users\user\Desktop\vlm\Defect_VLM_Project_ksdd2_g5\
  runs\ksdd2_g5_single_unet_v1\posthoc_skeptical_audit\
```

주요 파일:

- `SKEPTICAL_OBJECTIVE_AUDIT_KO.md`
- `annotation_burden_by_seed.csv`
- `training_seed_pair_deltas.csv`
- `size_stratified_performance.csv`
- `audit_summary.json`

## trainer가 실제로 종료된 경우에만 재개

먼저 중복 프로세스가 없는지 확인한다.

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -like "*train_g5.py*" } |
  Select-Object ProcessId, CommandLine
```

아무 결과도 없고 60/60이 아니라면 다음 명령으로 동일 freeze에서 재개한다.
완료된 run은 seal을 확인하고 자동으로 건너뛴다.

```powershell
cd "C:\Users\user\Desktop\vlm\Defect_VLM_Project_ksdd2_g5"

& "C:\Users\user\Desktop\vlm\Defect_VLM_Project\.python311\python.exe" `
  scripts\07_ksdd2_g5_bounded_learner\train_g5.py --all
```

중간 결과를 보고 threshold, epoch, loss, architecture를 바꾸면 현재 G5의
prospective 근거가 무효화되므로 금지한다.
