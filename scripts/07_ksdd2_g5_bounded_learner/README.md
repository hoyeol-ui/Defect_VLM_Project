# KSDD2 G5 single-learner bounded screen

이 폴더는 `KSDD2_G5_SINGLE_UNET_V1`을 learner outcome 확인 전에
검토·동결하기 위한 protocol package다.

## 현재 허용된 작업

```powershell
cd "C:\Users\user\Desktop\vlm\Defect_VLM_Project"

.\.python311\python.exe `
  scripts\07_ksdd2_g5_bounded_learner\validate_prefreeze.py

.\.python311\python.exe `
  scripts\07_ksdd2_g5_bounded_learner\test_protocol.py
```

두 명령은 upstream G1–G4 artifact와 G5 설정의 정적 무결성만 확인한다.
Segmentation learner를 학습하거나 official test를 읽지 않는다.

## Freeze 전 확인

1. `protocol.md`의 claim boundary와 G5 threshold를 검토한다.
2. `config.yaml`의 learner 설정을 변경해야 한다면 지금 한 번만 변경한다.
3. `validate_prefreeze.py --write-ledger`로 exact G5 input ledger를 만든다.
4. Synthetic overfit과 image-mask alignment QA를 통과한다.
5. Code/config/protocol/input ledger를 clean Git commit에서 SHA-256 freeze한다.
6. Freeze 이후 learner outcome을 확인한다.

## 설정 경계

- Compact U-Net 한 개
- Fixed training seed 세 개
- PatchCore/Random × acquisition seed 10개 × training seed 3개 = 60 runs
- Fixed epoch 80
- No early stopping/checkpoint selection
- Development screen only
- Official test locked

## 구현·동결·실행

```powershell
$python = "C:\Users\user\Desktop\vlm\Defect_VLM_Project\.python311\python.exe"

& $python scripts\07_ksdd2_g5_bounded_learner\test_protocol.py
& $python scripts\07_ksdd2_g5_bounded_learner\test_learner.py
& $python scripts\07_ksdd2_g5_bounded_learner\qa_preflight.py
& $python scripts\07_ksdd2_g5_bounded_learner\validate_prefreeze.py --write-ledger

# clean Git commit을 만든 뒤 실행
& $python scripts\07_ksdd2_g5_bounded_learner\freeze_g5.py
& $python scripts\07_ksdd2_g5_bounded_learner\train_g5.py --all
& $python scripts\07_ksdd2_g5_bounded_learner\analyze_g5.py
```

`train_g5.py`와 `analyze_g5.py`는 freeze manifest가 없거나 변경된 경우
실행을 거부한다.
