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
4. Code/config/protocol/input ledger를 clean Git commit에서 SHA-256 freeze한다.
5. Freeze 이후 learner outcome을 확인한다.

## 설정 경계

- Compact U-Net 한 개
- Fixed learner seed 한 개
- PatchCore/Random × acquisition seed 10개 = 20 runs
- Fixed epoch 80
- No early stopping/checkpoint selection
- Development screen only
- Official test locked

현재 폴더는 protocol과 pre-freeze validator만 포함한다. 실제 learner
implementation은 이 protocol이 승인된 뒤 동일 폴더에 추가하고, 실행 전에
함께 freeze해야 한다.
