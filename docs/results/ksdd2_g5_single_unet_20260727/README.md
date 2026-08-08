# KSDD2 G5 single-learner compact evidence package

Protocol: `KSDD2_G5_SINGLE_UNET_V1`  
Frozen decision: **`TRANSLATION_NOT_OBSERVED_STOP`**  
Completed runs: **60/60**  
Official test accessed: **No**

이 폴더는 raw checkpoint와 개발 데이터 없이 G5 판정 및 해석을 검토할 수
있도록 만든 GitHub용 compact evidence package다.

## 핵심 결과

- PatchCore macro Dice: `0.6084`
- Random macro Dice: `0.5709`
- Paired absolute delta: `+0.0376`
- Acquisition-seed bootstrap 95% CI: `[+0.0196, +0.0570]`
- Frozen minimum effect: `+0.05`
- Pixel AP delta: `+0.0498`
- PatchCore normal FPR: `0.1601`
- Normal FPR delta: `+0.0106`

효과 방향, CI, 8/10 positive seed, leave-one-seed-out 및 safety 조건은
통과했지만, primary mean Dice delta가 frozen minimum effect `+0.05`에
미달하여 확장은 중단한다.

## 파일

- `RESULT_ANALYSIS_KO.md`: 결과의 통계적·연구적 해석과 claim boundary
- `g5_authorization_decision.json`: frozen Gate의 기계 판정
- `paired_seed_metrics.csv`: primary 통계 단위인 acquisition seed 10개의 paired 결과
- `posthoc/audit_summary.json`: annotation burden, query overlap, training-seed 진단
- `posthoc/SKEPTICAL_OBJECTIVE_AUDIT_KO.md`: 판정을 변경하지 않는 냉정한 사후 감사
- `ARTIFACT_MANIFEST.sha256`: 원본 compact artifact SHA-256

## 재현

코드와 frozen protocol은
`scripts/07_ksdd2_g5_bounded_learner/`에 있으며, post-hoc 감사 코드는
`scripts/08_ksdd2_g5_posthoc/`에 있다. raw learner checkpoints와 dataset은
저장소에 포함하지 않는다.

현재 판정 이후에는 hyperparameter, epoch, threshold 또는 backbone rescue가
승인되지 않는다. Official test도 계속 잠근다.
