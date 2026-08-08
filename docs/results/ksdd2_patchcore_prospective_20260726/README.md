# KSDD2 PatchCore prospective 결과 패키지

이 폴더는 `KSDD2_PATCHCORE_G1_G4_V1` 실행의 발표·검토용 compact
evidence package다.

- 최종 authorization: `BOUNDED_SCREEN`
- Frozen commit:
  `f10fd92985d63408212edd24e80d741de30ba5f7`
- 실행 branch: `codex/ksdd2-prospective-run`
- 전체 한국어 분석: `RESULT_ANALYSIS_KO.md`
- G1 결과: `g1/`
- G2–G4 결과: `g2_g4/`
- 사후 shortcut 점검: `posthoc_diagnostics/`

각 원본 결과 옆의 `.sha256` 파일로 복사 후 무결성을 확인할 수 있다.
사후 shortcut 점검은 preregistered gate가 아니며 frozen G1–G4 판정을
변경하지 않는다.

전체 raw run은 다음 clean worktree에 보존되어 있다.

`C:\Users\user\Desktop\vlm\Defect_VLM_Project_ksdd2_frozen\runs\ksdd2_patchcore_prospective_v1`

Official test image와 mask는 읽지 않았으며 detector/segmentation learner도
실행하지 않았다.
