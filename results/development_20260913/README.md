# Completed three-way development analysis — 2026-09-13

「소량 라벨 환경에서 특징 기반 능동학습을 활용한 금속 표면 결함 객체 탐지」

**Evidence: Development validation; statistical unit: acquisition seed n=10; learner A/B averaged within seed/round; locked-final: not used.** Endpoint identity grade A; overall conditional comparison grade B due to GPU/OS differences and incomplete historical environment fields.

[실험 결과 기록](EXPERIMENT_RESULT_BRIEF.md) · [해석](FINAL_DEVELOPMENT_INTERPRETATION.md) · [검증 보고](ANALYSIS_VALIDATION_REPORT.md) · [그림 설명](FIGURE_CAPTIONS.md) · [로컬 HTML 목록](index.html)

원래 검증된 package의 모든 85 payload 자료를 유지했다. 원표/그림의 수치는 바꾸지 않았다. [원래 실행 코드](../../scripts/analysis/development_20260913/original/)와 [공개 재현 코드](../../scripts/analysis/development_20260913/)를 구분해 보존했다.

- [tables/](tables/): 23종 CSV/Markdown 쌍. Physical 644, conceptual AL 720, A/B 평균 seed-round 360, seed nAULC 60. Full reference는 실제 learner 2개/dataset.
- [figures/](figures/): 대표 5종 × PNG/SVG/PDF.
- [evidence/](evidence/README.md): 기존 완료 200 training/100 acquisition/Full 4와 source/config의 public-safe 기록.
- [historical_geometry/](historical_geometry/): 기존 frozen post-hoc CSV와 provenance. 새 feature extraction 없음.
- [audit/input_hashes_public.json](audit/input_hashes_public.json): 원래 입력 5751개의 SHA·size·logical identity. Private absolute path는 공개하지 않음.
- [Public release audit](audit/PUBLIC_RELEASE_AUDIT.md): 공개 manifest, 검증 범위와 제한.

Public verifier: repository root에서 `python scripts/verify_development_20260913.py`. NumPy가 있으면 `--bootstrap`으로 100,000 paired draws × 6 CI까지 대조한다. [재현 단계](../../scripts/analysis/development_20260913/README.md).

Historical locked-final은 [별도 표](tables/historical_locked_final_reference.md)에만 남겼다. 신규 204 checkpoint의 locked-final은 **NOT EXECUTED / PENDING SEPARATE FREEZE**다. 새 모델 실행이나 RunPod 접속은 없었다. 모든 checkpoint/image/raw label bytes 및 private archive는 제외했다.

Logical root는 원본 소유자가 private inputs를 연결할 때 쓰는 identity이며 공개 clone의 실재 파일 경로가 아니다. 공개 CSV 검증은 private inputs를 요구하지 않는다. 원본 실행 source와 path wrapper를 구별하고, public manifest가 현재 공개 bytes를 봉인한다.
