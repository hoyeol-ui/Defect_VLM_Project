# VisA GT-Free Annotation Triage Pilot Protocol

작성일: 2026-07-15  
상태: Phase 0 ingestion/audit 사전 고정  
Detector training: 금지  
Final test evaluation: 금지

## 1. 연구 질문

기존 NEU/GC10 결과는 balanced 또는 비교적 작은 acquisition pool에서 Random이 class coverage, instance richness, visual diversity 면에서 강하다는 것을 보였다. VisA pilot의 질문은 selector를 다시 튜닝하는 것이 아니다.

> 결함 영상이 약 11%인 자연스러운 anomaly-sparse production pool에서, frozen GT-free selection signal이 Random보다 annotation budget당 더 많은 실제 anomaly와 defect instance를 발견하는가?

VisA는 공식 배포 기준 12 object, 10,821 images, 9,621 normal, 1,200 anomalous images와 pixel mask를 제공한다.

## 2. Phase 0 범위

이번 단계에서 허용되는 작업:

- 공식 VisA tar 다운로드 및 SHA-256 기록
- 공식 `image_anno.csv` 12개 통합
- image/mask 존재성, 크기, label 일관성 감사
- anomaly mask connected component를 post-hoc bbox로 변환
- exact/perceptual duplicate group 감사
- deterministic acquisition/development/final manifest 생성
- split 간 path/SHA-256 exact duplicate leakage 감사
- final manifest hash 기록 후 잠금

금지 작업:

- YOLO 또는 다른 detector 학습
- final split inference/evaluation
- GT/mask/anomaly label을 selector feature로 사용
- VisA 결과를 보고 prompt, DINO distance, selector weight를 조정
- anomaly가 많은 seed/조건만 선택해 보고

## 3. Split protocol

- Split seed: `20260715`
- Unit: exact SHA-256 duplicate group; byte-identical image는 서로 다른 split에 들어갈 수 없다.
- pHash는 aligned industrial image에서 별개 생산 sample도 같은 값 또는 가까운 값을 가질 수 있으므로 audit-only로 사용하고 hard split grouping에는 사용하지 않는다.
- Stratification: `object_category × image_label(normal/anomaly)`
- Target ratio: acquisition 80%, development 10%, final 10%
- 원본 비율을 유지하며 모든 object/category가 각 split에 포함되도록 한다.
- final manifest는 생성 직후 SHA-256을 기록하고 이후 selection/analysis loader에서 경로 접근을 거부한다.
- development와 final은 detector 학습 dataset YAML에 동시에 등장할 수 없다.

공식 split은 원 논문의 anomaly-detection benchmark 재현에는 사용될 수 있으나, 본 annotation-triage 연구에서는 위의 독립된 group-stratified split을 primary protocol로 사용한다. 공식 split과의 차이는 문서에 명시한다.

## 4. Annotation representation

- Primary image label: `normal` 또는 `anomaly`
- Detector class: 첫 pilot에서는 binary `defect`
- anomalous mask의 각 connected component를 하나의 bbox로 변환
- 너무 작은 component를 사후 제거하지 않는다. 최소 크기 기준을 결과 확인 후 정하면 tuning이 되기 때문이다.
- multi-class mask id와 object category는 audit metadata로 보존한다.
- normal image는 bbox 0개의 valid negative annotation으로 처리한다.

## 5. Selection-only feasibility audit

Phase 0 데이터 감사가 통과한 뒤 별도 실행한다.

- Acquisition seeds: 200개 이상
- Shared initial labeled size: 20
- Query size: 20
- 평가 budget: 40까지만
- Strategies:
  - `GTFreeRandom`
  - frozen DINO farthest-first visual diversity
- DINO model/config는 selection 결과 확인 전에 고정
- selector 입력 허용: image pixels, object-unaware DINO embedding, shared initial-set identity
- selector 입력 금지: image label, mask, bbox, defect class, object category inferred from annotation CSV

Post-hoc 지표:

- query anomaly-image yield
- cumulative anomaly discovery rate
- annotations to first anomaly
- unique defect-mask id coverage
- connected-component bbox count/area coverage
- object category coverage
- query internal DINO redundancy
- distance to initial set

## 6. Selection-only gate

Detector 학습은 아래를 모두 만족할 때만 허용한다.

1. Random query의 평균 anomaly yield가 `3/20` 이하로 annotation scarcity가 실제로 확인됨
2. frozen Visual이 Random보다 평균 anomaly images를 최소 `+2/20` 더 발견
3. 200 paired seeds bootstrap CI가 0을 포함하지 않음
4. Visual 승률이 최소 65%
5. Visual의 object category coverage가 Random보다 평균적으로 악화되지 않음
6. 효과가 특정 object 한 개에만 의존하지 않음

Gate 실패 시 DINO selector는 VisA detector 학습으로 확장하지 않는다. Prompt/weight/score를 추가하여 gate를 사후 통과시키지 않는다.

## 7. Gate 통과 후에만 가능한 학습

- 동일 detector, 동일 training seed, 동일 labeled set size로 paired Random/Visual 비교
- development split만 사용
- final split은 전체 method와 hyperparameter가 잠긴 뒤에도 별도 승인 전까지 사용 금지
- primary endpoint는 development defect AP50-95와 anomaly-image recall
- selection-only 이득과 detector 성능 이득을 별개의 조건으로 보고

## 8. 예상 산출물

- `visa_source_audit.csv`
- `visa_mask_bbox_audit.csv`
- `visa_acquisition_pool.csv`
- `visa_development_eval.csv`
- `visa_final_test_locked.csv`
- `visa_split_leakage_audit.csv`
- `visa_protocol_config.json`
- `visa_ingestion_audit_summary.md`
- 이후 별도 `visa_selection_only_audit_*` 결과

## 9. 중단 규칙

- source annotation/image/mask 불일치가 해결되지 않으면 split 생성 중단
- split leakage가 하나라도 있으면 selection audit 중단
- selection-only gate 실패 시 detector 학습 중단
- 이 pilot의 실패를 이유로 NEU final test 또는 VisA final test를 열지 않음
