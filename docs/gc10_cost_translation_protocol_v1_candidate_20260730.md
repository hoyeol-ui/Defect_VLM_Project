# GC10 Cost-Translation Protocol v1 Candidate

Protocol version: `gc10-cost-translation-v1-candidate`

Date: 2026-07-30

Status: **UNFROZEN — RESEARCHER AND SUPERVISOR DECISIONS REQUIRED**

## 1. Research Question

GC10에서 equal-image budget으로 관찰된 `FrozenDINOVisualDiversity`의 선택·학습 이점이 실제 human annotation active time, human-label quality, equal-time learner utility 및 rare-defect safety로 유지되는가?

연구의 주 분석 흐름은 다음과 같다.

```text
frozen q20 selection
  -> unique blinded human annotation
  -> image active time and edit events
  -> human-label omission/localization/class quality
  -> fixed human-time labeled manifest
  -> development-only learner utility
  -> rare-defect safety
```

## 2. Study Position and Claim Boundary

본 연구의 성격은 다음으로 고정한다.

> **prospective cost-endpoint validation of a retrospectively identified frozen candidate**

DINO 후보와 기존 GC10 learner outcome은 이미 알려져 있다. 앞으로 사전 동결할 수 있는 것은 human-time, human-label quality, equal-time learner, rare-safety endpoint와 이들의 PASS/FAIL/STOP 규칙이다.

허용 가능한 주장은 GC10 fixed acquisition/development setting에서 해당 후보의 비용 번역 여부를 bounded audit했다는 것이다. 새 selector, 범용 우위, 외부 제조현장 일반화, 실제 금액 절감을 주장하지 않는다.

## 3. Existing Evidence

기존 seed 0–4, q20, 3 training seed 결과:

| 지표 | DINO − Random |
|---|---:|
| development mAP50–95 | +0.017378 |
| paired bootstrap CI across 5 acquisition seeds | [+0.000580, +0.036332] |
| acquisition-seed wins | 4/5 |
| recall | +0.035381 |
| frequent macro AP | +0.033345 |
| rare macro AP | **−0.019877** |
| class 9 AP | **−0.031907** |
| class 10 AP | **−0.028495** |

따라서 overall learner gain과 rare safety failure가 동시에 존재한다. 새 연구는 이를 유리하게 해석하기 위한 rescue가 아니라, image-count budget이 실제 사람시간과 human-label을 거친 뒤 어떻게 변하는지를 측정한다.

근거 파일:

- `runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_acquisition_seed_summary.csv`
- `runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_class_group_macro_summary.csv`
- `runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_per_class_mean_comparison.csv`
- `runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_development_confirmation_gate.csv`

## 4. Prospectivity Boundary

### 이미 알려진 것

- selector 종류와 q20 selection
- 200-seed selection audit
- seed 0–4 learner outcome
- DINO의 높은 fixed-pool concentration

### 아직 결과가 없고 지금 동결 가능한 것

- Stage 0 instrumentation performance
- image active time
- edit events
- human-label quality
- adjudication time
- fixed \(T\) manifest
- seed 5–14 learner outcome
- rare/small safety outcome

Primary seeds는 learner 결과에 사용되지 않은 숫자상 첫 10개 seed인 5–14이다. 그러나 200-seed selection record에서 selection outcome은 이미 계산되었으므로 완전한 unseen selection holdout이라고 표현하지 않는다.

## 5. Stage 0 Instrumentation Smoke

### 목적

효과 비교가 아니라 logger, event export, idle 처리, annotation guide, annotation/reference linkage가 작동하는지 확인한다.

### 데이터

- Phase A union과 겹치지 않는 acquisition 이미지 24장
- 8개 stress stratum에서 각 3장
- random seed `20260730`
- stratum 안에서는 `SHA256(seed|stratum|sample_id)` 오름차순

Strata:

- rare + small
- rare + multi-box
- rare regular
- small common
- large common
- multilabel common
- single-box common
- high-box-count common

GC10 acquisition pool은 모두 한 개 이상의 reference box를 가지므로 no-box stress case는 만들 수 없다. single-box가 low-burden proxy다.

### 허용되는 결과

- log completeness
- event/state validity
- checksum
- monotonic clock
- active/wall reconciliation
- export repeatability
- idle-threshold measurement behavior

전략별 시간 또는 품질 비교는 금지한다. Stage 0 이미지는 Phase A/B에 재사용하지 않는다.

## 6. Phase A Annotation Burden and Quality

### 데이터와 작업량

- acquisition seeds 5–14
- strategies 2
- q20
- seed-batch rows 400
- unique union 233
- 2 annotators 선택 시 466 tasks
- 4 annotators 선택 시 932 tasks

각 annotator는 233 unique images를 한 번씩만 본다. 반복 selection membership은 manifest에서 재구성하며, 같은 사람에게 같은 이미지를 seed마다 반복 노출하지 않는다.

### 1차 estimand

각 seed에서 q20 batch를 완성하는 데 필요한 total active time의 DINO−Random 차이를 계산하고, annotator를 seed 안에서 먼저 요약한 뒤 10개 frozen seed의 평균 paired contrast를 보고한다.

### 품질

- box precision/recall
- false-negative count
- image-level missed-defect rate
- matched-box mean/median IoU
- class accuracy
- duplicate count
- small-defect recall
- rare pooled/class 8/9/10 recall
- adjudication correction count/time

Primary \(T\), idle cutoff, burden materiality, quality non-inferiority margin은 아직 미결정이다. Phase A arm 결과를 보기 전에 결정·서명해야 한다.

## 7. Phase B Equal-Time Learner

Phase A가 사전 규칙을 통과할 때만 실행한다.

- fixed human-time \(T\)
- seed 5–14
- strategies 2
- training seeds 42, 43, 44
- 예상 60 learner runs
- development split 평가
- human/adjudicated label이 primary
- 동일 manifest의 existing GT learner는 허용될 경우 secondary decomposition만 담당

Primary endpoint는 training seed를 acquisition seed 안에서 먼저 평균한 development mAP50–95 DINO−Random paired difference다. 60 runs를 60개의 독립 표본으로 간주하지 않는다.

Safety family:

- rare macro AP
- rare recall
- class 8, 9, 10 individual guard
- overall recall
- frequent-class non-inferiority

수치 threshold는 연구자·지도교수 서명 전까지 미결정이다. 기존 rare failure를 보고 약화하지 않는다.

## 8. Seed and Overlap Structure

### Primary chronological set

Seeds: `5, 6, 7, 8, 9, 10, 11, 12, 13, 14`

| 지표 | Random | DINO |
|---|---:|---:|
| slots | 200 | 200 |
| unique images | 187 | 51 |
| within-strategy pairwise Jaccard mean | 0.007467 | 0.390546 |

두 arm의 unique union은 233장이고 cross-strategy overlap은 5장이다. DINO seed들은 같은 prototype을 반복 선택하므로 10개 seed를 10개의 독립 제조 lot로 해석하지 않는다.

### Appendix sensitivity set

Sample ID와 same-strategy Jaccard만으로 계산된 overlap-minimized set:

`5, 198, 95, 100, 15, 75, 24, 37, 55, 87`

이 set은 appendix design sensitivity일 뿐 primary seed를 교체하지 않는다.

## 9. Human Annotation Tool Requirements

UI는 아직 구현하지 않는다. 구현 전 다음을 만족해야 한다.

- required event 13종
- UTC와 monotonic clock 동시 기록
- explicit pause/resume
- frozen idle rule
- box create/move/resize/delete
- class assign/change
- per-image submit snapshot
- append-only JSONL
- checksum
- post-submit correction time 분리
- annotator pseudonym
- strategy/seed/GT 비노출

상세 명세:

- `protocols/gc10_cost_translation_v1/annotation_event_schema.json`
- `protocols/gc10_cost_translation_v1/annotation_logger_requirements.md`
- `protocols/gc10_cost_translation_v1/annotation_export_validation_spec.md`

## 10. Quality and Adjudication

Frozen geometry rule:

1. human-reference IoU matrix
2. Hungarian one-to-one assignment
3. IoU ≥ 0.50 geometry match
4. same class면 TP
5. wrong class면 classification error와 class-aware FP/FN
6. unmatched reference는 FN
7. unmatched human은 FP
8. duplicate 별도
9. adjudication correction와 active time 별도

Rare classes는 8, 9, 10이다. Small defect는 acquisition bbox area-ratio 하위 25%를 반올림한 `≤0.004831`이다.

Adjudication 방식과 Phase B primary label provenance는 미결정이며 결과 전에 동결한다.

## 11. Statistical Analysis

### Primary: overlap-aware image-and-annotator cluster bootstrap

- 233 unique image에 cluster weight를 부여
- 같은 이미지가 여러 seed batch에 나타나는 모든 membership에 동일 weight 적용
- annotator cluster도 resampling
- 각 batch 안의 weight를 q20 scale로 정규화
- 10개 paired seed contrast로 전체 estimand 재구성

이 방법은 DINO의 반복 이미지 때문에 생기는 batch 간 의존성을 보존한다.

### Secondary: seed-level paired bootstrap

10개 seed contrast를 재표집한다. 기존 연구와 비교하기 위한 sensitivity analysis이며 image overlap을 무시하는 한계를 명시한다.

### 해석 범위

Image cluster CI도 다른 공장·lot·site를 포함하는 외부 일반화 CI가 아니다. Fixed GC10 pool과 참여 annotator에 조건화된 bounded uncertainty다.

## 12. Cost Accounting

다섯 비용을 분리한다.

1. initial annotation active time
2. review/adjudication active time
3. selector compute
4. detector training compute
5. engineering/setup person-time

DINO는 embedding reuse의 amortized scenario와 feature extraction을 포함한 cold-start scenario를 모두 보고한다.

필수 log:

- wall-clock seconds
- GPU model/minutes/peak memory
- CPU time
- person-minutes
- annotation tool/version
- embedding cache hit

금액은 문서화된 인건비와 compute 단가가 있을 때만 secondary로 계산한다.

## 13. PASS / FAIL / STOP

### Stage 0

- PASS: signed completeness·state·clock·reconciliation 기준 충족
- FAIL: logger 또는 export가 primary endpoint를 신뢰성 있게 복원하지 못함
- 다음 행동: PASS 후 threshold/config freeze; FAIL이면 UI 수정 후 새 Stage 0 version

### Phase A

- PASS: logger 유효, common \(T\) 구성 가능, signed burden materiality 충족, signed quality/rare/small safety 충족
- FAIL/STOP: burden translation 미관찰 또는 quality/safety 미달
- 다음 행동: PASS에서만 Phase B

### Phase B

- PASS: signed overall learner minimum effect와 모든 safety guard 충족
- FAIL/STOP: overall minimum 미달 또는 rare/recall/frequent guard 중 하나라도 미달
- 다음 행동: selector/backbone/loss rescue 없이 bounded result로 종료

최종 수치 규칙은 `unresolved_researcher_decisions.md`의 모든 항목이 서명되기 전에는 유효하지 않다.

## 14. Human Participation Requirements

소속기관에 다음을 확인한다.

- IRB 심의/면제/비해당 판단
- 참여 동의
- 작업시간·편집행동 기록 동의
- annotator ID 가명화
- 중도 철회
- 화면 녹화 여부
- 보상
- 보관 기간
- 공개 범위

공식 판단 전 human annotation은 차단된다.

## 15. Official/Final Test Lock

- Manifest builder는 acquisition source만 읽는다.
- 패키지 validator는 `split == acquisition`만 확인하며 final manifest를 열지 않는다.
- Phase B는 development split만 사용한다.
- 패키지에 official/final split row는 0이어야 한다.
- 별도 명시적 승인 전 final access는 허용하지 않는다.

## 16. Unresolved Researcher Decisions

총 9개:

1. annotator 2명 또는 4명
2. primary \(T\) 근거
3. idle threshold
4. Phase A burden materiality
5. Phase A quality non-inferiority
6. Phase B minimum mAP effect
7. rare AP/recall safety와 class guards
8. adjudication
9. human-participation approval/consent

상세 선택지는 `threshold_option_matrix.md`에 있으며 Codex가 대신 최종 결정하지 않는다.

## 17. Freeze Checklist

- [ ] D01–D09 결정·서명
- [ ] institution determination
- [ ] annotator information/consent approved
- [ ] annotation guide finalized
- [ ] annotation UI implemented under a new reviewed change
- [ ] Stage 0 validator PASS
- [ ] primary \(T\) and all margins entered
- [ ] blinded annotator orders generated and counterbalanced
- [ ] human-label provenance/adjudication finalized
- [ ] config/protocol/manifest/code/initial weights hashed
- [ ] package validator PASS
- [ ] Git commit/tag created before Phase A
- [ ] official/final lock rechecked

현재는 candidate이므로 commit할 수는 있지만, **동결된 실행 protocol로 tag하거나 annotation을 시작하면 안 된다.**

## 18. File and Hash Ledger

Manifests:

- `protocols/gc10_cost_translation_v1/stage0_instrumentation_manifest.csv`
- `protocols/gc10_cost_translation_v1/phase_a_unique_image_manifest.csv`
- `protocols/gc10_cost_translation_v1/phase_a_seed_batch_manifest.csv`
- `protocols/gc10_cost_translation_v1/phase_b_frozen_candidate_manifest_template.csv`
- `protocols/gc10_cost_translation_v1/acquisition_overlap_matrix.csv`
- `protocols/gc10_cost_translation_v1/image_seed_membership.csv`
- `protocols/gc10_cost_translation_v1/overlap_minimized_seed_sensitivity.csv`
- `protocols/gc10_cost_translation_v1/stage0_annotator_blinded_manifest.csv`
- `protocols/gc10_cost_translation_v1/phase_a_annotator_blinded_manifest.csv`

Ledgers:

- `protocols/gc10_cost_translation_v1/manifest_hashes.sha256`
- `protocols/gc10_cost_translation_v1/source_evidence_hashes.sha256`
- `protocols/gc10_cost_translation_v1/manifest_build_metadata.json`

Builder/validator:

- `scripts/08_gc10_cost_translation/build_protocol_manifests.py`
- `scripts/08_gc10_cost_translation/validate_protocol_package.py`

Source evidence commit: `ee331ffd9e7f93951d570ef2020735f1f66d904a`

The hashes in this candidate prove current file identity only. They become temporal evidence only after the reviewed package is committed and tagged before Phase A.
