# GC10 Cost-Translation Novelty and Feasibility Audit

감사일: 2026-07-30

감사 범위: 현재 저장소의 코드·Git 이력·GC10 원시 결과, 지정된 선행연구 5편, CVAT 공식 문서
제안 연구: *From Image Budgets to Human Time: A Prospective Cost-Translation Audit of Active Learning for Industrial Defect Detection*

> 이 문서는 제안 연구를 옹호하기 위한 기획서가 아니다. 이미 확인된 DINO 결과를 사후적으로 살리는 규칙을 만들지 않고, 현재 증거로 실행 가능한 연구인지와 주장 가능한 범위를 독립적으로 판정한다.

## 1. Executive Verdict

### 최종 결정

**`GO_AFTER_PROTOCOL_REVISION`**

연구 질문은 타당하고 현재 학위 연구의 다음 단계로서 정보가치가 높다. 특히 “이미지 수가 같은 선택 우위가 실제 사람의 작업시간·주석 품질·희귀 결함 안전성·동일 시간 학습효용으로도 이어지는가?”는 현재 Translation 연구에서 빠져 있던 비용 단위를 직접 연결한다.

그러나 현재의 5 acquisition seed·137-image union·2 annotator 구상 그대로는 강한 확증 연구로 부르기 어렵다. 다음 다섯 가지를 **인간 주석 시작 전에** 다시 동결해야 한다.

1. 기존 learner 결과에 쓰이지 않은 acquisition seed의 결정론적 선택
2. instrumentation smoke와 Phase A 본 실험의 완전 분리
3. 이미지별 active time과 수정 이벤트를 복원할 수 있는 실제 계측기
4. 누락 box·희귀/소형 결함을 포함한 human-label 품질 및 adjudication 규칙
5. equal-time \(T\), 최소효과, 통계 단위, cost field 및 STOP 규칙

### 현재 연구의 정확한 성격

이 연구는 다음 표현으로 한정해야 한다.

> **Prospective cost-endpoint validation of a retrospectively identified frozen candidate**

즉, DINO 후보 자체의 prospective validation이 아니다. DINO 후보와 기존 GC10 learner 결과는 이미 2026-07-18에 존재했고, cost study 질문은 2026-07-30에 생겼다. 앞으로 prospective하게 만들 수 있는 것은 annotation-time, human-label quality, equal-time manifest, learner endpoint, rare-class safety, 최소효과 및 STOP 규칙이다.

### 냉정한 가치 판정

| 항목 | 판정 | 이유 |
|---|---|---|
| 알고리즘 novelty | 낮음 | 새 selector나 학습 알고리즘을 제안하지 않음 |
| 평가방법 novelty | 중간 이상 | equal-image → human time → label quality → equal-time learner를 한 frozen chain으로 연결 |
| 산업 관련성 | 잠재적으로 높음 | 실제 인력시간과 누락 결함을 비용·안전 변수로 취급 |
| 현재 증거의 확증력 | 낮음~중간 | fixed public pool, 기존 5 acquisition seed, 인간 주석 미실시 |
| 석사학위 기여 | 방어 가능 | bounded prospective empirical audit로 제한할 경우 |
| main-track 가능성 | 현재 낮음 | 다수 annotator와 외부 lot/site가 없으면 일반화가 부족 |

## 2. Literature Overlap Matrix

### 지정 선행연구 직접 확인 결과

| 연구 | Task/domain | Cost unit | Human study | Annotation mode / selection | Learner test | Safety metric | 본 연구와 겹치는 부분 | 남는 차이 |
|---|---|---|---|---|---|---|---|---|
| Lüth et al., *Navigating the Pitfalls of Active Learning Evaluation*, NeurIPS 2023 | Image classification, 다수 benchmark | 주로 labeled sample budget와 compute 고려 | 없음 | 다양한 AL query method의 체계적 benchmark | 있음 | 별도 희귀결함 safety 아님 | 현실적 AL 평가, seed·budget·학습설정 통제 | 산업 object detection의 실제 human time·human label·rare safety 연결 |
| Liao et al., *Box-Level Class-Balanced Sampling for Active Object Detection* | Object detection | box-level annotation budget | 확인되지 않음 | box-level sampling, pseudo-label, class balance | 있음 | minority-class balance | 이미지 수가 실제 box burden을 반영하지 않는 문제 | 사람의 active time과 human-label 오류를 직접 계측하지 않음 |
| Burges et al., *Active Learning Meets Foundation Models*, ICCV 2025 | Remote-sensing object detection | 고정 사용자 시간·완료/승인 box | 있음: 공식 논문은 8명의 human-geography 전문가, 4개 세션, 10분 안내+50분 annotation을 보고 | SAM proposal, uncertainty/diversity, dynamic box switching, interactive UI | 있음 | 본 연구식 rare-defect gate는 아님 | fixed-time human study, foundation-assisted annotation, downstream 성능 | 산업 결함·manual bbox·equal-image 대 equal-time translation·희귀 안전성이라는 조합은 다름 |
| Zhang et al., *Foundation-Assisted Active Learning for Object Detection Annotation*, arXiv v1 2026-07-18 | Remote-sensing object detection | annotation budget와 manual burden proxy | arXiv 초록에서 human-time study 확인 안 됨 | UPN+SAM2, detector source, DINOv2/SAM2 object-centric diversity, box switching | 있음 | 별도 rare safety 확인 안 됨 | foundation model, DINO 기반 diversity, annotation burden | 최신 알고리즘 경쟁은 강함. 본 연구는 새 hybrid selector가 아니라 비용 번역 검증으로 차별화해야 함 |
| Baeuerle et al., *Are Foundation Models Ready for Industrial Defect Recognition?*, arXiv 2025 | Industrial defect recognition | setup/label effort의 문제의식 | 없음 | foundation-model reality check | 있음 | 별도 AL safety gate 아님 | 공개 benchmark와 실제 산업 데이터 간 괴리 | GC10만으로 production 일반화 불가함을 오히려 확인 |

출처:

- https://arxiv.org/abs/2301.10625
- https://arxiv.org/abs/2508.17849
- https://openaccess.thecvf.com/content/ICCV2025/html/Burges_Active_Learning_Meets_Foundation_Models_Fast_Remote_Sensing_Data_Annotation_ICCV_2025_paper.html
- https://github.com/mburges-cvl/ICCV_AL4FM
- https://arxiv.org/abs/2607.16671
- https://arxiv.org/abs/2509.20479

### 출판 상태 주의

- Lüth et al.은 arXiv 레코드에 NeurIPS 2023 accepted로 명시되어 있다.
- Burges et al.은 ICCV 2025 CVF proceedings 논문이다.
- `2508.17849`의 arXiv 페이지에는 “Accepted to ICIP2024”라고 적혀 있으나, arXiv 최초 제출일은 2025-08-25이다. 이 서지상 불일치는 논문 작성 시 proceedings 서지를 추가 확인해야 한다.
- `2607.16671`과 `2509.20479`는 감사 시점 arXiv 페이지에 별도 학회 게재 정보가 없어 **preprint**로 취급한다.

### novelty 결론

“최초의 cost-aware AL”, “최초의 human-time AL”, “최초의 box-level AL”, “최초의 foundation-assisted annotation”은 모두 방어할 수 없다. 본 연구가 차별화할 수 있는 지점은 아래 조합이다.

> 산업 결함 object detection에서, retrospectively identified candidate의 equal-image selection advantage를 실제 manual annotation active time·human-label quality·equal-time learner utility·rare-defect safety까지 하나의 사전 동결된 translation chain으로 감사한다.

이는 **새 알고리즘 기여가 아니라, 부분적으로 새로운 empirical evaluation contribution**이다.

## 3. Current GC10 Evidence Recalculation

### 사용한 원자료

- `runs/gc10_taxonomy_selection_audit/gc10_random_vs_dino_200seed_20260715/gc10_selection_records_posthoc.csv`
- `runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_run_metrics.csv`
- `runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_per_class_metrics.csv`
- `runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_acquisition_seed_summary.csv`
- `runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_class_group_macro_summary.csv`
- `runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_per_class_mean_comparison.csv`
- `runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/detector_development_confirmation_gate.csv`

### 기존 learner 사용 acquisition seed 0–4

| 지표 | Random | FrozenDINOVisualDiversity | 비고 |
|---|---:|---:|---|
| query slots | 100 | 100 | 5 seeds × q20 |
| unique images | 96 | 44 | DINO는 중복 선택이 매우 큼 |
| box instances in slots | 141 | 170 | unique-image 기준이 아니라 slot 합 |
| rare-class-containing slots | 12 | 22 | class 8–10 포함 |
| multilabel slots | 17 | 26 | DINO가 복합 장면을 더 선택 |
| pairwise Jaccard mean | 0.010391 | 0.338460 | seed 간 set overlap |
| pairwise Jaccard min | 0 | 0.25 |  |
| pairwise Jaccard max | 0.052632 | 0.481481 |  |
| max image selection frequency | 2/5 | 5/5 | DINO의 고정 prototype 반복 |
| top-10 image slot share | 0.14 | 0.42 | 설명용 concentration 지표 |
| slot-frequency ESS | 92.59 | 33.33 | **기술통계일 뿐 inferential n 아님** |

두 arm의 unique union은 137장이고, cross-arm unique overlap은 3장이다. 따라서 기존 137장 모두를 두 annotator가 주석하면 274 image-annotator tasks가 된다. 그러나 이 137장은 이미 detector outcome을 확인한 seed 0–4에서 나온 집합이므로 새 cost study의 확증 표본으로 쓰는 것은 권하지 않는다.

### 기존 detector 결과 재확인

| 지표 | DINO − Random |
|---|---:|
| mean mAP50–95 delta | +0.017378 |
| acquisition-seed paired bootstrap 95% CI | [+0.000580, +0.036332] |
| acquisition seed wins | 4/5 |
| mean recall delta | +0.035381 |
| recall wins | 5/5 |
| frequent-class macro AP delta | +0.033345 |
| rare-class macro AP delta | **−0.019877** |
| class 8 AP delta | +0.000772 |
| class 9 AP delta | **−0.031907** |
| class 10 AP delta | **−0.028495** |

기존 frozen gate는 overall mAP, win count, bootstrap CI, recall non-inferiority, frequent-class non-inferiority는 통과했지만 rare macro AP `≥ +0.020`과 class 9 positive 조건을 통과하지 못했다. 따라서 현재 사실은 다음 두 문장을 동시에 유지해야 한다.

1. DINO는 이 개발 split의 5 acquisition seed에서 overall mAP와 recall을 개선했다.
2. 그 개선은 frozen rare-class safety 조건을 충족하지 못했다.

### bootstrap 단위 확인

`scripts/02_active_learning/run_gc10_development_detector_confirmation.py`는 세 training seed를 acquisition seed 안에서 먼저 평균하고, 5개의 acquisition-seed paired delta를 bootstrap한다. 30개 learner run을 30개의 독립 표본으로 취급하지 않는다. 이 부분은 현재 코드 설계가 맞다.

## 4. Independence and Seed Audit

### 200-seed 기록의 의미

selection CSV에는 acquisition seed 0–199의 selection outcome이 이미 모두 계산되어 있다. 따라서 seed 5–199는 **learner-unused**이지만 **selection-outcome-naïve holdout**은 아니다. “완전히 보지 않은 acquisition seed”라고 표현하면 안 된다.

그럼에도 human time·human label·equal-time learner outcome은 아직 존재하지 않으므로, 다음 결정론적 규칙으로 seed를 고르면 새 비용 endpoint의 prospective 검증에는 사용할 수 있다.

> 기존 learner에 쓰인 0–4를 제외하고, 숫자가 가장 작은 연속 10개 seed인 **5–14**를 사용한다. selection metric이나 class 결과를 보고 seed를 고르지 않는다.

### 후보 seed 수별 재계산

| seed count | seed IDs | prior learner use | Random unique | DINO unique | union | cross-arm overlap | Random Jaccard mean | DINO Jaccard mean | 2 annotators-all tasks |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 5 | 5–9 | 없음 | 98 | 43 | 138 | 3 | 0.005128 | 0.367037 | 276 |
| **10** | **5–14** | **없음** | **187** | **51** | **233** | **5** | **0.007467** | **0.390546** | **466** |
| 20 | 5–24 | 없음 | 364 | 69 | 418 | 15 | 0.005142 | 0.366243 | 836 |

추가 기술통계:

- seed 5–14에서 Random은 200 slots/187 unique, DINO는 200 slots/51 unique이다.
- seed 5–14의 slot-frequency ESS는 Random 176.99, DINO 33.33이다.
- DINO의 높은 overlap은 “DINO seed가 33개 독립표본과 같다”는 뜻이 아니다. overlap만으로 effective independent batch count를 식별할 수 없다.
- 모든 acquisition seed는 같은 fixed pool의 perturbation이므로 seed 간 완전 독립 생산 lot으로 해석할 수 없다.

### 권고

- seed 5개: feasibility screen은 가능하지만 확증 해석은 약함.
- **seed 10개: 현재 학위 범위에서 권하는 최소 bounded confirmatory-style design.**
- seed 20개: 통계적으로 더 낫지만 인간 주석량이 크게 늘고 fixed-pool 일반화 한계는 그대로 남음.

## 5. Prospectivity Audit

### 시간 순서

| 사건 | 확인 결과 |
|---|---|
| DINO candidate와 GC10 protocol/code/result가 Git에 들어온 시점 | commit `ee331ffd9e7f93951d570ef2020735f1f66d904a`, 2026-07-18 23:13:19 +09:00 |
| 기존 GC10 learner outcome 확인 | 위 commit에 protocol·code·result가 함께 포함되어 outcome-before 독립 Git freeze는 아님 |
| cost-translation study idea | 2026-07-30 대화·문서 |
| annotation endpoint freeze | 아직 없음 |
| equal-time \(T\) freeze | 아직 없음 |
| minimum effect/STOP freeze | 아직 없음 |

### 정확한 라벨

| 후보 표현 | 판정 |
|---|---|
| A. prospective selector validation | **불가** |
| B. prospective cost endpoint validation of a retrospectively identified candidate | **권고** |
| C. retrospective human-cost analysis | 주석 전에 protocol을 동결하지 않으면 이 상태로 하락 |
| D. exploratory feasibility study | Stage 0만 수행하거나 annotator 1명일 때 적절 |

### freeze에 포함해야 하는 파일

인간 주석 전에 별도 protocol commit/tag에 최소 다음을 포함해야 한다.

- acquisition seed 5–14와 query rank manifest hash
- Stage 0/Phase A image manifest 분리와 hash
- annotator blinding·ordering·break·idle rule
- event schema와 active-time 계산식
- human-label quality matching 규칙
- primary \(T\) 산출 규칙
- Phase A authorization/STOP
- Phase B learner·metric·rare safety·STOP
- official/final test lock
- 코드·config·초기 weight·input manifest hash

## 6. CVAT Instrumentation Feasibility

### 공식 문서에서 확인된 사실

CVAT paid Analytics는 job/project/task 단위 working time, start/end UTC, created/updated/deleted objects, event export를 제공하지만 CVAT Online paid tier와 Enterprise 기능이다.

- https://docs.cvat.ai/docs/qa-analytics/analytics/

Community analytics 문서는 `draw:object`, `drag:object`, `resize:object`, `delete:object`, `change:frame`, `save:job`, `click:element` 등의 client event와 working-time 계산 규칙을 설명한다. working time은 이전 event와의 간격이 100초 이하일 때 그 차이를 더하고, `change:frame` duration을 합산한다. 생성된 working-time event는 사용된 client event에 따라 job/task/project에 귀속되지 않을 수도 있다.

- https://docs.cvat.ai/docs/administration/community/advanced/analytics/

Events API는 event 생성, event export, CSV event log 조회와 job/task/user/time filter를 제공한다.

- https://docs.cvat.ai/docs/api_sdk/sdk/reference/apis/events-api/

### 필요한 필드별 판정

| 필드 | CVAT paid Analytics | Community Events/API | 별도 logger 필요 가능성 |
|---|---|---|---|
| annotator_id | 지원 | 지원 | 낮음 |
| sample/frame_id | job/frame 설계에 따라 간접 | event payload와 `change:frame`으로 복원 가능성 | **smoke 확인 필요** |
| task order | start/event 순서로 복원 | timestamp로 복원 | 낮음 |
| start/end UTC | 지원 | event timestamp | 낮음 |
| per-image active ms | multi-image job에서 직접 보장 안 됨 | 재구성 필요 | **높음** |
| box create/update/delete | aggregate 지원 | object events 지원 | 낮음~중간 |
| class change | 명시 전용 column 확인 안 됨 | payload/click·update 해석 필요 | 중간~높음 |
| frame navigation | aggregate 화면에는 없음 | `change:frame` 있음 | 중간 |
| idle intervals | 100초 rule 기반 working time | event timestamp로 재구성 | smoke 필요 |
| save/submit | job event | `save:job` 등 | 낮음 |
| final annotation export | 지원 | 지원 | 낮음 |

### 현재 로컬 가용성

- Docker: 설치되지 않음
- `cvat_sdk`: 설치되지 않음
- Label Studio CLI/module: 설치되지 않음
- 유료 CVAT Online/Enterprise 계정: 저장소에서 확인 불가

따라서 **CVAT 사용 가능성은 `CONDITIONAL / NOT CURRENTLY INSTALLED`**이다. 특히 multi-image job의 정확한 per-image active time을 CVAT가 자동으로 제공한다고 전제하면 안 된다.

### 현재 환경에서 권하는 계측 방식

**권고: 최소 기능의 로컬 bbox annotation UI를 별도로 freeze하고, 이미지별 explicit start/pause/resume/submit과 모든 편집 event를 JSONL로 기록한다.**

필수 event schema:

```text
annotator_id, session_id, sample_id, acquisition_manifest_ids,
strategy_blind_task_id, order_index,
event_type, event_utc, active_ms_since_previous,
box_id, class_before, class_after, box_before, box_after,
pause_reason, idle_flag, tool_version
```

이 선택은 CVAT보다 기능이 뛰어나서가 아니라, 현재 Windows 환경에서 Docker/유료 계정 의존 없이 **primary endpoint인 per-image active time을 직접 통제**할 수 있기 때문이다. 단, 이 UI도 Phase A 전에 별도 smoke·export validator·hash freeze가 필요하다. 현재 감사에서는 구현하지 않는다.

대안 우선순위:

1. 유료 CVAT 접근이 이미 있다면 one-image-per-job CVAT를 smoke로 검증
2. 최소 로컬 bbox UI
3. multi-image CVAT + Events API 재구성
4. 화면 녹화 기반 수동 복원은 최후 수단이며 primary 계측으로 비권고

## 7. Human-Study Risk Audit

### 가장 큰 위험

| 위험 | 왜 문제인가 | 사전 통제 |
|---|---|---|
| learning effect | 뒤에 본 이미지가 빨라짐 | balanced block order, practice images 분리 |
| fatigue | arm/order와 시간이 혼입 | 세션 길이·휴식 고정 |
| image-order imbalance | 어려운 이미지가 특정 후반부에 몰림 | annotator별 counterbalanced order |
| repeated-image memory | DINO high overlap 때문에 동일 이미지를 반복하면 시간 왜곡 | 233 unique union을 annotator별 한 번만 주석 |
| annotator expertise | 일반 학생과 검사 전문가 결과가 다름 | 경력·경험 기록, 주장 범위 제한 |
| class-guide ambiguity | class change와 FN 증가 | 예시 포함 guide, smoke 질의 후 freeze |
| small-defect omission | mean IoU로 포착 불가 | small-defect recall, FN 명시 |
| idle contamination | 사람이 자리를 비운 시간이 포함 | explicit pause + idle flag + smoke 검증 |
| inter-annotator variation | 두 명 평균이 일반화되지 않음 | annotator별 결과와 interaction 보고 |
| adjudication leakage | GT를 보고 label을 고치면 비용 누락 | blinded adjudication 및 시간 별도 계측 |
| ethics/consent | 인간 참여 데이터 | 기관 IRB/면제/동의 필요 여부를 사전 공식 확인 |

### 연구 성격과 annotator 수

| annotator 수 | 가능한 표현 |
|---:|---|
| 1 | single-annotator case study |
| **2** | **석사학위 최소: bounded paired annotation study** |
| 3–5 | 제한된 human-subject study |
| **4** | 이상적인 현실적 권고: annotator effect를 더 정직하게 제시 가능 |
| 다수 산업 검사 전문가 | industrial user study 주장 가능성 증가 |

2명으로도 학위 보완 실험은 가능하지만 “산업 사용자의 평균 시간을 줄였다”는 일반화는 불가하다. 각 annotator 결과를 숨기지 않고 분리 보고해야 한다.

### 권고 배정

- 권고 confirmatory-style bounded design: annotator 2명이 seed 5–14의 233 unique images를 모두 각각 한 번씩 주석
- Phase A task: 233 × 2 = **466**
- Stage 0: Phase A union과 겹치지 않는 24 images × 2 = **48**
- 총 인간 annotation task: **514**, adjudication 별도
- 부담 축소안: 233장을 두 annotator에게 나누고 25%만 중복하면 약 292 tasks이나, annotator-image confounding 때문에 exploratory로 격하

## 8. Revised Stage 0 / Phase A / Phase B

### Stage 0 — Instrumentation smoke

| 항목 | 설계 |
|---|---|
| 질문 | 이미지별 active time·편집 event·최종 annotation을 손실 없이 복원할 수 있는가? |
| 데이터 | Phase A 233장과 겹치지 않는 24장 |
| annotator | Phase A와 동일한 2명 |
| inference unit | 없음; 계측 검증 단계 |
| primary endpoint | valid event log/export 비율 |
| safety endpoint | annotation/export mismatch, clock/order mismatch, idle contamination |
| PASS 예시 | task의 ≥95%에서 완전 log+annotation export, unresolved mismatch 0, pause/idle test 성공 |
| 다음 행동 | PASS 후 logger·guide·idle rule·\(T\) 산출 규칙·minimum effect를 commit |
| 금지 | arm 비교, hypothesis test, Phase A 결과에 smoke image 재사용 |

Stage 0는 연구효과 pilot이 아니다. 계측기가 측정 가능한지 확인하는 공학적 smoke다.

### Phase A — Frozen annotation burden and quality study

| 항목 | 설계 |
|---|---|
| 질문 | q20 equal-image batch에서 DINO와 Random의 실제 active time과 human-label quality가 다른가? |
| 데이터 | acquisition seed 5–14, 두 arm unique union 233장 |
| annotation | strategy/seed/GT blind, annotator별 각 image 1회 |
| primary inference unit | acquisition seed 10개의 paired arm contrast |
| image-level unit | unique image, annotator와 교차된 repeated observation |
| primary endpoint | seed별 q20 total active time의 DINO−Random 차이; 두 annotator를 seed 안에서 먼저 평균 |
| quality endpoints | box precision/recall, FN, matched IoU, class accuracy, duplicates, small/rare recall, adjudication corrections |
| PASS | logger PASS + common-support \(T\) 구성 가능 + smoke 후 동결한 burden/quality materiality 조건 충족 + rare/small safety 허용범위 충족 |
| FAIL/STOP | 신뢰할 수 없는 계측, material burden 차이 없음, 또는 rare/small human-label safety 악화 |
| 다음 행동 | PASS일 때만 Phase B manifest 생성 |

Phase A의 최소효과 수치는 현재 데이터로 정할 수 없다. 실제 annotation 시간이 전혀 없기 때문이다. 다음 순서를 지켜야 한다.

1. Stage 0에서 계측오차와 작업시간 분포만 확인
2. 외부 운영 기준이 있으면 그것을 우선 사용
3. 없으면 Stage 0 종료 직후, Phase A arm을 보기 전에 measurement-resolvable/operationally meaningful margin을 동결
4. Phase A 결과 확인 후 margin 변경 금지

### Human-label quality matching

사전 규칙:

1. 예측 human boxes와 reference boxes 간 IoU 행렬 계산
2. Hungarian one-to-one assignment로 total IoU 최대화
3. geometry match는 IoU ≥ 0.50
4. class-aware TP는 geometry match이면서 class 동일
5. geometry는 맞고 class가 다르면 classification error이며 class-aware FP/FN에 반영
6. unmatched reference는 FN, unmatched human box는 FP
7. 동일 결함을 중복 box로 그린 경우 duplicate count 별도 보고

보고 지표:

- box precision/recall
- FN count 및 image-level missed-defect rate
- matched IoU
- class accuracy on geometry matches
- duplicate-box count
- small-defect recall
- rare-class recall
- adjudication correction count와 adjudication active time

small defect는 기존 acquisition bbox GT audit에서 **bbox area ratio 하위 25%**, 즉 `area_ratio ≤ 0.004831`로 Phase A 전에 고정할 수 있다. rare class는 기존 정의인 class 8, 9, 10을 유지한다.

### Phase B — Equal-time learner study

| 항목 | 설계 |
|---|---|
| 질문 | 동일한 실제 human active-time \(T\)에서 DINO가 Random보다 learner utility를 높이면서 rare safety를 지키는가? |
| authorization | Phase A PASS인 경우에만 |
| 데이터 | development split만 사용, official/final test 계속 locked |
| labels | primary는 frozen human label/adjudicated label; review 시간도 비용 포함 |
| strategies | FrozenDINOVisualDiversity 대 GTFreeRandom |
| acquisition seeds | 5–14, 총 10 |
| training seeds | 기존과 동일한 3개 |
| learner runs | 2 × 10 × 3 = **60** |
| primary endpoint | fixed \(T\)에서 seed별 training-seed 평균 mAP50–95의 DINO−Random 차이 |
| safety | rare macro AP/recall 및 사전 지정 rare-class 조건 |
| inference unit | acquisition seed 10개; 60 run을 n=60으로 취급 금지 |
| secondary diagnostic | 같은 equal-time manifest에 benchmark GT를 쓴 oracle-label learner; primary 대체 금지 |
| STOP | overall 이득 미달, CI/win rule 미달, 또는 rare safety 미달 시 확장 중단 |

기존 결과를 보고 safety threshold를 완화하면 안 된다. 최소한 기존 detector gate의 다음 조건은 유지해야 한다.

- mean mAP50–95 delta ≥ +0.010
- acquisition-seed win/CI rule
- rare macro AP delta ≥ +0.020
- 기존 class 8·9 positive guard
- recall/frequent non-inferiority

새 protocol에서는 class 10까지 individual rare guard에 추가하는 것을 권한다. 이는 기존 결과를 구제하는 변경이 아니라, 기존 gate에서 빠진 rare class를 향후 안전성에 포함하는 **더 엄격한 사전 규칙**이다. 과거 5-seed 판정을 소급 변경하지 않는다.

### equal-time \(T\)의 권고 규칙

저장소에는 실제 운영 session 길이가 없다. 따라서 지금 임의로 30분·45분·60분을 고를 근거가 없다.

우선순위:

1. Phase A 전에 문서화된 실제 현장 session 제한이 제공되면 그것을 사용
2. 없으면 Stage 0의 동일한 사전지정 20-image sequence를 기준으로, 각 annotator의 총 active time 중 더 짧은 값을 5분 단위로 내림하여 공통 \(T\)로 고정

두 번째 규칙은 arm 결과를 보지 않는 instrumentation calibration이다. Stage 0 이미지와 Phase A 이미지는 겹치지 않아야 한다. primary \(T\)는 하나만 두고, 여러 checkpoint의 time-AULC는 사전 지정된 secondary로만 허용한다.

## 9. Statistical Analysis Unit

### 단위 구분

| 단위 | 역할 | 독립 표본으로 간주 가능한가 |
|---|---|---|
| image | annotation burden/quality 관찰 | annotator와 seed에 중첩·교차됨 |
| annotator | 사람 차이 | 2명이면 population inference 약함 |
| acquisition seed | arm 간 primary paired contrast | Phase A/B의 주 통계 단위 |
| training seed | learner stochasticity 안정화 | acquisition seed 안에서 먼저 평균; 독립 n 아님 |
| selected batch | 실제 AL decision unit | seed마다 q20 |
| fixed pool | 연구가 조건화된 모집단 | 다른 lot/site로 일반화 불가 |

### 권고 분석

Phase A:

- 각 image의 annotator별 active time과 quality를 먼저 보존
- 각 acquisition seed/strategy의 rank 1–20 image time을 합산
- annotator별 seed total을 만든 뒤 annotator 평균
- 10개의 seed-level paired DINO−Random contrast를 보고
- paired bootstrap CI, median, sign/win count를 함께 보고
- annotator별 결과와 annotator×strategy 차이를 분리 보고

Phase B:

- 각 acquisition seed/strategy/training seed의 learner metric 계산
- 같은 acquisition seed 안에서 3 training seed 평균
- 10개의 acquisition-seed paired difference에 대해 CI와 win count 계산

### 중요한 한계

DINO의 seed 5–14 pairwise Jaccard mean은 0.3905로 높다. acquisition seed 10개는 동일 pool과 반복 선택 이미지에 조건화되어 있으므로 10개의 독립 공장 lot가 아니다. bootstrap CI는 fixed-pool seed perturbation에 대한 bounded inference이며 제조 현장 일반화 CI가 아니다.

“effective independent seed 수”를 Jaccard나 slot-frequency ESS로 대체하지 말아야 한다. 이 지표들은 concentration 설명용이다.

## 10. Cost Model

### 비용을 분리해 보고할 것

1. human annotation active time
2. review/adjudication active time
3. selection compute
4. detector training compute
5. engineering/setup effort

### DINO의 두 시나리오

| 시나리오 | 포함 |
|---|---|
| amortized | 이미 생성된 embedding을 여러 AL round에서 재사용; ranking + human + learner cost |
| cold-start | embedding extraction부터 ranking, human, learner까지 모두 포함 |

현재 확인 가능한 기존 compute:

- `outputs/gc10_visual_embeddings/dinov2_small_protocol_20260715/embedding_config.json`: 1,836 images embedding runtime `49.6444 s`, CUDA/AMP/batch 32
- 기존 detector 30-run: mean `80.5565 s/run`, sum `2416.694 s`

하지만 GPU 모델, peak memory, ranking runtime, engineering person-time가 빠져 있다. 따라서 금액 절감이나 완전한 GPU-hour 비용을 현재 주장할 수 없다.

향후 필수 log:

```text
wall_clock_seconds
gpu_model
gpu_minutes
peak_gpu_memory_mb
cpu_model_or_threads
cpu_seconds
person_minutes
annotation_tool_name_and_version
embedding_cache_hit
```

경제적 primary endpoint는 단순 `mAP/min` 비율 하나가 아니다.

- fixed-time \(\Delta\)mAP50–95
- rare-class safety
- completed unique images와 completed boxes
- review/adjudication을 포함한 person-minutes

performance-per-hour와 금액은 secondary다. 금액은 문서화된 인건비와 GPU 단가가 있을 때만 계산한다.

## 11. Novelty and Claim Boundary

### 후보 contribution 판정

| 후보 | 판정 | 설명 |
|---|---|---|
| equal-image와 equal-human-time에서 AL ranking 비교 | **PARTIALLY NOVEL** | fixed-time AL user study는 존재하지만 산업 결함과 frozen translation chain 조합은 차별화 가능 |
| selection advantage를 actual annotation burden에 연결 | **PARTIALLY NOVEL** | cost-aware/box-budget 연구는 이미 있으나 실제 active time 계측이 추가 가치 |
| human-label quality를 통한 learner translation | **PARTIALLY NOVEL** | 사람이 만든 label의 오류와 learner utility를 동일 실험에서 분해하는 것이 핵심 |
| overall utility와 rare-defect safety 충돌 | **PARTIALLY NOVEL** | 새 metric은 아니지만 현재 결과와 산업 결함 맥락의 중요한 empirical finding |
| frozen minimum-effect 기반 Phase B authorization | **PARTIALLY NOVEL** | 알고리즘 novelty가 아니라 evaluation discipline |
| industrial defect domain-specific evidence | **PARTIALLY NOVEL** | GC10 한 데이터셋이면 case study; 외부 lot/site 전에는 일반화 불가 |

### 최종 contribution은 최대 3개

1. **산업 결함 object detection에서 equal-image selection advantage를 equal-human-time learner utility로 번역하는 prospectively frozen empirical audit**
2. **실제 human annotation active time과 label omission/class/localization quality를 downstream learner 결과에 연결한 분석**
3. **overall learning gain과 rare-defect safety가 충돌할 수 있음을 precommitted gate로 분리한 사례**

### 허용 가능한 claim

> GC10 development setting에서, retrospectively identified DINO visual-diversity candidate의 equal-image advantage가 사전 동결된 human-time·label-quality·equal-time learner·rare-safety endpoint로 번역되는지를 bounded prospective audit로 평가했다.

### 금지할 claim

- 최초의 cost-aware Active Learning
- 최초의 human-time AL evaluation
- 최초의 box-level AL
- 최초의 foundation-assisted annotation
- universal evaluation framework
- production cost saving proven
- DINO generally outperforms Random
- 2명의 annotator 결과를 산업 작업자 전체에 일반화
- GC10 결과를 외부 lot/site 재현성으로 표현

## 12. Estimated Workload

### 확정 가능한 task 수

| 단계 | 설계 | task/run 수 |
|---|---|---:|
| Stage 0 smoke | 24 images × 2 annotators | 48 human tasks |
| Phase A | 233 unique images × 2 annotators | 466 human tasks |
| adjudication | 불일치 건수에 따라 변동 | 현재 NA |
| Phase B | 10 acquisition × 2 strategies × 3 training seeds | 60 learner runs |
| Phase C | 현재 범위 밖 | 0 |

### 사람시간 시나리오

실제 per-image 시간이 없으므로 아래는 계획용 산식일 뿐 결과 추정이 아니다.

| 가정한 평균 active time | Stage 0+Phase A 514 tasks | 의미 |
|---:|---:|---|
| 2 min/task | 17.1 person-hours | 단순 이미지가 많은 경우 |
| 5 min/task | 42.8 person-hours | 중간 시나리오 |
| 10 min/task | 85.7 person-hours | 복합·다중 box가 많은 경우 |

여기에 training, guide, break, adjudication, setup 시간이 추가된다. 따라서 Stage 0 측정 전에는 달력 기간과 인건비를 확정하지 말아야 한다.

기존 detector runtime을 단순 선형 적용하면 Phase B 60-run은 약 80.6분의 detector wall time이지만, 이는 기존 장비·config가 그대로 유지된다는 조건의 **계획용 외삽**이다. 실제 GPU 모델과 peak memory가 기록되지 않아 경제적 비용 근거로 사용할 수 없다.

## 13. Go/No-Go Decision

### 결정 코드

**`GO_AFTER_PROTOCOL_REVISION`**

### 실행 전 필수 수정 5개

1. **seed 5–14 고정:** learner-unused이지만 selection outcome은 이미 본 seed라는 사실을 명시하고 결정론적 선택 규칙으로 동결
2. **smoke/Phase A 분리:** 24장의 instrumentation smoke를 233장의 Phase A union과 완전히 분리
3. **정확한 event logger:** per-image start/pause/resume/submit, box create/update/delete, class change와 active milliseconds를 직접 기록
4. **human-label quality/adjudication:** Hungarian matching, FN, small/rare recall, class accuracy, duplicate, correction time을 사전 동결
5. **\(T\)·통계 단위·cost freeze:** Stage 0 후 Phase A 전에 primary \(T\), minimum effect, seed-level inference, rare safety, compute/person-time 필드를 Git commit

### 승인 조건

다음 조건이 충족되기 전에는 Phase A를 시작하지 않는다.

- annotator 최소 2명 확보
- 기관의 인간 참여 연구/동의/면제 요구에 대한 공식 판단 확보
- Phase A와 disjoint한 smoke images 확정
- logger smoke PASS
- seed 5–14 manifest와 query rank hash
- primary \(T\) 산출 규칙 및 Phase A authorization margin freeze
- human label 사용·adjudication 절차 합의
- official/final test lock 재확인

### 최종 해석

이 연구의 가치는 “DINO가 더 좋음을 다시 증명”하는 데 있지 않다. 오히려 세 결과가 모두 학술적으로 남는다.

- **equal-time learning PASS + rare safety PASS:** 특정 GC10 조건에서 selection advantage가 실제 비용효용으로 번역됨
- **learning gain + rare safety FAIL:** 전체 효용과 희귀결함 안전성이 충돌함
- **translation not observed:** equal-image selection advantage가 실제 human-time/label-quality를 거쳐 learner utility로 유지되지 않음

어느 결과가 나오든 protocol을 사전에 동결했다면 현재 Translation 연구의 한 단계를 실제 비용 단위로 발전시킨다. 반대로 기존 137장을 pilot과 confirmatory에 함께 쓰거나 Phase A를 본 뒤 \(T\)·minimum effect·safety rule을 바꾸면, 결과가 좋아도 강한 증거가 되지 못한다.

---

## Audit Traceability

본 감사에서 직접 확인한 주요 로컬 근거:

- `docs/final_repository_research_audit_20260730.md`
- `docs/노션_업로드용_최종_연구방향_재정의_및_검토요청_20260730.md`
- `docs/gc10_taxonomy_selection_protocol_20260715.md`
- `docs/gc10_development_detector_confirmation_protocol_20260715.md`
- `scripts/03_analysis/run_gc10_taxonomy_selection_audit.py`
- `scripts/02_active_learning/run_gc10_development_detector_confirmation.py`
- `runs/gc10_taxonomy_selection_audit/gc10_random_vs_dino_200seed_20260715/gc10_selection_records_posthoc.csv`
- `runs/gc10_detector_confirmation/gc10_dev_confirm_5acq_3train_20260715/`
- `outputs/gc10_visual_embeddings/dinov2_small_protocol_20260715/embedding_config.json`

본 감사 중 수행하지 않은 작업:

- 코드 수정
- 새 모델 학습
- 인간 annotation
- official/final test 접근
- threshold·seed·metric의 결과 기반 재선택
