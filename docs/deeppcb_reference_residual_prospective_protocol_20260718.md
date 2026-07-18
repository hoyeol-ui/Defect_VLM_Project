# DeepPCB Prospective Reference-Residual Annotation Triage Protocol

**Frozen date:** 2026-07-18  
**Branch identity:** prospective single-branch selection-only demonstration  
**Primary purpose:** validity-gated authorization, not guaranteed selector success

## 1. 연구 질문

정렬된 defect image–clean template 쌍을 제공하는 산업 검사 조건에서, GT-free local reference residual richness가 같은 이미지 budget의 Random보다 bbox-rich annotation candidates를 안정적으로 농축하면서 class/small-defect safety를 유지하는가?

이 branch는 기존 global DINO/VLM selector를 사후 수정하지 않는다. DeepPCB가 공식적으로 제공하는 paired reference condition을 이용해, local defect change에 직접 정렬된 단 하나의 candidate signal을 결과 확인 전에 고정한다.

## 2. 데이터 역할과 잠금

- Source: DeepPCB official repository, research use only.
- Official dataset: 1,500 aligned tested/template image pairs, 6 bbox classes.
- `PCBData/trainval.txt`: 1,000 images. 이번 selection-only development audit에서만 사용한다.
- `PCBData/test.txt`: 500 images. **LOCKED OFFICIAL TEST**로 유지한다.
- Official test image content, annotation content, detector evaluation은 이번 branch에서 사용하지 않는다.
- Trainval의 `group92000` 111장은 future bounded detector screen의 development evaluation group으로 예약한다.
- Selection-only gate는 나머지 6 source/design groups에서 평가한다.
- directory group을 production lot으로 주장하지 않는다. 이번 분석의 inferential unit과 source-stratification unit으로만 사용한다.

## 3. 사전 event-count audit

Trainval에는 1,000 images, 6,873 boxes, 6 classes가 있으며 exact duplicate tested image는 0개다. 모든 7 trainval group에 6개 클래스가 존재한다. 이 audit은 score–GT 관계 또는 candidate selection outcome을 계산하지 않았다.

## 4. GT-free candidate signal

각 640×640 tested image `I_test`와 정렬된 clean template `I_temp`를 grayscale로 읽는다.

1. `D = abs(I_test - I_temp)`
2. binary residual mask: `D >= 32`
3. 8-connected components를 계산한다.
4. component area가 9 pixel 이상인 component만 유지한다.
5. `residual_area = retained residual pixel count`
6. `residual_components = retained component count`
7. 각 source group 내부에서 `log1p(residual_area)`와 `log1p(residual_components)`의 percentile rank를 계산한다.
8. Primary signal:

   `reference_residual_richness = 0.5 * area_rank + 0.5 * component_rank`

GT/XML/TXT annotation은 위 score 계산과 selection에 사용하지 않는다. Threshold, min-area, weighting은 결과 확인 후 변경하지 않는다.

## 5. Query policy

- 각 eligible source group에서 score 상위 20%를 선택한다.
- `query_n = ceil(0.20 * group_size)`
- tie는 canonical relative path의 오름차순으로 결정한다.
- Random baseline은 동일 group, 동일 query_n의 10,000회 무복원 추출 평균이다.
- Random seed는 `20260718`로 고정한다.
- selection file과 SHA256을 기록한 뒤에만 annotation을 결합한다.

## 6. Post-hoc endpoint

### G1 Signal validity

- group별 Spearman correlation: candidate score vs bbox instance count
- primary summary: 6개 eligible group의 평균 Spearman

Gate:

- mean Spearman >= 0.30
- positive Spearman in at least 5/6 groups

### G2 Target/annotation discovery

Primary endpoint:

- bbox instance enrichment@20% image budget vs group-matched Random

Gate:

- mean group enrichment >= 1.20
- group-bootstrap 95% CI lower bound > 1.00
- enrichment > 1.00 in at least 5/6 groups

Secondary diagnostics cannot rescue a failed primary gate:

- normalized bbox-area enrichment
- query images, boxes, boxes/image

### G3 Composition safety

- six-class coverage delta vs Random mean >= -0.25
- small-box enrichment >= 0.90
- max-class-share delta vs Random mean <= +0.10

Small box is frozen as bbox area <= 1,024 pixels (`32×32`, 0.25% of a 640×640 image).

## 7. 판정과 다음 단계

모든 G1–G3 check가 통과할 때만 `PASS_SELECTION_ONLY`다. PASS는 다음을 의미한다.

- external aligned-reference condition에서 local reference signal의 annotation-triage potential이 확인됨.
- 정확히 한 번의 bounded detector screen을 설계할 권한.

PASS가 의미하지 않는 것:

- Random 대비 detector 성능 향상 확정
- universal industrial AL selector
- annotation time/cost reduction
- independent production generalization
- official test 사용 권한

하나라도 실패하면 `STOP`한다. Signal 재혼합, threshold tuning, 다른 residual variant 탐색으로 rescue하지 않는다.

## 8. PASS 후에만 허용되는 bounded detector screen

- Train: 6 eligible trainval groups에서 동일 image budget으로 candidate vs group-stratified Random.
- Development: reserved `group92000` only.
- Learner: frozen YOLOv8n.
- Primary: development mAP50-95 delta.
- Safety: recall, worst-class AP, macro per-class AP.
- Candidate의 box 수 증가를 annotation-cost reduction으로 해석하지 않으며, box-matched Random sensitivity를 함께 보고한다.
- Official test 500장은 downstream development gate 통과 전까지 계속 잠근다.

## 9. Source provenance

- Official repository: `https://github.com/tangsanli5201/DeepPCB`
- Original paper: Tang et al., *Online PCB Defect Detector On A New PCB Defect Dataset*, arXiv:1902.06197.
- Local source commit and protocol/code SHA256 are recorded in the run directory.

