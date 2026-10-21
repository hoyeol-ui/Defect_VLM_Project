# Thesis claim boundary v3

Date: 2026-07-18  
Evidence Freeze v3 decision: **A. THESIS_CORE_STRENGTHENED_BY_PROSPECTIVE_STOP**

## Supported

- 산업 결함 AL 후보 신호의 `signal-discovery-safety-reproducibility-learning-operational` 단계는 서로 대체할 수 없다.
- GC10, MPDD, VisA의 fixed-pool discovery gain은 composition safety 또는 downstream utility와 분리됐다.
- DeepPCB에서 외부 데이터 protocol, score, query fraction, selection hash와 primary gate가 GT audit 전에 고정됐다.
- DeepPCB primary total-bbox gate가 실패한 뒤 detector screen이 중단됐다.
- Positive secondary small-box signal은 parent PASS로 승격되지 않고 mechanism audit과 development-only benchmark로 분리됐다.
- DeepPCB detector training/inference actual count는 0이고 official/final test actual use는 0이다.
- 이 사례는 workflow의 prospective authorization conformance와 claim-boundary enforcement를 강화한다.

## Exploratory only

- DeepPCB bbox area 577-1024 px² 구간과 frozen residual selection의 관계.
- DeepPCB small-share uplift 1.1828, group-bootstrap 95% CI [1.0814,1.3711].
- Parent secondary small enrichment 1.307591.
- V2.3 FN 1.379693, rare-FN 1.785910, confidence rare-FN 2.683022.

## Rejected or closed

- DeepPCB total bbox-richness utility: 1.107693 < frozen 1.20.
- General small-defect annotation triage: Phase A A2_MECHANISM_AMBIGUOUS.
- Independent reference-residual candidate: Phase B NO_CANDIDATE.
- DeepPCB detector improvement: not authorized and not tested.
- Universal VLM/DINO/detector-uncertainty selector superiority.

## Not identifiable

- Workflow의 prospective predictive accuracy, early-stop recall, false-advance rate와 good-selector sensitivity.
- STOP branch의 counterfactual detector outcome.
- Independent production generalization.
- Actual annotation-time or monetary-cost reduction.

## Literature-positioning boundary (2026-07-19 update)

- **Supported by adjacent direct work:** Object detection AL의 detector-training cost, validation-set-dependent ranking, detector-training-free selected-set evaluation 필요성은 Sbeyti et al.의 OSS 연구가 직접 지지한다.
- **Mechanistically plausible only:** 일반 VQA의 sampled-path self-consistency 결과를 본 연구의 cross-prompt SBERT consistency와 동일시하지 않는다.
- **Moderately supported only:** AnomalyVFM은 VFM의 anomaly-specific adaptation 필요성을 지지하지만, 본 데이터의 full-image DINO source/background confounding을 직접 입증하지 않는다.
- **Insufficiently supported:** OWL-ViT pseudo-box의 미세 산업 결함 localization 실패를 generative VLM hallucination 또는 PostAlign 근거만으로 설명하지 않는다.
- **Internally unresolved:** NEU-DET에서 Random이 강한 정확한 원인은 외부 문헌으로 확정하지 않고 내부 composition/acquisition audit 범위에 둔다.

## DeepPCB 금지 표현

- DeepPCB를 selector 성공 사례로 표현하지 않는다.
- 1.307591을 confirmatory PASS로 표현하지 않는다.
- bbox area 577-1024 px² 결과를 tiny defect 전반의 효과로 표현하지 않는다.
- M1/M2/M3 결과를 literature methods의 일반적 무효성으로 표현하지 않는다.
- DeepPCB STOP 사례를 predictive screening accuracy로 표현하지 않는다.
- 문서화되지 않은 prevented model-run 수를 추가하지 않는다.
- DeepPCB official test, detector 학습 또는 score/threshold rescue를 재개하지 않는다.

## 문헌·novelty 금지 표현

- “detector 학습 전 AL 평가를 최초로 제안했다” 또는 “기존에 이런 workflow가 없다”고 쓰지 않는다.
- Sbeyti et al.이 본 실험의 Random 우위나 proxy-mAP 불일치 원인을 증명했다고 쓰지 않는다.
- `완벽히 설명`, `원인을 입증`, `Strong direct evidence`를 인접 문헌과 내부 결과의 연결에 사용하지 않는다.
- Mann et al.의 sampled reasoning-path agreement를 본 연구의 cross-prompt SBERT cosine과 같은 지표로 쓰지 않는다.
- AnomalyVFM을 full-image DINO의 source confounding에 대한 direct evidence로 분류하지 않는다.
- OWL-ViT를 autoregressive generative VLM과 동일한 hallucination 메커니즘으로 단정하지 않는다.

## 논문 본체 방화벽

1. 논문 본체의 성립 조건은 future positive selector가 아니다.
2. DeepPCB 결과가 A1 또는 candidate PASS였어도 external utility는 별도 독립 확인이 필요했다.
3. 현재 A2와 NO_CANDIDATE는 optional extension을 닫지만 retrospective core evidence를 소거하지 않는다.
4. 이번 사례가 강화하는 것은 success prediction이 아니라 `freeze -> audit -> authorization STOP -> locked test preservation`의 실제 준수다.
5. 교수님이 positive algorithm을 필수로 요구하면 해당 연구는 별도의 supervised/reference-inspection 주제와 split으로 분리한다.
