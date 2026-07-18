# 지도교수 결정 보고: Evidence Freeze v2

## 연구 방향 변경 요약

최초 연구는 VLM Self-Consistency를 epistemic uncertainty로 간주하고, 이를 DINO diversity·detector uncertainty와 결합한 selector가 Random보다 detector mAP를 안정적으로 높인다는 가설에서 출발했습니다. 초기 계획에는 +2~3% mAP, 50~80% label-cost 절감, domain adaptation, local deployment, 검사원 신뢰 향상까지 포함됐습니다.

현재 원자료는 이 중심 가설을 지지하지 않습니다. VLM consistency-groundedness gate, oracle crop, 세 소형 VLM 비교가 모두 실패했고, NEU의 seed45 positive set은 새 acquisition seeds에서 재현되지 않았습니다. GC10에서는 selection coverage가 좋아져도 rare AP·recall·mAP로 번역되지 않았습니다. 따라서 논문의 중심을 새 selector의 성능 우위가 아니라 **산업 결함 AL 후보 신호의 단계적 타당성과 번역 실패 조건을 검증하는 empirical evaluation**으로 변경하고자 합니다.

## 무엇이 실패했는가

`낮은 consistency = 높은 epistemic uncertainty`, `VLM/DINO/uncertainty selector > Random`, `global coverage = detector utility`, `flip disagreement = epistemic uncertainty`, `above-chance error ranking = operational AL utility`, `200 seeds = independent production generalization`이라는 연결이 성립하지 않았습니다. 초기의 +2~3% mAP, label-cost 50~80%, domain adaptation, inspector trust 목표도 달성 또는 검증되지 않았습니다.

## 무엇이 연구 자산으로 남았는가

- **Positive evidence**: GC10 rare discovery +2.720, MPDD anomaly +6.245, VisA anomaly +14.480; seed45 fixed-set mAP +0.016236; V2.3 AUROC 0.766432.
- **Failure mechanisms**: VLM grounding collapse, global/local representation mismatch, category/source concentration, acquisition-set non-generalization, coverage-utility gap, ranking-operational enrichment gap, metadata-independent-pool gap.
- **Workflow**: Signal -> Discovery -> Safety -> Acquisition Reproducibility -> Learning Utility -> Operational Validity의 단계별 frozen gate.
- **Cost prevention**: 명시적으로 계획된 detector model run 최소 45개(D2R 15 + YOLOv8s 30)를 시작하지 않았고 final test는 0회 사용했습니다. GPU-hours나 금액은 측정 자료가 없어 주장하지 않습니다.
- **Reusable assets**: 200-seed selection records, manifests, DINO cache, YOLO checkpoints, 15개 sealed Random140 predictions, gate code, metadata provenance, unit tests.

## 새 중심 연구 질문

> 산업 결함 데이터에서 Active Learning 후보 신호의 signal validity, target discovery, composition safety, acquisition-set 재현성, downstream detector utility와 operational enrichment는 어떻게 분리되며, 고비용 학습과 locked evaluation 전에 이 번역 실패를 어떻게 판정할 수 있는가?

## 새 중심 기여

1. 산업 결함 AL 후보 신호를 여섯 단계로 분해하고 다음 단계 실행을 frozen gate로 제한하는 validity-gated evaluation framework.
2. GC10·MPDD·VisA에서 fixed-pool discovery gain과 composition safety가 분리됨을 보인 paired empirical evidence.
3. GC10/NEU에서 selection coverage와 training-seed stability가 detector utility/acquisition generalization을 보장하지 않음을 보인 translation evidence.

## 현재 추가 실험이 불필요한 이유

추가 selector search는 기존 failure mechanism을 해결하는 독립 자원 없이 development reuse와 사후 method search를 늘립니다. 현재 benchmark metadata는 independent production pools를 만들지 못하며, FN branch도 untouched non-final bbox validation이 없습니다. 따라서 지금은 결과를 더 만드는 것보다 claim boundary와 evidence provenance를 동결하고 본문을 작성하는 편이 연구 윤리에 맞습니다. Locked final은 계속 보호합니다.

## 교수님께 결정받아야 할 항목

- **A. 권고**: Validity-gated empirical evaluation을 학위 중심 기여로 인정하고 논문 작성을 시작한다.
- **B. 조건부**: 제한적 positive algorithmic contribution을 추가로 요구한다면, 새 independent production pool/untouched validation 확보 전까지 구현은 보류한다.
- **C. 분리**: supervised detector/backbone/small-defect capacity 연구가 필요하면 본 AL evidence와 별도 연구 질문·split·protocol로 분리한다.

## 교수님께 구두로 말씀드릴 1분 설명

“교수님, 처음에는 VLM 설명 일관성과 DINO·detector uncertainty를 결합한 selector가 Random보다 안정적으로 성능을 높일 것으로 예상했습니다. 하지만 consistency와 grounding은 oracle crop과 세 모델에서도 유효성을 통과하지 못했고, NEU의 positive seed도 새 acquisition set에서는 재현되지 않았습니다. 반면 GC10, MPDD, VisA에서 target discovery 자체는 실제로 증가했지만, category/source safety 또는 rare-class detector utility와 분리됐습니다. 특히 GC10에서는 coverage가 좋아져도 mAP, rare AP, recall로 번역되지 않는 결과가 두 protocol에서 확인됐습니다. 그래서 selector 성공을 주장하는 대신, 신호·발견·안전성·재현성·학습 효용·운영 효용을 단계적으로 검증하고 실패 시 학습과 final test를 중단하는 validity-gated 평가를 논문의 중심으로 바꾸고 싶습니다. 현재 결과만으로 본문 작성은 가능하다고 판단하지만, 이 empirical evaluation을 학위의 중심 기여로 인정할지, 아니면 별도의 supervised detector 연구를 분리해 요구하실지 결정 부탁드립니다.”
