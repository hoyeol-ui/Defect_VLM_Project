# Thesis claim boundary

## 반드시 주장해야 하는 것

- 산업 결함 AL 후보 신호의 `signal-discovery-safety-reproducibility-learning-operational` 단계는 서로 대체할 수 없다.
- Fixed benchmark same-pool paired records에서 GC10, MPDD, VisA의 target discovery gain이 확인됐다.
- Discovery gain은 VisA category collapse, MPDD source confound, GC10 rare-utility loss와 공존했다.
- GC10 K40 selection coverage PASS는 downstream mAP/rare AP/recall PASS로 번역되지 않았다.
- NEU 한 고정 선택 집합의 training-seed 안정성은 새로운 acquisition-set confirmation을 통과하지 못했다.
- 모든 branch에서 final test를 사용하지 않았고, frozen gate가 최소 45개 후속 detector model run을 차단했다.

## 조건부로 주장할 수 있는 것

- GC10 rare discovery +2.720, MPDD anomaly discovery +6.245, VisA anomaly discovery +14.480은 **동일 fixed acquisition pool의 200 paired leave-20 perturbations** 범위에서만 말한다.
- GC10 first translation의 overall mAP +0.017378은 해당 development protocol의 부분 효과이며 rare AP -0.019877과 항상 함께 보고한다.
- MPDD capture-day 및 GC10 filename-sequence 결과는 mechanism sensitivity proxy다. Production lot/generalization이 아니다.
- V2.3 AUROC 0.766432는 above-chance ranking evidence이며 operational AL validity가 아니다.

## Exploratory로만 남겨야 하는 것

- Combined FN enrichment 1.379693.
- Combined rare-FN enrichment 1.785910.
- Confidence-only rare-FN enrichment 2.683022.
- Local feature misalignment, recall-repair triage, human-facing explanation interface.

## 절대로 주장하면 안 되는 것

- 새 selector가 Random보다 일반적으로 우수하다.
- Target sparsity가 gain을 조절한다는 법칙을 확인했다.
- 200 seeds는 200 independent production pools다.
- MPDD EXIF day는 production lot이다.
- GC10 filename group은 official production group이다.
- VisA category는 capture session이다.
- VLM consistency가 epistemic uncertainty임을 증명했다.
- Label cost 50-80% 절감, detector performance improvement guarantee, inspector trust 향상, domain adaptation 향상, 0.2 s/image deployment를 달성했다.

## 용어 사용 규칙

| 용어 | 허용 사용 | 금지 사용 |
|---|---|---|
| validity | 사전 정의된 endpoint/gate 충족 여부 | 일반적 진실성의 동의어 |
| robustness | 명시된 training/acquisition seed 범위 | production robustness |
| generalization | 새 acquisition sets 등 실제 독립 축을 명시 | fixed-set 반복을 일반화로 표현 |
| independent pool | source-documented, target-blind production unit만 | leave-k perturbation, category, EXIF proxy |
| uncertainty | 계산된 signal 이름 또는 후보 신호 | epistemic uncertainty로 단정 |
| diversity | 사용한 metric(pairwise similarity, coverage, entropy)을 명시 | 시각적으로 다양하다는 모호한 표현 |
| discovery | fixed query budget의 target image count | detector utility와 동일시 |
| safety | category/source/rare/recall gate의 구체적 endpoint | 모든 위험을 포괄하는 표현 |
| utility | detector mAP/rare AP/recall 또는 review enrichment를 명시 | coverage/discovery를 utility로 승격 |
| efficiency | 실제 model-run prevention처럼 측정된 값 | 시간·비용 미측정 상태의 경제성 주장 |
| label cost reduction | matched-performance annotation cost를 측정한 경우만 | 50-80% 기대값을 결과로 사용 |
| explainability | 생성 설명의 존재 또는 interface 기능 | grounding/인간 신뢰가 검증됐다는 표현 |
