# Reframed thesis outline

## Recommended titles

### Korean
1. **산업 결함 능동학습 후보 신호의 단계적 타당성 평가: 발견-안전성-학습 효용 간극 분석**
   - 장점: 초기 문제와 현재 증거를 가장 정확히 묶는다.
   - 위험: 새 selector가 없어 방법론 기여가 약하다는 질문을 받을 수 있다.
2. **산업 결함 검출을 위한 Validity-Gated Active Learning 평가 체계**
   - 장점: 평가 workflow를 전면에 둔다.
   - 위험: 실증된 failure mechanism을 부제나 초록에서 구체화해야 한다.
3. **산업 결함 Active Learning에서 후보 신호의 발견 효용과 검출 학습 효용의 분리**
   - 장점: selection-learning gap이 선명하다.
   - 위험: VLM과 metadata 감사 범위가 제목에서 덜 드러난다.
4. **산업 결함 데이터의 Annotation-Efficient Learning을 위한 Acquisition Signal Audit**
   - 장점: annotation triage와 AL을 함께 포괄한다.
   - 위험: 국문 제목에 영문 용어가 많다.
5. **산업 결함 능동학습의 실패 조건과 중단 규칙에 관한 실증 연구**
   - 장점: negative/conditional contribution이 솔직하다.
   - 위험: 긍정적 discovery 결과가 제목에서 약해 보일 수 있다.

### English
1. **Validity-Gated Evaluation of Acquisition Signals for Industrial Defect Active Learning: Discovery, Safety, and Learning-Utility Gaps**
   - Strength: Covers the full empirical contribution without a superiority claim.
   - Risk: Long title; may be shortened after advisor review.
2. **A Validity-Gated Audit of Active Learning Signals for Industrial Defect Detection**
   - Strength: Compact and methodologically clear.
   - Risk: The discovery-safety distinction must be explicit in the abstract.
3. **When Selection Gains Do Not Translate: An Empirical Study of Industrial Defect Active Learning**
   - Strength: Memorable and centered on the strongest translation result.
   - Risk: Could sound exclusively negative unless balanced in the subtitle.
4. **Separating Discovery, Composition Safety, and Detector Utility in Industrial Active Learning**
   - Strength: Directly names the three stages.
   - Risk: The VLM validity branch is implicit rather than explicit.
5. **Failure-Condition Mapping and Stopping Rules for Annotation-Efficient Industrial Defect Learning**
   - Strength: Highlights operational value and stopping discipline.
   - Risk: Active Learning should be prominent in keywords and abstract.

## Chapter plan

### 제1장 서론

- 내용: 산업 결함 annotation 비용; acquisition signal 검증 문제; 강한 Random; 연구 질문과 기여
- 사용할 표: Table 1 initial-to-revised hypothesis summary
- 사용할 그림: Fig. 1 research hypothesis evolution
- 사용할 실험/핵심 주장: 초기 계획과 최종 질문의 차이
- 제외할 내용: 비용 50-80%, 신뢰 향상 수치
- 예상 심사 질문: 왜 새 알고리즘이 아니라 평가 연구인가?

### 제2장 관련 연구

- 내용: AL uncertainty/diversity; object detection AL; VLM semantic uncertainty; groundedness/hallucination; industrial anomaly/defect; evaluation pitfalls
- 사용할 표: Table 2 literature-to-risk map
- 사용할 그림: 없음 또는 taxonomy diagram
- 사용할 실험/핵심 주장: 문헌이 예고한 consistency!=truth, cross-model risk, Random strength
- 제외할 내용: 검증되지 않은 현장 통계
- 예상 심사 질문: Negative result의 novelty는 무엇인가?

### 제3장 Validity-Gated Evaluation Framework

- 내용: Signal, Selection, Composition Safety, Acquisition Generalization, Learning Utility, Operational Validity; gates and stopping
- 사용할 표: Table 3 gate definitions and authorization
- 사용할 그림: Fig. 2 evidence pyramid
- 사용할 실험/핵심 주장: 다음 단계는 이전 gate PASS로만 허용
- 제외할 내용: 하나의 universal threshold 주장
- 예상 심사 질문: Branch-specific gate가 사후적이지 않은가?

### 제4장 VLM 신호 타당성 감사

- 내용: legacy consistency; structured grounding; oracle crop; paired model comparison
- 사용할 표: Table 4 VLM validity results
- 사용할 그림: response-collapse examples; claim boundary inset
- 사용할 실험/핵심 주장: 0/6 legacy and 0/3 model pass; grounding collapse
- 제외할 내용: VLM acquisition superiority
- 예상 심사 질문: 모델 크기를 키우면 달라지지 않는가?

### 제5장 Selection Discovery와 Composition Safety

- 내용: GC10, MPDD, VisA paired effects; DINO mechanism; metadata audit
- 사용할 표: Table 5 dataset-specific discovery/safety; Table 6 metadata feasibility
- 사용할 그림: Fig. 3 discovery-composition-utility matrix
- 사용할 실험/핵심 주장: Same-pool discovery exists, safety is dataset-specific
- 제외할 내용: Cross-dataset raw average; sparsity law
- 예상 심사 질문: 200 seeds가 독립 pool인가?

### 제6장 Selection-Learning Translation

- 내용: Random audit; q20 translation; K40; seed45; independent confirmation
- 사용할 표: Table 7 translation contrasts; Table 8 Random properties
- 사용할 그림: selection-learning translation diagram
- 사용할 실험/핵심 주장: coverage/training stability do not guarantee utility/generalization
- 제외할 내용: positive mAP without rare loss; training seeds as independent acquisitions
- 예상 심사 질문: 왜 overall mAP +0.017을 성공이라 하지 않는가?

### 제7장 Detector-Native Signal과 Operational Gate

- 내용: flip disagreement; V2.3 sealed predictions; ranking vs enrichment; FN exploratory
- 사용할 표: Table 9 signal validity and operational thresholds
- 사용할 그림: AUROC vs enrichment schematic
- 사용할 실험/핵심 주장: above-chance ranking can fail operational threshold
- 제외할 내용: FN triage confirmation
- 예상 심사 질문: AUROC 0.766이면 충분하지 않은가?

### 제8장 종합 논의

- 내용: discovery!=safety; safety!=utility; stability!=generalization; metadata!=production; cost prevention/final protection
- 사용할 표: Table 10 mechanism matrix; Table 11 claim boundary
- 사용할 그림: Fig. 4 claim boundary map
- 사용할 실험/핵심 주장: Repeated translation failures form the empirical contribution
- 제외할 내용: production generalization or universal laws
- 예상 심사 질문: Checklist 이상의 학술적 기여는 무엇인가?

### 제9장 결론 및 향후 연구

- 내용: primary/secondary contributions; limits; independent production pools; FN restart conditions; separate supervised detector branch
- 사용할 표: Table 12 future authorization criteria
- 사용할 그림: none
- 사용할 실험/핵심 주장: 현재 evidence freeze와 future protocol의 분리
- 제외할 내용: 추가 selector 결과 예측
- 예상 심사 질문: 추가 실험 없이 논문을 써도 되는가?
