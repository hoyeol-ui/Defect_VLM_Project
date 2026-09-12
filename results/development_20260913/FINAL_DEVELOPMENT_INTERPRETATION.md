# Development 통합 결과의 해석

「소량 라벨 환경에서 특징 기반 능동학습을 활용한 금속 표면 결함 객체 탐지」

Evidence: Development validation; Statistical unit: acquisition seed, n=10; A/B averaged within acquisition seed; Locked-final: not used

전체 비교 등급: **B (조건부 development 비교)**. 동일 split·numeric seed·학습 설정·추출 규칙은 확인했으나 과거/신규 GPU 및 OS 효과를 분리하지 못했다. Bootstrap은 10개 acquisition seed를 짝지어 100,000회 복원추출한 기술적 구간이며 confirmatory multiplicity family는 이번에 만들지 않았다.

## Q1. 특징 기반 DINO 선택은 Random 대비 label efficiency를 개선하는가?

**SUPPORTED — 현재의 평균 development 비교 범위.** DINO−Random nAULC는 GC10 +0.391pp, 95% CI [+0.034,+0.796], 양/음 8/2; NEU +0.792pp, CI [+0.344,+1.235], 양/음 9/1이다. 모든 seed에서 개선한다는 뜻은 아니다.

## Q2. 그 효과는 GC10과 NEU에서 모두 관찰되는가?

**SUPPORTED.** 두 dataset의 평균 차이가 모두 양수다. GC10 R4에서는 −0.305pp로 평균 역전이 있어 budget별 일관성까지 보장하지 않는다. Historical locked-final에서 DINO−Random +0.581pp/+0.578pp였다는 기존 근거는 별도로 유지한다.

## Q3. Entropy와 비교하면 DINO는 경쟁력이 있는가?

**PARTIALLY SUPPORTED.** DINO−Entropy는 GC10 −0.256pp (CI [−0.808,+0.270]), NEU +0.079pp (CI [−0.283,+0.447])이며 양/음 seed가 각각 5/5다. 평균 효율이 비슷한 범위에 놓여 있으나, 비열등성·동등성 margin과 검정이 없어 통계적 경쟁력이나 동등성을 확정하지 않는다.

## Q4. DINO 또는 Entropy가 모든 budget/seed/class에서 우수한가?

**NOT SUPPORTED.** seed별 효과 부호와 budget 평균 순위가 바뀐다. NEU DINO−Entropy는 R1 +0.177pp, R2 −0.140pp, R3 +0.374pp, R4 +0.078pp, R5 −0.189pp다. 모든 클래스의 공통 development AP 자료도 없어 classwise 우월성은 **NOT YET EVALUABLE**이다. 선택 구성의 차이는 AP 차이의 대용물이 아니다.

## Q5. Full-data 대비 소량 라벨의 성능은 어느 수준인가?

**SUPPORTED — descriptive reference 비교에 한정.** GC10에서 DINO/Entropy는 각각 full의 87.37%/87.39%, NEU는 90.48%/90.92%다. 이미지 비율은 각각 34.95%/35.03%이며 full 대비 약 3.90–4.38pp 낮다. reference는 A/B 두 learner 평균이고 n=10의 독립 reference가 아니며 upper bound도 아니다. 이미지 수를 annotation 비용으로 치환하지 않는다.

## Q6. 현재 결과가 학위논문 제목과 연구목적을 지지하는가?

**SUPPORTED — 두 데이터셋, YOLOv8n, 고정 특징·예산 범위 안에서.** 특징 선택의 Random 대비 이득, detector uncertainty와의 비교, 선택 구성의 차이 및 full-label 대비 위치를 제시할 수 있다. 제목은 활용과 label efficiency 연구를 말하며 포괄적인 알고리즘 우월성을 요구하지 않는다. 최종 3-way test 성능의 증거는 아직 별도 필요하다.

## Q7. 추가 AL training 없이 논문을 닫는 판단은 합리적인가?

**SUPPORTED — 학습 실험 범위를 닫는 판단.** Random/DINO 440 + Entropy 200 + Full-data 4로 현재 질문을 다룰 수 있다. TypiClust/BADGE/mixture 등은 추가 질문을 위한 선택 강화이며 이번 결과를 더 유리하게 만들 목적으로 붙이지 않는다. 논문 전체의 최종 검증이 이미 끝났다는 뜻은 아니다. 신규 204 checkpoint의 별도 frozen evaluation/statistics release와 승인된 같은 locked-final 평가가 남아 있다. 이번에는 실행하지 않았다.

## 시나리오와 데이터셋 간 차이

**B에 가까운 기술적 양상 + D/E의 제한.** 두 AL 방법은 Random보다 평균 nAULC가 높고 직접 차이는 작지만, formal equivalence를 증명하지 않았다. 점추정 순위는 GC10 Entropy>DINO>Random, NEU DINO>Entropy>Random으로 달라진다. GC10의 Entropy 초기 R1 이득은 더 크지만 NEU R1에서는 DINO가 앞서므로 초기 우월성도 보편적이지 않다. GC10 직접 차이는 R5에 가까워지지만 단조 감소는 아니며 NEU에는 일관된 수렴 패턴이 없다.

Entropy score–검출 수 Spearman의 query 중앙값은 GC10 .918, NEU .945다. SUM aggregation의 크기 효과와 연관된 관찰이며 알고리즘을 재정의하지 않는다. DINO는 frozen feature geometry, Entropy는 detector 예측 신호를 통해 다른 class mixture를 만든다. 모든 class/round 진단과 seed composition–nAULC 상관은 사후 탐색으로 공개하고 인과효과나 유의한 class 발견으로 해석하지 않는다.
