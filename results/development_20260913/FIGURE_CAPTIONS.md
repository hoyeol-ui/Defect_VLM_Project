# 대표 그림 설명

논문 제목: 「소량 라벨 환경에서 특징 기반 능동학습을 활용한 금속 표면 결함 객체 탐지」

1. **Research workflow.** Sealed pool과 공통 R0에서 Random, frozen DINOv2 cosine farthest-first, NMS detection-confidence binary entropy SUM을 비교한다. Entropy parent는 직전 learner A이며 detector는 모두 cold start한다. R0의 물리적 재학습은 없다. Full-data는 별도 A/B reference다.
2. **Three-way development curves.** Evidence: Development validation; Statistical unit: acquisition seed, n=10; A/B averaged within acquisition seed; Locked-final: not used. 선은 seed 평균, 음영은 seed SD다. SD는 평균의 CI가 아니다. 각 cell은 100-epoch CSV의 최고 mAP50–95를 동일하게 사용한다. validation 선택 편향 및 GPU/OS 차이로 test 성능을 대신하지 못한다.
3. **Paired nAULC.** Evidence: Development validation; Statistical unit: acquisition seed, n=10; A/B averaged within acquisition seed; Locked-final: not used. 회색 점은 10 seed, 마름모는 평균, 막대는 100,000회 paired percentile bootstrap CI. 두 endpoint 점에 0.1, 내부 네 점에 0.2의 가중치가 적용된다. 구간이 0을 포함하는 것과 동등성은 다르다.
4. **R5 versus full-label reference.** Evidence: Development validation; Statistical unit: acquisition seed, n=10; A/B averaged within acquisition seed; Locked-final: not used. AL의 점은 10개 A/B 평균이고 Full의 ×는 실제 learner 두 개다. Full을 n=10으로 복제하거나 upper bound로 취급하지 않는다. 약 35%는 이미지 수 비율이며 annotation 비용 비율이 아니다.
5. **Selection behavior.** Evidence: acquisition-only diagnostics; 10 acquisition trajectories; 5 dependent rounds each; locked-final not used. 위: 각 query의 선택/현재 후보 class prevalence 차이(pp)를 seed·round에 걸쳐 평균한다. 가운데: 기존 branch의 class pairwise cosine distance와 R5 DINO−Random 누적 class 선택 수. 작을수록 compact하며, numeric class ID가 다른 NEU 자료는 class name으로 대조했다. 아래: saved entropy SUM과 n_detections의 query 내부 Spearman; 점은 각 seed, 선은 해당 round 중앙값. 라벨을 보고 새 선택을 만들지 않았으며, 표본 구성과 성능의 인과관계는 검증하지 않았다.

공통 제한: 모든 AL 통계 표본 수는 training 횟수가 아닌 acquisition seed n=10이다. GC10/NEU는 별도로 분석하며 seed ID가 같다는 이유로 dataset 사이를 짝지어 추론하지 않는다. 원래 locked-final 결과는 별도 표에만 둔다.
