# clean split v2 최종 본실험: development 결과 해석 초안

**평가 경계.** 아래 숫자는 `ANNOTATION_PATCH_v2_0_2`의 162개 완료 모델 중 5 acquisition seed×R0–R5/Full-data의 `sealed_dev_metrics.json`을 재계산한 **development 탐색 결과**다. test는 봉인되어 있다. 단위는 mAP@0.50:0.95 **%**, 방법 차이와 Full-data gap은 **%p**다. `R0`는 세 방법이 공유한다. 모든 seed/round를 포함했다.

## 1. GC10에서 관찰된 것

R0 평균 18.69%에서 35% 이미지 예산(R5; 561/1,604장)에 Random 24.41±0.54%, DINO-F518 26.72±1.60%, Entropy 24.32±0.61%로 증가했다. 사전 정의 곡선 요약 nAULC는 각각 21.85±0.44%, 23.69±0.33%, 21.73±0.41%다. DINO는 R1–R5의 모든 예산에서 Random과 Entropy보다 **평균** 성능이 높았다. 평균값은 `02_FINAL_NUMERIC_RESULTS.csv`와 `03_FINAL_NAULC_RESULTS.csv`에, 각 seed의 값과 run ID는 `RAW_SEED_METRICS.csv`에 있다.

## 2. NEU에서 관찰된 것

R0 평균 28.22%에서 R5(440/1,258장)에 Random 33.72±1.11%, DINO-F518 33.62±1.30%, Entropy 33.71±1.13%였다. nAULC는 각각 31.84±0.81%, 31.92±0.54%, 31.76±0.75%다. 방법 순위가 예산에 따라 바뀌었고, R5 평균에서 DINO가 Random보다 0.10%p 낮다. 따라서 NEU에서 DINO의 일관된 우위를 주장할 수 없다.

## 3. DINO vs Random

GC10의 seed-paired nAULC 차이는 **+1.832%p**, 5/5 seed에서 양수였다. R5 평균 mAP 차이는 **+2.308%p**다. NEU의 paired nAULC 차이는 **+0.086%p**, 3/5 seed에서 양수였고, R5 평균 차이는 **−0.103%p**다. 두 데이터셋을 하나의 n=10 효과로 합치지 않았다.

## 4. DINO vs Entropy

GC10의 paired nAULC 차이는 **+1.952%p**, 5/5 seed에서 양수이며 R5 평균 차이는 **+2.400%p**다. NEU는 paired nAULC **+0.159%p**, 4/5 seed에서 양수지만 R5 평균 차이는 **−0.091%p**다. 작은 NEU 차이를 실질적 우월성으로 해석하지 않는다.

## 5. Entropy vs Random

GC10의 paired nAULC는 **−0.120%p**(2승/3패), NEU는 **−0.073%p**(2승/3패)다. R5 평균 차이는 GC10 −0.091%p, NEU −0.012%p다. 이 대비는 v2 계획에서 secondary descriptive이므로 우월성 검정의 핵심 주장에 쓰지 않는다.

## 6. 라벨 예산과 방법 간 차이

GC10의 DINO−Random 평균 round 차이는 R0 0, R1 +1.73, R2 +1.75, R3 +1.46, R4 +3.08, R5 +2.31%p다. 차이가 모든 예산에서 일정하게 커지는 단조 패턴은 아니다. NEU는 0, −0.54, +0.44, +1.11, −0.52, −0.10%p로 부호가 바뀐다. `04_ROUNDWISE_RESULTS.csv`와 부록의 paired 차이 그림이 seed별 변동도 보여준다.

## 7. Full-data reference와 절대 gap

GC10 Full-data 1회는 27.36%; R5 평균과의 `Full − R5` gap은 Random 2.95, DINO 0.65, Entropy 3.04%p다. NEU Full-data 1회는 38.51%; gap은 각각 4.79, 4.89, 4.80%p다. Full-data는 각 데이터셋에 **1회**만 있으므로 SD/CI 또는 paired 유의성 비교를 붙이지 않는다. “35% 라벨로 full-data 성능의 X%를 달성” 같은 비율 문장은 본문 핵심 표현으로 사용하지 않는다.

## 8. 클래스별 효과의 균일성

GC10 R5에서 DINO−Random 평균 class AP 차이는 `welding_line` **+12.81%p**인 반면 `waist_folding` **−2.20%p**다. DINO의 최저 클래스는 `inclusion` 3.23%이고 Random 3.13%, Entropy 3.51%와 모두 낮다. NEU R5에서 DINO−Random은 `rolled-in_scale` **−4.79%p**, `scratches` +1.71%p이며 최저 클래스 `crazing`은 DINO 8.93%, Random 9.16%, Entropy 11.31%다. classwise 전체 10/6클래스를 `05_CLASSWISE_RESULTS.csv`에 포함했고, 5 seed의 paired 원자료는 `R5_PAIRED_CLASS_DIFFERENCES.csv`에 있다. 특정 클래스만 골라 전체 방법의 우월성을 주장하지 않는다.

실제 최종 annotation overlay 라벨의 선택 구성에서도 R5 GC10 `welding_line`이 포함된 누적 이미지 평균은 Random **122.4장**, DINO **205.4장**, Entropy **90.6장**이다. GC10 `waist_folding`은 **36.4/17.0/12.0장**이다. NEU `rolled-in_scale`은 **74.4/32.8/76.6장**이다. 이 기술적 연관은 각 클래스 AP 차이를 **설명할 가능성**을 제기하지만 선택 구성만으로 원인이나 기여량을 분리할 수 없다. 이미지 incidence와 bbox incidence, 신규·누적 선택 수는 `SELECTION_COMPOSITION.csv`와 `SELECTION_COMPOSITION_R5_SUMMARY.csv`에 있다. 최종 overlay는 train 2,862장 각각의 라벨 파일을 포함하며 실제 비어 있는 파일은 객체 수 0으로 취급했다.

## 9. 데이터셋 간 결과 차이

GC10에서 DINO의 평균 곡선 이점이 뚜렷하게 관찰되지만 NEU에서는 평균 차이가 거의 없고 예산별 부호도 바뀐다. 데이터셋별 클래스 체계·영상 크기·결함 양상·split이 다르므로 이 차이를 하나의 원인으로 환원하지 않는다. 두 데이터셋의 absolute AP 수준을 서로 같은 난이도의 척도로 해석하지 않는다.

## 10. 현재 결과가 지지하는 주장

**관찰된 사실:** GC10 development에서 DINO-F518의 평균 nAULC와 R5 AP가 Random/Entropy보다 높고 5 seed의 nAULC paired 방향이 모두 양수였다. NEU에서 세 방법의 평균 nAULC는 근접했다. 클래스별 이득은 균일하지 않았다.

**통계적으로 지지되는 주장:** 사전 지정한 네 대비의 exact 양측 sign-flip raw p는 GC10의 두 대비 모두 0.0625, Holm p는 0.25다. NEU 두 대비도 유의하지 않다. 따라서 현 자료는 **p<0.05의 우월성 주장**을 지지하지 않는다. 더구나 이는 development의 탐색적 계산이며 사전 정의한 test 1차 결과가 아니다.

## 11. 현재 결과로 지지할 수 없는 주장

“DINO가 두 데이터셋에서 통계적으로 유의하게 우수하다”, “NEU의 차이는 F518 확대 때문이다”, “라벨링 인건비를 65% 줄였다”, “Full-data와 동등하다”, “test 일반화 성능을 검증했다”는 문장은 현 증거로 성립하지 않는다. NEU에서 F224와 F518을 동일 조건으로 직접 비교한 clean split v2 실험이 없다.

## 12. 한계

각 데이터셋의 반복은 acquisition seed 5개이며 split과 learner seed가 고정돼 있다. 따라서 학습 randomness·split 민감도·독립 데이터셋에 대한 변동을 별도로 추정하지 못한다. Full-data는 1회, test는 withheld다. x축은 라벨 이미지 수이지 실제 bbox 주석 시간/비용이 아니다. n=5에서 양측 exact sign-flip의 최소 raw p는 0.0625이므로 검정 해상도가 제한된다. Bootstrap 구간은 작은 표본의 탐색적 불확실성 요약이다.

## 13. 후속연구

사전 봉인 절차에 따라 **별도 test release 후** 동일 best checkpoint를 한 번씩 평가하고 v2 SAP의 test nAULC 분석을 수행한다. NEU의 F518 입력 민감성은 필요하다면 독립된 동일 조건 F224/F518 ablation으로 검증한다. 실제 annotation 시간이나 박스별 비용을 측정한다면 이미지 수 축과 별도로 비용–성능 곡선을 만들 수 있다. 이는 현재 결과를 수정하는 분석이 아니라 후속 실험 주제다.
