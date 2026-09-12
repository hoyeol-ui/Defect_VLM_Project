# 실험 결과 기록 — 최종 development 통합분석

2026-09-13 · 로컬 파일 분석 · 새 모델 실행 없음

## 1. 현재 논문 제목

**「소량 라벨 환경에서 특징 기반 능동학습을 활용한 금속 표면 결함 객체 탐지」**

이 문서는 내가 수행한 실험의 범위, 수치, 해석과 남은 판단을 기록한다. 이번 분석은 완성된 학습 기록을 같은 development validation 기준으로 통합한 것이다. 새 locked-final 성능을 측정한 결과가 아니다.

## 2. 연구에서 확인하려는 질문

금속 표면 결함 객체 탐지에서 라벨을 붙일 이미지 수가 제한될 때, 사전학습 특징의 다양성을 따라 고르는 방법이 무작위 선택보다 효율적인가를 확인한다. 선택의 정보원을 바꾸면 어떤 표본이 모이고 탐지 성능이 어떻게 달라지는지도 함께 본다.

Random은 균등 무작위 선택의 기준이다. DINO는 frozen DINOv2-small 특징과 cosine distance 기반 farthest-first를 사용한다. Entropy는 직전 round learner A의 best checkpoint가 낸 NMS 검출 confidence의 binary entropy를 이미지별로 합한다. 이 세 방법은 서로 다른 질문을 다루므로, 평균 점수의 순위만으로 선택 과정이 같은지 판단하지 않는다.

## 3. 기존 440회 실험에서 확인한 것

기존 Random/DINO 실험은 두 dataset, acquisition seed a00–a09, learner A/B, R0–R5로 구성되었다. 공통 R0를 물리적으로 한 번씩만 학습해 총 440회이며, 두 방법의 곡선 표현은 480개 conceptual 행이다. 각 seed에서 A/B를 평균한 뒤 seed 10개를 통계 단위로 쓴다.

**Evidence: Historical locked-final — 아래 두 수치는 이번 development 표와 분리한다.**

기존 확정 평가에서 DINO−Random nAULC는 GC10 +0.581pp, NEU +0.578pp였다. 이 값은 과거 이미 수행한 locked-final 평가에 속한다. 이번 분석의 GC10 +0.391pp, NEU +0.792pp와 다르다고 해서 기존 결과가 바뀐 것이 아니다. endpoint와 metric 추출이 다르다. 원래 final 결과는 그대로 보존한다.

## 4. Entropy를 추가한 이유

Random보다 낫다는 결과만으로 특징 기반 선택이 detector uncertainty 기반 선택과 비교해 어떤 위치에 있는지 알 수 없다. Entropy를 추가한 목적은 서로 다른 선택 신호를 같은 데이터·seed·budget·learner 조건에서 비교하기 위해서다. 결과가 favorable한 seed만 남기거나 점수 정의를 바꾸는 추가 탐색은 하지 않았다.

Entropy는 detection-confidence binary entropy **SUM**이며 클래스 확률 전체의 entropy와 다르다. conf=.25, IoU=.70, max_det=300, 빈 검출의 점수 0을 유지했다. acquisition에 쓰는 parent는 직전 learner A이지만, 각 새 detector training은 동일 pretrained YOLOv8n에서 cold start한다. learner B도 같은 선택 목록을 학습하며 A/B의 역할 비대칭을 공개한다.

## 5. 완료범위와 비교 가능성

Entropy 200 training/100 acquisition, Full-data 4 training이 완료되어 physical training 입력은 644개다. 추가 비교에는 기존 R0 40개를 참조한다. 세 방법의 conceptual AL 행은 720개, A/B 평균 뒤 seed×round 행은 360개, seed별 nAULC는 60개다. 과거 4회 remote 속도 확인 실행은 440에 합산하지 않았다.

GC10 pool 1,605장·validation 229장, NEU pool 1,259장·validation 180장의 ID가 방법 간 일치한다. train/validation overlap은 없고, 공통 R0 checkpoint/manifest/args의 identity도 확인했다. numeric seed와 경로를 제외한 학습 인자는 일치했다. 100 epochs, imgsz 640, batch 8, SGD, resume false 조건을 유지했다.

**동일 endpoint 자료구조는 A등급, 전체 해석은 B등급의 조건부 development 비교다.** 과거 로컬과 추가 원격 실험의 GPU/OS가 다르고, 비교에 참조된 과거 feasibility metadata 4개에는 package 정보가 완전하지 않다. 기록이 있는 torch/Ultralytics/numpy/pillow 버전은 일치한다. 이 설계가 hardware 효과를 별도로 분리하거나 bitwise 재현성을 보장하지는 않는다.

모든 방법에 대해 각 100-epoch `results.csv`의 최대 validation mAP50–95를 같은 방식으로 뽑았다. 과거 별도 best-checkpoint validation 평가값을 새 CSV 최대값과 섞지 않았다. validation은 checkpoint 선택 및 Entropy learner 경로에 관여하므로 독립 test처럼 해석하면 안 된다.

## 6. GC10 결과

Evidence: Development validation; Statistical unit: acquisition seed, n=10; A/B averaged within acquisition seed; Locked-final: not used

nAULC 평균은 Random **27.002%**, DINO **27.393%**, Entropy **27.649%**다. 이는 R0에서 R5까지 라벨 비율에 대해 정규화한 곡선 면적이며 R5 단일 점수가 아니다.

|비교|평균 차이(pp)|중앙값(pp)|95% bootstrap CI(pp)|양/음 seed|
|---|---:|---:|---|---:|
|DINO − Random|+0.391|+0.252|[+0.034, +0.796]|8/2|
|Entropy − Random|+0.647|+0.684|[+0.130, +1.164]|7/3|
|DINO − Entropy|-0.256|-0.164|[-0.808, +0.270]|5/5|

DINO와 Entropy 모두 Random보다 양의 평균 차이가 관찰된다. 그러나 DINO−Entropy는 seed별 승패가 5/5로 나뉘고 구간이 0을 포함한다. GC10의 점추정 순위는 Entropy > DINO > Random이지만 Entropy가 일반적으로 우월하다고 결론내릴 근거는 부족하다.

초기 R1에서 DINO−Random은 +0.864pp, Entropy−Random은 +1.268pp로, Entropy의 초기 이득이 더 컸다. 반면 R4에서는 각각 −0.305pp, −0.026pp여서 Random 평균이 앞선다. R5의 DINO−Entropy는 −0.008pp로 거의 같지만, 중간 round 차이가 단조롭게 줄어들지는 않는다. R5만 보면 초기 및 중간 구간의 차이를 놓치게 된다.

## 7. NEU 결과

Evidence: Development validation; Statistical unit: acquisition seed, n=10; A/B averaged within acquisition seed; Locked-final: not used

nAULC 평균은 Random **35.189%**, DINO **35.981%**, Entropy **35.902%**다.

|비교|평균 차이(pp)|중앙값(pp)|95% bootstrap CI(pp)|양/음 seed|
|---|---:|---:|---|---:|
|DINO − Random|+0.792|+0.702|[+0.344, +1.235]|9/1|
|Entropy − Random|+0.714|+0.710|[+0.410, +1.038]|9/1|
|DINO − Entropy|+0.079|+0.038|[-0.283, +0.447]|5/5|

두 방법의 Random 대비 평균 이득이 GC10과 같은 방향으로 관찰된다. DINO−Entropy 평균은 +0.079pp로 GC10과 부호가 반대지만, 역시 양/음 seed가 5/5이고 구간이 0을 포함한다. 특징 기반 선택과 confidence 합 기반 선택의 안정적인 우열을 확정하기 어렵다.

DINO−Random 이득은 R2에서 +1.412pp로 가장 컸고 R5에서는 +0.303pp였다. Entropy−Random도 R2에서 +1.552pp로 가장 컸다. DINO−Entropy는 R1 +0.177pp, R2 −0.140pp, R3 +0.374pp, R4 +0.078pp, R5 −0.189pp다. 같은 데이터셋 안에서도 예산별 평균 순위가 바뀌므로 한 줄의 엄격한 순위로 압축하지 않는다.

## 8. Full-data reference와의 거리

Evidence: Development validation; Statistical unit: acquisition seed, n=10; A/B averaged within acquisition seed; Locked-final: not used · Full-data는 dataset별 learner 2개의 산술평균이며 n=10으로 복제하지 않는다.

|Dataset|방법|R5 mAP (%)|Full 평균 (%)|R5−Full (pp)|Full 대비 (%)|라벨 이미지/풀|
|---|---|---:|---:|---:|---:|---|
|GC10-DET|Random|29.756|34.717|-4.961|85.71|561/1605|
|GC10-DET|DINO|30.332|34.717|-4.385|87.37|561/1605|
|GC10-DET|Entropy|30.340|34.717|-4.377|87.39|561/1605|
|NEU-DET|Random|38.508|42.895|-4.387|89.77|441/1259|
|NEU-DET|DINO|38.811|42.895|-4.084|90.48|441/1259|
|NEU-DET|Entropy|39.000|42.895|-3.895|90.92|441/1259|

GC10 Full A/B는 34.922%/34.512%, 평균 34.717%다. NEU는 42.912%/42.879%, 평균 42.896%다. 약 35%의 학습 이미지를 사용한 DINO·Entropy는 full-label reference의 GC10 약 87.4%, NEU 약 90.5–90.9%에 도달했다. Full과의 절대 차이는 약 3.90–4.38pp 남는다.

이 결과는 소량 이미지 라벨의 성능 수준을 설명하지만 “전체 성능을 회복했다”, “성능 손실이 없다”, “65% annotation 비용을 절감했다”는 뜻은 아니다. 이미지별 객체 수와 주석 난도가 다르고, full 기준도 learner 2개뿐이다. Full-data는 statistical upper bound가 아닌 **full-label reference**다.

## 9. 기존 결론은 어떻게 달라졌는가

Random 대비 특징 선택의 label efficiency 이득이라는 기존 방향은 유지된다. 이제 uncertainty 비교군을 붙이면 DINO의 독점적 우월성보다, **다른 선택 신호가 비슷한 평균 label efficiency에 도달할 수 있으면서 서로 다른 표본구성을 만든다**는 해석이 더 정확하다. 시나리오 B의 기술적 형태에 가장 가깝고, 데이터셋별 점추정 순위가 바뀌는 D와 seed/budget 의존성 E가 함께 있다. 여기서 “비슷하다”는 formal equivalence 판정이 아니다.

선택 진단은 10-seed 전체로 다시 계산했다. Entropy score와 detection count의 query 내부 Spearman 중앙값은 GC10 **0.918**(범위 0.878–0.958), NEU **0.945**(0.882–0.977)다. GC10 선택 이미지 평균 검출 수는 R1 5.06 대 비선택 1.31, R5 2.99 대 1.10이다. NEU는 R1 7.84 대 2.52, R5 4.82 대 2.02다. 합산 점수와 객체 검출 수의 강한 연관성을 보여주지만, uncertainty가 아무 정보도 없다는 증거나 합산 규칙을 사후 교체할 근거로 쓰지 않는다.

현재 candidate 대비 새 query의 클래스 포함률을 보면 GC10 DINO는 welding_line +14.77pp, Entropy는 +21.70pp이며, punching_hole은 각각 +4.44pp와 +19.07pp다. DINO는 crease +4.40pp, Entropy는 −0.13pp다. NEU의 pitted_surface는 DINO +13.27pp, Entropy −16.55pp로 선택 방향이 반대이고, inclusion은 +2.08pp와 +19.67pp다. 각 이미지는 여러 클래스에 포함될 수 있다.

기존 DINO 클래스 geometry 진단을 읽기 전용으로 재사용했다. 클래스 내부 cosine distance와 R5 DINO−Random 누적 선택 수의 관계는 NEU에서 강하며 GC10은 silk_spot 등 예외가 있다. 현재 geometry 표와 YOLO의 NEU 숫자 class ID 순서가 달라, **dataset+class name으로 연결**했다. 새 feature extraction은 하지 않았다. 클래스별 development AP는 공통 원본이 없어, 표본 비율만으로 특정 클래스 탐지 성능 향상을 주장하지 않는다. seed별 composition–nAULC 상관도 탐색 기록으로만 둔다.

## 10. 현재 논문에 쓸 수 있는 내용

두 금속 표면 결함 데이터셋에서 frozen feature diversity 선택은 Random 대비 평균 development label efficiency가 높았다. confidence entropy 합을 사용한 비교군도 같은 방향의 이득을 보였다. 두 AL 방법의 직접 차이는 seed와 budget에 따라 달랐다. 약 35%의 이미지로 full-label reference의 약 87–91% 성능 수준을 얻었고, 선택 신호에 따른 클래스 구성의 차이를 확인했다.

이 문장은 historical locked-final 결과와 이번 development 확장 결과의 근거 수준을 각각 표시할 때 유효하다. 추가 training 없이 현재 AL 설계를 닫는 판단도 이 좁은 연구질문에서는 합리적이다. 더 유리한 숫자가 나올 가능성 자체는 실험 확대 이유가 아니다.

## 11. 아직 쓰면 안 되는 내용

DINO 또는 Entropy가 모든 budget·seed·class에서 우월하다는 주장, 두 방법의 통계적 동등성, 포괄적 SOTA, class composition의 성능 인과효과, full reference와 동등한 성능, annotation 비용의 확정 절감률을 주장하지 않는다. 두 데이터셋 밖 산업 환경으로 일반화하지 않는다.

새 Entropy/Full-data locked-final은 아직 평가하지 않았다. 기존 Random/DINO locked-final은 이미 알려져 있으므로 후속 평가가 처음부터 test-blind한 3-way 실험인 것처럼 쓰지 않는다. 필요한 다음 단계는 새 학습 확대보다, 기존 final을 유지하면서 신규 204 checkpoint의 평가 범위·동일 endpoint·통계 family를 별도로 고정하고 승인받는 일이다. 이번 기록은 그 실행 권한을 부여하지 않는다.

## 12. 결과 설명에 사용할 핵심 그림

Figure 1은 세 선택 신호와 공통 학습·분석 구조를 설명한다. Figure 2는 두 dataset의 전 예산 곡선과 seed 변동을 보여준다. Figure 3은 10개 paired effect와 평균·bootstrap 구간으로 과도한 우월성 주장을 막는다. Figure 4는 R5의 라벨 수와 full reference 사이의 남은 차이를 보여준다. Figure 5는 선택 구성, DINO geometry, Entropy 검출 수 연관성을 함께 보여준다.

모든 대표 그림은 PNG/SVG/PDF로 저장했다. `tables/`에는 CSV와 Markdown 원표를 함께 남겼으며, `ANALYSIS_VALIDATION_REPORT.md`가 계산·행 수·근거 구분 검증을 기록한다. 연구 목적은 선택 신호의 효과와 한계를 설명하는 것이며, 어느 방법이 이겼는지에 맞춰 protocol을 바꾸는 것이 아니다.
