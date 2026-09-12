"""Create experiment records from the audited local numerical outputs."""
from pathlib import Path
import json,csv,ast,hashlib,shutil,html
R=Path(__file__).resolve().parent;T=R/'tables';W=R.parent
s=json.loads((R/'analysis_summary.json').read_text());E='Evidence: Development validation; Statistical unit: acquisition seed, n=10; A/B averaged within acquisition seed; Locked-final: not used'
title='소량 라벨 환경에서 특징 기반 능동학습을 활용한 금속 표면 결함 객체 탐지'
def read(name):return list(csv.DictReader((T/(name+'.csv')).open(encoding='utf-8-sig')))
def put(name,text):(R/name).write_text(text.strip()+'\n',encoding='utf-8')
def comparison_table(ds):
    rows=['|비교|평균 차이(pp)|중앙값(pp)|95% bootstrap CI(pp)|양/음 seed|','|---|---:|---:|---|---:|']
    for r in s['paired_effects']:
        if r['dataset']==ds:rows.append(f"|{r['comparison'].replace(' - ',' − ')}|{r['mean_difference_pp']:+.3f}|{r['median_difference_pp']:+.3f}|[{r['ci95_low_pp']:+.3f}, {r['ci95_high_pp']:+.3f}]|{r['positive_seeds']}/{r['negative_seeds']}|")
    return '\n'.join(rows)
fullrows=['|Dataset|방법|R5 mAP (%)|Full 평균 (%)|R5−Full (pp)|Full 대비 (%)|라벨 이미지/풀|','|---|---|---:|---:|---:|---:|---|']
for r in s['r5_full']:fullrows.append(f"|{r['dataset']}|{r['method']}|{r['R5_map50_95_pct']:.3f}|{r['full_map50_95_pct']:.3f}|{r['R5_minus_full_pp']:+.3f}|{r['percent_of_full_reference']:.2f}|{r['R5_images']}/{r['full_images']}|")
fulltable='\n'.join(fullrows)
put('EXPERIMENT_RESULT_BRIEF.md',f'''
# 실험 결과 기록 — 최종 development 통합분석

2026-09-13 · 로컬 파일 분석 · 새 모델 실행 없음

## 1. 현재 논문 제목

**「{title}」**

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

{E}

nAULC 평균은 Random **27.002%**, DINO **27.393%**, Entropy **27.649%**다. 이는 R0에서 R5까지 라벨 비율에 대해 정규화한 곡선 면적이며 R5 단일 점수가 아니다.

{comparison_table('GC10-DET')}

DINO와 Entropy 모두 Random보다 양의 평균 차이가 관찰된다. 그러나 DINO−Entropy는 seed별 승패가 5/5로 나뉘고 구간이 0을 포함한다. GC10의 점추정 순위는 Entropy > DINO > Random이지만 Entropy가 일반적으로 우월하다고 결론내릴 근거는 부족하다.

초기 R1에서 DINO−Random은 +0.864pp, Entropy−Random은 +1.268pp로, Entropy의 초기 이득이 더 컸다. 반면 R4에서는 각각 −0.305pp, −0.026pp여서 Random 평균이 앞선다. R5의 DINO−Entropy는 −0.008pp로 거의 같지만, 중간 round 차이가 단조롭게 줄어들지는 않는다. R5만 보면 초기 및 중간 구간의 차이를 놓치게 된다.

## 7. NEU 결과

{E}

nAULC 평균은 Random **35.189%**, DINO **35.981%**, Entropy **35.902%**다.

{comparison_table('NEU-DET')}

두 방법의 Random 대비 평균 이득이 GC10과 같은 방향으로 관찰된다. DINO−Entropy 평균은 +0.079pp로 GC10과 부호가 반대지만, 역시 양/음 seed가 5/5이고 구간이 0을 포함한다. 특징 기반 선택과 confidence 합 기반 선택의 안정적인 우열을 확정하기 어렵다.

DINO−Random 이득은 R2에서 +1.412pp로 가장 컸고 R5에서는 +0.303pp였다. Entropy−Random도 R2에서 +1.552pp로 가장 컸다. DINO−Entropy는 R1 +0.177pp, R2 −0.140pp, R3 +0.374pp, R4 +0.078pp, R5 −0.189pp다. 같은 데이터셋 안에서도 예산별 평균 순위가 바뀌므로 한 줄의 엄격한 순위로 압축하지 않는다.

## 8. Full-data reference와의 거리

{E} · Full-data는 dataset별 learner 2개의 산술평균이며 n=10으로 복제하지 않는다.

{fulltable}

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
''')
put('FINAL_DEVELOPMENT_INTERPRETATION.md',f'''
# Development 통합 결과의 해석

「{title}」

{E}

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
''')
put('FIGURE_CAPTIONS.md',f'''
# 대표 그림 설명

논문 제목: 「{title}」

1. **Research workflow.** Sealed pool과 공통 R0에서 Random, frozen DINOv2 cosine farthest-first, NMS detection-confidence binary entropy SUM을 비교한다. Entropy parent는 직전 learner A이며 detector는 모두 cold start한다. R0의 물리적 재학습은 없다. Full-data는 별도 A/B reference다.
2. **Three-way development curves.** {E}. 선은 seed 평균, 음영은 seed SD다. SD는 평균의 CI가 아니다. 각 cell은 100-epoch CSV의 최고 mAP50–95를 동일하게 사용한다. validation 선택 편향 및 GPU/OS 차이로 test 성능을 대신하지 못한다.
3. **Paired nAULC.** {E}. 회색 점은 10 seed, 마름모는 평균, 막대는 100,000회 paired percentile bootstrap CI. 두 endpoint 점에 0.1, 내부 네 점에 0.2의 가중치가 적용된다. 구간이 0을 포함하는 것과 동등성은 다르다.
4. **R5 versus full-label reference.** {E}. AL의 점은 10개 A/B 평균이고 Full의 ×는 실제 learner 두 개다. Full을 n=10으로 복제하거나 upper bound로 취급하지 않는다. 약 35%는 이미지 수 비율이며 annotation 비용 비율이 아니다.
5. **Selection behavior.** Evidence: acquisition-only diagnostics; 10 acquisition trajectories; 5 dependent rounds each; locked-final not used. 위: 각 query의 선택/현재 후보 class prevalence 차이(pp)를 seed·round에 걸쳐 평균한다. 가운데: 기존 branch의 class pairwise cosine distance와 R5 DINO−Random 누적 class 선택 수. 작을수록 compact하며, numeric class ID가 다른 NEU 자료는 class name으로 대조했다. 아래: saved entropy SUM과 n_detections의 query 내부 Spearman; 점은 각 seed, 선은 해당 round 중앙값. 라벨을 보고 새 선택을 만들지 않았으며, 표본 구성과 성능의 인과관계는 검증하지 않았다.

공통 제한: 모든 AL 통계 표본 수는 training 횟수가 아닌 acquisition seed n=10이다. GC10/NEU는 별도로 분석하며 seed ID가 같다는 이유로 dataset 사이를 짝지어 추론하지 않는다. 원래 locked-final 결과는 별도 표에만 둔다.
''')
# Keep the old endpoint in a physically separate reference table.
old=W/'runpod_final_20260913/agent_upload/historical_440_sources/confirmatory_v1_1/final_analysis/results/primary_summary_2.csv'
shutil.copyfile(old,T/'historical_locked_final_reference.csv')
put('tables/historical_locked_final_reference.md','''# Historical locked-final reference

Evidence: Historical locked-final; acquisition seed n=10; original primary analysis retained.

|Dataset|Original DINO−Random nAULC (pp)|
|---|---:|
|GC10-DET|+0.581153|
|NEU-DET|+0.578156|

This is an existing result, not a new evaluation. Do not merge this endpoint with the current development comparisons. Complete original fields remain in the adjacent CSV.
''')
source=W/'runpod_final_20260913/agent_upload/historical_440_sources/confirmatory_v1_1/final_analysis/src/analyze_final_results.py'
tree=ast.parse(source.read_text(encoding='utf-8-sig'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'trapezoid_aulc','percentile_ci'}]
put('frozen_numerical_functions.py','from __future__ import annotations\nimport numpy as np\nfrom typing import Any\n\n'+ '\n\n'.join(ast.unparse(n) for n in nodes))
put('audit/numerical_source.json',json.dumps({'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'functions':[n.name for n in nodes],'extraction':'AST-only pure numerical functions; historical evaluator never imported','CI':'PCG64 percentile bootstrap 100000 paired seed draws','bootstrap_seeds':{'GC10-DET':421397045,'NEU-DET':1672358496}},indent=2))
put('README.md',f'''
# 최종 development 실험 기록

「{title}」

먼저 `EXPERIMENT_RESULT_BRIEF.md`를 읽고, 직접적인 판단은 `FINAL_DEVELOPMENT_INTERPRETATION.md`, 검증은 `ANALYSIS_VALIDATION_REPORT.md`에서 확인한다. `index.html`은 로컬 그림·문서 탐색용이며 외부 통신을 하지 않는다.

{E}. 전체 비교 등급 B: endpoint 정합성 A, hardware/OS 차이는 제한으로 남김. nAULC는 mAP 비율을 라벨 비율의 R0–R5 구간에서 정규화한 면적이다. `*_pp`는 percentage points, `*_pct`는 100을 곱한 %, 나머지 mAP/nAULC는 0–1 비율이다.

- `tables/`: 요구한 8개 대표 표와 감사·seed·query 보조 표, 각각 CSV/Markdown. Full-data 단위는 실제 learner 2개.
- `figures/`: 대표 그림 5종 × PNG/SVG/PDF. `FIGURE_CAPTIONS.md`에 정의와 제한.
- `audit/`: 분석에 읽은 파일 SHA-256, 사전 자료구조 판정, 검증 결과. 읽지 않은 원본까지 보호 hash를 검증했다고 주장하지 않는다.
- `historical_geometry/`: `research/entropy-a00-audit-20260908`의 read-only blob 5개와 commit/blob/SHA provenance. 새 특징 추출 없음.
- `frozen_numerical_functions.py`: 기존 소스에서 순수 수치 함수만 AST로 추출; 원본 evaluator 실행 금지.

`verify_analysis.py`는 출력 CSV로 집계와 bootstrap을 독립 대조하고 원래 읽은 입력의 hash를 확인한다. `analyze.py`는 로컬 원본이 갖춰진 경우에만 전체 재현하며, 네트워크/child process를 audit hook으로 차단한다. `plot_figures.py`는 완성 CSV만 읽는다. `export_historical.py`는 로컬 git의 정확한 blob 5개만 읽고 checkout/push를 하지 않는다.

입력은 `runpod_final_20260913/agent_upload/`의 동결 파일과 기존 로컬 acquisition-pool label allowlist다. 원본 ZIP SHA-256은 `d3512c9a2bfea5537e5e83090e2a19d848cec1d30e211aa70c379fbabb40cd7d`이며 이 분석 ZIP을 raw artifact 백업으로 오인하지 않는다. 새 전달 ZIP에는 분석 결과와 재현 코드가 있으며, raw `results.csv`부터의 완전 재현에는 기존 agent-upload 원본과 hash로 특정한 acquisition label metadata가 필요하다. 이 폴더의 `acquisition_pool_class_membership.csv`는 class composition 산술을 다시 계산할 수 있게 한다.

원본 checkpoint bytes, 이미지, validation/final 라벨은 이 분석 패키지에 넣지 않는다. 기존 n=9 intermediate 출력은 대표 결과로 사용하지 않았다. Historical locked-final은 `historical_locked_final_reference.*`에만 별도로 남겼다. 새 evaluation authorization은 없으며 training/acquisition/inference/RunPod/GitHub push는 모두 수행하지 않았다.
''')
cards=''.join(f'<section><h2>Figure {i}</h2><a href="figures/{n}.png"><img src="figures/{n}.png" alt="Figure {i}"></a><p><a href="figures/{n}.svg">SVG</a> · <a href="figures/{n}.pdf">PDF</a></p></section>' for i,n in enumerate(['figure_1_research_workflow','figure_2_development_curves','figure_3_paired_nAULC','figure_4_full_label_reference','figure_5_selection_behavior'],1))
put('index.html',f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>최종 development 실험 기록</title><style>body{{font:17px/1.65 system-ui,sans-serif;background:#f2f5f9;color:#243347;margin:0}}main{{max-width:1150px;margin:auto;padding:40px 28px}}h1{{font-size:30px}}.badge{{background:#e0ecfb;padding:12px 20px;border-radius:9px}}section{{background:white;padding:20px;border-radius:14px;margin:28px 0}}img{{width:100%;height:auto}}a{{color:#1257a0}}small{{color:#556}}</style><main><h1>{title}</h1><p>2026-09-13 · 최종 development 실험 기록</p><div class="badge">자료구조 PASS · Endpoint A / 전체 비교 B · 644 physical cells · 10 acquisition seeds</div><p><a href="EXPERIMENT_RESULT_BRIEF.md">실험 결과 기록</a> · <a href="FINAL_DEVELOPMENT_INTERPRETATION.md">질문별 판단</a> · <a href="ANALYSIS_VALIDATION_REPORT.md">검증 보고</a> · <a href="FIGURE_CAPTIONS.md">그림 설명</a></p><p>DINO−Random ΔnAULC: GC10 +0.391pp / NEU +0.792pp.<br>DINO−Entropy 직접 차이는 두 데이터셋 모두 CI가 0을 포함하며 동등성을 확정하지 않는다.</p><small>{E}</small>{cards}<p>Full-data는 2-learner reference. 현재 결과는 새 locked-final 평가가 아니다.</p></main></html>''')
print('Experiment records and local figure index created.')
