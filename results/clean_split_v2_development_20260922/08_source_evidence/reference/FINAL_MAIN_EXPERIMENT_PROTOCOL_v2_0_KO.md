# 최종 본실험 설계서 v2.0

2026-09-19 · ID `metal_al_final_v2_20260919` · **NOT READY**

**GC10-DET와 NEU-DET, Random/DINOv2-F518/Entropy, dataset당 acquisition seed 5개, scientific training 전부 신규 162회**로 설계를 교정한다. 과거 56개 재사용과 Entropy 25개만 추가하는 경로는 없다. 기존 440회·Entropy 후속·입력 진단·clean 81회는 설계 근거로만 보존한다. 최신 clean 81회 회수는 이 설계의 실행 gate가 아니다.

이 문서는 새 split과 protocol의 설계다. 새 split 생성·feature extraction·selection·학습·test inference·RunPod 접속·git commit/push는 수행하지 않았다. 학습을 계획한 것과 실행을 승인받은 것은 다르다.

## 1. 목적과 연구 질문

논문 제목은 「소량 라벨 환경에서 특징 기반 능동학습을 활용한 금속 표면 결함 객체 탐지」다. 같은 수의 라벨 이미지를 사용하는 조건에서 사전학습 특징의 다양성 기반 선택이 Random과 명시적 detector uncertainty 선택에 비해 탐지 학습에 어떤 영향을 주는지 묻는다.

- RQ1: 제한된 라벨 환경에서 평균 탐지 학습 성능에 도움이 되는가?
- RQ2: 평균 효과가 모든 결함 클래스에 고르게 나타나는가?
- RQ3: 입력·데이터 구성·선택 데이터의 특성이 그 결과의 해석을 어떻게 제한하는가?

RQ1 바로 다음에 전체 클래스별 RQ2 결과를 배치한다. F518을 새로운 AL 알고리즘으로 제안했다고 주장하지 않는다. 세 RQ에 직접 답하지 않는 새 detector·selector는 추가하지 않는다.

## 2. 과거 문제와 이번 수정

| 과거 관찰 | v2 수정 |
|---|---|
| 기존 GC10 split의 proxy group 공유 | 새 raw audit와 proxy/content 합집합 component로 새 split |
| NEU의 확정 cross-split 내용 중복 | NEU 제외 대신 전체 corpus content graph와 component-disjoint split 설계 |
| C224의 주변부 결함 손실 | 두 데이터셋 모두 F518 전체 시야 정책. GC10 축소와 NEU 확대의 의미 구분 |
| 특징/선택 변화가 탐지 개선을 보장하지 않음 | 전체 R0–R5 탐지 곡선과 전체 클래스 결과를 직접 비교 |
| Entropy SUM의 검출 수 의존성 | normalized class-score entropy MEAN 한 arm으로 새 정의 |
| 평균 향상과 일부 클래스 하락 공존 | 전체 10/6클래스와 worst-class 곡선을 사전 고정 |
| 원자료·실행 로그 부족 | 새 cell identity, source/data/config SHA, 별도 backup·restore gate |
| trainer 내부의 자동 복구 | OOM batch 자동 축소와 NaN recovery를 차단하는 fail-fast 실행 경계 필요 |

이 수정은 과거 성능의 재포장이나 기존 결과 삭제가 아니다. 최종 main의 모집단·분할·수식·seed·학습 계보를 새로 정의한다.

## 3. 데이터와 새 split

GC10 공식 원 논문은 3570개 이미지를 서술하지만 로컬은 2312 raw 경로·2294 XML·과거 canonical 2292다. NEU 공식 구성은 1800장·6범주·200×200이며 로컬 raw 1800장과 과거 canonical 1798장이다. 출처 차이와 정제 후 N을 구분한다. 공식 및 로컬 근거와 전 클래스 통계는 두 DATASET_AUDIT_PLAN 파일과 DATASET_FACTS_v2.json에 있다.

GC10의 기존 canonical에는 인식된 bbox 3563개 외 미정의 class d 객체 1개가 있다. 이를 폴더 class로 추정하지 않는다. NEU의 canonical에는 multi-label 이미지 123장이 있어 filename의 단일 클래스만으로 층화하지 않는다.

**두 데이터셋 모두 새 train/dev/test 목표 비율 70/15/15.** train 규모를 유지하면서 dev-best와 test의 클래스 지원을 균형 있게 확보하려는 설계 선택이다. 실제 component 제약이 우선이다. 70/10/20 및 60/20/20의 장단점은 데이터 계획에 비교했다.

GC10은 exact identity+검증된 filename proxy+content relation의 합집합 연결요소, NEU는 exact identity+확정/보수적으로 판정된 content relation의 연결요소를 hard unit으로 삼는다. 이것은 physical production group을 복원했다는 뜻이 아니다. weak candidate를 무조건 union하거나 genuine edge를 비율 때문에 끊지 않는다.

split seed는 GC10 2026091906, NEU 2026091907. component 단위 multilabel support를 우선하고 사전 고정 image/class incidence 오차를 최소화한다. 목표 ±5%p 범위와 각 클래스 train ≥10 component·10장, dev/test 각각 ≥5 component·5장·5 bbox를 운영 최저 기준으로 둔다. 검정력 보증은 아니다. 불가능하면 audit/설계 검토를 하고 NEU 자동 제외나 image-level split으로 후퇴하지 않는다.

component assignment 후 독립된 cross-split residual audit를 통과해야 한다. 새 test의 과거 데이터 접근 이력을 숨기지 않는다. 지금은 새 canonical/split이 없으므로 **실제 train N과 각 round 이미지 수는 미확정**이다.

## 4. 입력·표현·farthest-first

F518은 full-frame 비율 유지 resize와 중앙 padding이며 crop은 없다. GC10 2048×1000→518×253+padding 51.1583%, NEU 200×200→518×518+padding 0%다. NEU 확대가 새 디테일을 만드는 것은 아니다. RGB pad=(124,116,104), half-up 짧은 변, uint8 bicubic antialias, processor resize/crop off, /255와 ImageNet 정규화를 고정한다.

frozen facebook/dinov2-small revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, weight SHA `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`. FP32 eval/inference_mode, autocast/TF32 off. patch14, 37×37 patch+CLS=1370 tokens, 최종 norm CLS/pooler_output 384D, L2 정규화.

cosine distance는 1−z_i·z_j. 각 미선택 후보에서 현재 labeled reference까지 최소 거리를 계산하고 가장 큰 후보 한 장을 선택해 즉시 reference를 갱신한다. 점곱·거리 비교는 canonical float32 cache를 float64로 읽어 CPU에서 수행하도록 명세한다. tie tolerance 1e-12 안에서는 opaque ID 오름차순이다. GT·class quota·bbox 크기를 선택기에 넣지 않는다. component는 분할과 사후 진단에 사용하며 AL의 선택 단위는 이미지다. component당 한 장이라는 추가 제약은 없다.

이는 [Sener–Savarese Algorithm 1](https://arxiv.org/html/1708.00489v4)의 greedy max-min 구조와 연결된다. 원 논문의 detector/representation, robust MIP, 이론적 보장을 그대로 구현했다는 주장은 하지 않는다. 단위벡터에서는 Euclidean 거리와 단조 관계지만 cosine distance 자체를 triangle-inequality metric이라고 부르지 않는다.

## 5. Random과 Entropy

Random은 정렬된 미선택 opaque ID에서 uniform without replacement다. 새 PCG64 namespace `0x464D5632`, dataset code GC10=1/NEU=2, acquisition seed, round, purpose로 SeedSequence를 구성한다. purpose 0은 shared R0, 1은 Random query다. learner seed를 acquisition RNG에 넣지 않는다. R0에 희소 클래스가 없다고 다시 뽑거나 GT 층화하지 않는다.

Entropy는 **ENTROPY_NCS_MEAN_V2**를 사용한다. post-NMS box의 원 C차원 sigmoid score s를 q_c=s_c/Σs로 정규화하고, h=−Σq ln(q)/ln(C)를 이미지 안에서 평균한다. no detection=0. 기존 binary confidence SUM은 사용하지 않는다. q는 YOLO가 학습한 native softmax posterior가 아니라 명시적 score 변환이다.

imgsz640, batch1 FP32, conf max(s)>.25, class-aware NMS IoU .70/max_det300, augmentation·TF32 off. class vector는 raw anchor index로 추적한다. GPU/CPU NMS의 동점 차이와 backend 변경을 피하기 위한 고정 CPU NMS 규칙을 ENTROPY_BASELINE_DECISION.md에 명세했다. U 내림차순·정확 동점 opaque ID 오름차순이다.

R1은 같은 dataset·seed의 새 shared R0 dev-best, R2–R5는 이전 Entropy round의 dev-best로 scoring한다. Random/DINO와 달리 learner와 dev에 의존하는 것이 이 방법의 구조다. classification confusion을 다루며 localization uncertainty·missed object·모든 uncertainty를 대표하지 않는다. MEAN이 detection-count 상관을 0으로 만든다고 보장하지 않는다.

## 6. seed·R0–R5·Full reference

각 dataset acquisition seed는 **2026091901–2026091905**, learner seed는 **2026091900** 하나로 고정한다. 데이터셋 namespace가 다르므로 같은 seed 숫자라도 같은 이미지 선택을 뜻하지 않는다. 이전 seed나 선택 목록을 복사하지 않는다.

**실제 train N을 확정한 뒤** 각 누적 수를 독립적으로 half-up 반올림한다.

`n_r = (N*(10+5*r)+50)//100, r=0..5`

`q_r = n_r−n_(r−1), r=1..5`

따라서 매 query가 항상 같은 정수라는 가정을 하지 않는다. n0<…<n5≤N, 각 query 양수, 세 방법의 seed별 R0 동일성을 검사한다. 각 round를 별도 percentage에서 반올림하는 v2 규칙은 과거 고정 query 규칙과 구별한다.

| 데이터셋 | 실제 v2 N·counts | 과거 canonical에서 단순 70% 목표를 적용한 비실행 예시 |
|---|---|---|
| GC10 | TBD_AFTER_NEW_SPLIT_AUDIT | N=1604일 때 160/241/321/401/481/561, query 81/80/80/80/80 |
| NEU | TBD_AFTER_NEW_SPLIT_AUDIT | N=1259일 때 126/189/252/315/378/441, query 63/63/63/63/63 |

각 dataset Full reference는 새 train 전체 N으로 같은 learner seed에서 **신규 1회**다. 35% 이미지의 성능 위치를 보여주는 조건부 참고값이다. n=1 SD·엄밀한 upper bound·65% 라벨링 비용 절감 주장은 하지 않는다.

## 7. YOLO 통제 조건과 실제 소스 감사

YOLOv8n, initial pretrained SHA `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`. 이전 연구의 학습된 checkpoint는 초기화나 parent에 사용하지 않는다. 각 round는 이 공통 공개 pretrained에서 cold start하고, 새 dataset head 초기화 전에 seed를 설정한다.

epochs100, patience100, batch8, imgsz640, SGD lr0=.01/lrf=.01, momentum=.937, weight_decay=.0005, nbs64, warmup3/.8/.1, cos_lr=False, AMP=True, deterministic=True, workers4, cache=False, rect=False, single_cls=False, fraction1, multi_scale=0.0. augmentation은 hsv .015/.7/.4, degrees0, translate.1, scale.5, shear0, perspective0, flipud0, fliplr.5, bgr0, mosaic1, close_mosaic10, mixup/cutmix/copy_paste0. box/cls/dfl loss weights 7.5/.5/1.5를 포함해 resolved kwargs 전체를 저장한다.

목표 소스는 읽어 확인한 Ultralytics 8.4.90이며 Python/PyTorch/torchvision/CUDA·GPU·source SHA까지 실행 전 봉인한다. backend가 달라지면 같은 프로토콜이라고 간주하지 않는다. CPU에서 읽은 버전만으로 목표 RunPod 준비를 PASS 처리하지 않는다.

**이번 소스 감사에서 확인한 중요한 동작:**

- metrics.fitness 가중치는 [P,R,mAP50,mAP50–95]=[0,0,0,1]이다. dev mAP50–95로 best를 고른다.
- trainer는 best fitness와 동률인 epoch에도 best.pt를 저장한다. 따라서 마지막 최고점 epoch를 고르는 규칙을 명시한다.
- training 중 validation loader는 현 코드상 batch8의 2배인 16을 사용한다. 학습 종료 후 봉인 dev 및 test 평가는 batch8이다. 이 차이를 숨기지 않고 모든 방법에 동일하게 적용한다.
- trainer 내부에 첫 epoch OOM 시 batch 자동 절반 축소와 NaN recovery가 있다. 바깥 runner의 automatic_retry=False만으로 막히지 않는다. **새 실행 adapter의 fail-fast guard와 fault-injection 검증이 필수**다. 자동 batch 변경·숨은 recovery가 발생한 attempt는 COMPLETE로 채택하지 않는다.

val은 dev만 사용하고 test 경로 fallback을 허용하지 않는다. best/last뿐 아니라 원래 training 중 복구 가능한 optimizer/scheduler/scaler/RNG 상태를 별도 보존한다. source·kwargs·loss/evaluation defaults와 guard 구현은 pre-flight에서 실제로 확인해야 한다.

## 8. train/dev/test 접근 경계

curator는 split·annotation integrity만 다루며 detector 성능을 보지 않는다. runner에는 train/dev만 mount한다. selector는 train opaque ID·이미지·feature만, trainer는 봉인된 선택 이미지의 라벨과 dev 라벨만 읽는다. test 이미지·라벨·feature를 runner에 넣지 않고 symlink/junction 우회도 검사한다.

GT 기반 feature 진단은 새 GT-blind DINO/Random 선택을 봉인한 뒤 수행한다. Entropy 구성 진단은 round별 query 봉인 뒤다. 진단값으로 선택 정책을 바꾸지 않는다. 과거 test에 속한 metadata를 curator가 본 사실과 현 test inference 차단을 구분한다.

모든 162개 scientific cell의 dev-best SHA와 분석 계획을 고정한 뒤 별도 test release와 승인으로 일괄 평가한다. imgsz640, batch8, conf=.001, IoU=.70, max_det300, 동일 evaluator와 실제 test membership을 봉인한다. 각 물리 checkpoint는 한 번 평가하고 shared R0 결과를 세 곡선에 참조한다. 낮은 test 결과를 이유로 재학습하거나 checkpoint를 다시 고르지 않는다.

## 9. 평가와 통계

Primary는 dataset별 **test mAP50–95 nAULC R0–R5**다. actual n_r에 따른 사다리꼴 면적을 n5−n0로 나눈다. Full AP로 정규화하지 않는다. 여섯 점 중 누락이 있으면 primary 미완으로 표시하고 보간하지 않는다.

`nAULC = Σ_r (n_r−n_(r−1))*(AP_r+AP_(r−1))/2 / (n5−n0)`

Primary 대비는 각 dataset의 DINO_F518−Random 및 DINO_F518−Entropy, 총 네 대비다. **네 대비를 하나의 Holm family**로 명시하되 효과와 raw p값은 dataset별로 계산한다. 두 dataset의 관측치를 n=10으로 합치거나 pooled p값을 만들지 않는다. Entropy−Random은 secondary descriptive다.

각 대비에서 paired seed 차이 5개, mean/sample SD(ddof1), median, min/max, wins/losses/ties를 모두 보고한다. paired percentile bootstrap 100000회, GC10 seed2026091912/NEU2026091913, 95% interval은 작은 n의 탐색적 불확실성 요약이다. round나 test image를 acquisition 독립 반복처럼 resample하지 않는다.

exact sign-flip은 32개 부호 조합, absolute mean, >= tail의 양측 reference test다. 대칭성/부호 교환 가능성 가정을 명시한다. 최소 raw p=.0625이며 네 대비 Holm 보정의 최소값은 .25다. p<.05 성공 조건, 사후 단측 검정, 유리한 seed 추가, CI로 유의성 우회는 금지한다. 비유의 결과를 동등성으로 해석하지 않는다.

5×1은 고정 learner·split에 조건부인 acquisition trajectory 변동을 측정한다. 고정 seed라도 데이터가 달라 training stochasticity가 사라지는 것은 아니며 이를 별도 분리 추정하지 못한다. 과거 10×2 결과와 분산을 합치지 않는다.

Secondary는 round/R5 mAP50–95, mAP50, 전체 class AP50–95, round별 최저 클래스 AP 및 이름, 그 minimum curve의 nAULC다. worst-class가 round마다 바뀔 수 있음을 명시한다. GT가 있는 클래스의 무검출 AP=0, GT가 없는 클래스는 NA이며 library maps의 fallback 평균을 AP로 기록하지 않는다.

선택 구성은 신규와 누적을 구분해 class image incidence·bbox incidence·component 수·pool 대비 비율을 보고한다. 구성 차이와 AP 차이의 상관은 기술적 연관이다. 전체 10/6클래스를 보고하며 유리한 클래스·round만 고르지 않는다.

## 10. 정확한 실행 수

**dataset당 5×(shared R0 1 + 3방법×5round) + Full 1 = 81. 두 dataset 합계 162개, 과거 학습 재사용 0개.**

구성은 새 shared R0 10개, Random 50개, DINO_F518 50개, Entropy 50개, Full 2개다. 각 matrix는 81개 고유 행이며 실제 N·선택·runtime SHA가 채워지기 전 실행할 수 없다. train N 변경은 이 cell 수 구조를 바꾸지 않는다.

기술 smoke는 새 protocol에서 dataset당 1 epoch 1회, 총 **2회**를 별도 계획한다. scientific 결과에는 포함하지 않는다. 실패·재시도가 없다면 학습 시작 호출은 162+2=164회이며, 실제 attempt 수는 원장에 별도 기록한다. feature extraction·Entropy scoring·test evaluation은 training call에 넣지 않고 비용 원장에는 별도 기록한다. Entropy scoring은 dataset당 25회, 총 50회다.

RTX3070은 CPU 감사·adapter 검증과 승인된 기술 smoke에 사용할 수 있다. 다른 장비의 결과만으로 목표 RunPod 환경 검증을 대체하지 않는다. method별로 다른 GPU 환경을 배정해 환경을 방법과 혼동시키지 않는다. 과거 smoke 경과시간을 새 N·환경의 정확한 비용으로 사용하지 않는다.

## 11. 실행 순서와 장애 정책

현재는 설계 단계다. 이후 별도 승인 아래 raw/content audit → 새 canonical/component/split → test 차단 → 새 feature cache·GT-blind 선택 봉인 → 사후 feature 진단 → Entropy/YOLO 기술 smoke와 runtime guard → source/config/data/seed/matrix 봉인 → 본학습 승인 → 필요한 새 cell 실행 → 전체 integrity → test release 승인 → 평가·분석 순서다.

Entropy의 미래 query는 이전 best가 있어야 하므로 R1–R5 runtime DAG gate를 별도로 적용한다. pre-flight에서 미래 선택 목록이 존재한다고 가장하지 않는다.

output은 새 experiment root, exist_ok=False, resolved path guard, scientific cell ID의 atomic lock을 사용한다. STARTED/COMPLETE/FAILED/INTERRUPTED 원장은 append-only다. COMPLETE는 100 epoch, finite, 실제 config, artifacts·hash·종료시각·backup 검증까지 필요하다. 학습 seed·membership·source/config가 다른 attempt를 같은 cell로 숨기지 않는다.

동일 v2에서 이미 COMPLETE인 run은 임의 반복하지 않는다. 새 162회 설계라는 지시는 같은 cell의 무제한 재시도 승인이 아니다. 중단 후 optimizer/scheduler/AMP/RNG/data-order의 연속성이 증명되지 않으면 자동 resume하지 않는다. 실패 cell의 동일 조건 restart나 새 protocol 변경은 사유와 별도 승인에 따른다. Entropy parent가 바뀌면 descendant 경로를 함께 검토한다. 낮은 AP는 실패가 아니다.

## 12. 논문 주장과 현재 상태

최종 main은 새 두 dataset 분할에서 정의된 세 방법의 고정 learner·5개 선택 seed·10–35% 이미지 구간에 대한 비교가 된다. 방향은 아직 모른다. 기존 연구는 설계 근거로만 연결하며 새 평균·SD·p값에 섞지 않는다.

“물리적 생산 그룹을 완전히 복원”, “어떤 산업에서도 일반화”, “DINO가 결함만 표현”, “F518이 두 dataset에서 정보를 추가”, “모든 uncertainty보다 우수”, “65% 실제 비용 절감”을 주장하지 않는다.

**NOT READY:** 새 component 감사와 split, 실제 N과 class support, 미정의/충돌 annotation 판정, 새 Entropy adapter, OOM/NaN fail-fast guard, 목표 환경과 test 차단을 아직 실행 검증하지 않았다. 최신 clean 81회 회수 여부는 이 판정의 gate가 아니다.
