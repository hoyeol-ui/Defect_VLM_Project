# 두 데이터셋의 F518 입력·특징·선택 진단 계획

**입력 정책은 두 데이터셋 모두 DINOv2-small F518로 채택한다.** 이유는 공통된 전체 시야 정책과 GC10의 crop 손실 방지다. 성능 우위가 확인됐다는 뜻은 아니다. 이 파일의 model inference·feature extraction·selection은 이번에 실행하지 않았다.

## 1. 같은 정책이 두 데이터셋에 주는 다른 효과

| 항목 | GC10 | NEU |
|---|---|---|
| 로컬 원본 | 2048×1000 | 200×200 |
| F518 유효 영상 | 518×253 | 518×518 |
| 변환 의미 | 긴 영상을 축소하고 위·아래 padding | 기존 patch를 2.59배 확대 |
| padding 면적 | 51.1583% | 0% |
| 인식된 기존 canonical bbox | 3563개, 별도 미정의 객체 1개 | 4183개 |
| F224→F518 짧은 변 median | 13.952→32.375px | 50.4→116.55px |
| 기하학적 bbox 소실/부분 절단 | 유효·인식 bbox에서 0/0 | 유효 bbox에서 0/0 |
| 해석 | F224보다 nominal defect pixel 수가 많고 전체 시야 보존 | 확대가 원래 없던 디테일을 복원하지 않음. parent 전체 프레임도 복원하지 않음 |

출처: 새 BBOX_GEOMETRY_v2.csv, 기존 canonical membership에 해당하는 raw XML의 좌표 재계산. 이는 새 train split 결과나 실제 tensor 검증이 아니다. F224는 좌표 진단용 비교이며 학습 arm이 아니다. GC10의 32.375px는 전체 기존 canonical 기준이므로 과거 clean train 1605장의 36.17px와 분모가 다르다.

## 2. 입력과 특징의 정확한 명세

RGB로 decode하고 긴 변을 518로 맞춘다. 짧은 변은 half-up 정수 반올림·최소 1px, torchvision uint8 bicubic antialias, 중앙 padding RGB=(124,116,104), 좌/상 여백은 남는 정수 pixel의 floor 절반이다. 이미지를 square로 늘이지 않는다. processor의 resize/center_crop은 둘 다 False로 명시한다. tensor는 B×3×518×518이다.

/255 후 ImageNet mean (.485,.456,.406), std (.229,.224,.225). frozen eval/inference_mode, FP32, autocast/TF32 off. 모델은 facebook/dinov2-small, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, weight SHA `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`를 고정한다.

patch14, grid37×37, patch1369+CLS1=1370 tokens. pooler_output은 확인한 구현의 최종 layer norm 후 CLS 384D이며 patch 평균이 아니다. feature를 L2 정규화해 float32 배열과 opaque ID를 함께 저장한다. 새 v2 membership에서 새 cache를 만들며 과거 feature/selection을 final 결과로 복사하지 않는다.

필수 기술 검사는 finite 전부, 384D, zero norm 0, 최대 norm 오차 ≤1e-6, feature 행과 train ID의 bijection, raw/model/renderer/feature SHA다. 동일 환경 재현 subset의 성분 오차 ≤1e-4, cosine drift ≤1e-5를 확인하되 그 tolerance만으로 farthest-first 선택 동일성을 보장하지 않는다. 실제 선택에는 단일 canonical cache SHA를 사용한다.

## 3. GT 접근 시점

데이터 curator의 annotation 정합성·분할 class support 확인은 데이터셋 구성 업무이며 선택기의 GT 사용과 분리한다. model·입력·distance·선택 규칙은 feature와 GT의 연관 결과를 보기 전에 고정한다.

향후 Random/DINO의 GT-blind 경로를 새 split에서 생성하고 선택 ID를 봉인한 뒤 train GT를 진단 공간에서 join한다. 그러면 DINO 경로를 봉인한 상태에서 학습 전 구조 진단을 할 수 있다. Entropy는 각 round query 봉인 뒤에만 그 선택 구성에 GT를 join한다. Entropy R2–R5까지 사전에 선택했다고 가장하지 않는다. 모든 GT 기반 feature/선택 진단 결과는 query score·seed·입력 튜닝으로 되돌리지 않는다. test/dev GT는 feature tuning에 사용하지 않는다.

## 4. 전체 분포 기반 진단

| 질문 | 계산 대상과 고정 규칙 | 해석·제한 |
|---|---|---|
| bbox 보존 | 실제 새 train의 모든 box에 변환. 소실·부분 절단, width/height/짧은 변·면적 p10/median/p90, class별 분포 | 실제 texture 보존이나 detector 개선으로 등치하지 않음 |
| ordinary 1-NN | 전체 train, self 제외, 동일 label 하나 이상 공유율과 label-set exact match | 전체 이미지 통계와 component별 평균을 함께 보고 |
| component-excluded 1-NN | 같은 최종 hard component를 모두 후보에서 제외 | 클래스가 한 component에 몰리면 무조건 0 처리하지 않고 지원 부족을 명시 |
| permutation chance | 각 component의 최소 opaque ID 대표 1장, 고정 NN graph, multilabel vector 전체를 9999회 permutation. seed GC10=2026091910, NEU=2026091911 | 대표 집합에 대한 null median/95% 범위. 전체 이미지와 같은 모집단이라고 부르지 않음 |
| same/different label 거리 | 서로 다른 component의 모든 pair. 라벨 공유/비공유를 분리해 ECDF·quantile | 같은 이미지가 여러 pair에 있으므로 pair 수로 p값을 부풀리지 않음 |
| 클래스 compactness | 클래스별 cross-component pair cosine 분포, class incidence·component 수 | multilabel은 여러 클래스에 기여. 모든 10/6클래스 보고 |
| 밝기 | 전체 train의 luminance mean/std와 NN 거리·선택 빈도의 Spearman, class별 분포 | 조명과 결함 종류의 교란을 함께 설명 |
| 방향/회전 | 전체 train 원본 대비 90°/180° 변환의 feature drift·NN 변화. 같은 F518 renderer | 몇 장의 예시로 결론내리지 않음. 실제 촬영 방향 label을 추정하지 않음 |
| bbox size | 전체 train 최대 bbox 면적·median 짧은 변과 NN 거리/선택 빈도, class별 분포 | 크기와 유용성·난이도의 인과를 단정하지 않음 |
| padding/background | GC10 전체 train에서 pad만 black으로 바꾼 고정 perturbation의 drift. NEU pad-only는 N/A | background mask가 없어 배경만의 인과 효과는 분리 불가. NEU padding 0이 배경 영향 0을 뜻하지 않음 |
| 선택 다양성 | seed·round·method별 신규 집합과 누적 집합의 모든 pair cosine, component 수, labeled reference와의 거리 | DINO 기본 F518 공간에서 세 방법을 비교. 서로 다른 공간의 절대값으로 우열을 정하지 않음 |
| 선택 구성 | 봉인 뒤 class image incidence, bbox incidence, component incidence, pool 대비 비율 | 합계는 multilabel 때문에 이미지 수보다 클 수 있음 |

이 계획은 대표 예시만 보는 검증이 아니다. 전체 train·전 클래스 분포가 기본이고, permutation 대표 집합은 component 의존성을 줄이는 별도 분석 단위다. 그림 예시는 수치 해석 보조다.

## 5. 진단 결과를 해석하는 규칙

기술 무결성은 PASS/FAIL로 판정한다. 의미 구조는 SUPPORTED/WEAK/INCONCLUSIVE로 보고한다. permutation 범위 내 1-NN이면 우연 수준을 넘는 구조가 확인됐다고 쓰지 않는다. 낮은 진단 수치를 개선할 때까지 입력·seed·GT 필터를 바꾸지 않는다. 기술적으로 올바른 약한 표현도 main 연구 질문의 유효한 결과가 될 수 있다.

“전체 영상 → 높은 NN 공유율 → 다양한 선택 → 높은 detector AP”를 필연적 연쇄로 가정하지 않는다. 새 final 결과의 방향은 미리 정하지 않는다. 선택 구성과 AP 변화는 descriptive association이며 causal mediation이 아니다.
