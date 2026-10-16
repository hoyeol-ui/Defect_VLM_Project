# 클래스별 데이터 지원량 의문: 독립 검토 및 최소 후속 점검

2026-10-16. 기존 작업 `c213491`이 원격 브랜치와 일치하고 working tree가 깨끗한 상태에서 별도 검토했다. 이번 변경은 문서만이며 새 학습·추론·Patch 추출·분할·선택·GPU 대여를 수행하지 않았다.

## 1. 판단: 일부 타당, 원인 확정은 현재 근거 부족

| 의문 | 독립 판단 |
|---|---|
| 클래스별 이미지/객체 사례가 적은가? | **일부 클래스의 절대 개수가 작고 클래스 간 불균형이 있다는 것은 공개 집계로 확인된다.** 그러나 학습에 필요한 충분한 표본 수의 문턱은 모른다. ‘작다’와 ‘낮은 AP의 원인이다’를 구분한다. |
| 다중클래스 학습 때문에 불리한가? | 다중클래스 자체가 원인이라는 근거는 없다. 클래스 구성·객체 크기·시각적 유사성·라벨 품질·학습 조건을 구분해야 한다. 클래스 간 음의 전이/혼동은 현재 AP만으로 입증 불가다. |
| CLS 다양성과 클래스별 다양성이 다른가? | 방법별 클래스 구성 차이와 AP의 비단조적 관계는 확인된다. 하지만 클래스 내 형태 다양성 부족까지 입증한 것은 아니다. K8은 사용자 요약만 있어 정식 대조에 아직 넣을 수 없다. |

**결정:** 새 실험을 추가하지 않고 기존 04 ‘0단계 항목4(이미지/box 수·크기·중복·coverage)’를 구체화한 공통 사후 점검 D0 하나를 남긴다. 이미 있는 클래스 구성/AP 통계를 다시 만드는 작업은 제외한다. FF 재현을 우선하며 이 의문 때문에 현재 실행 순서를 뒤집거나 epoch/모델/선택 규칙을 변경하지 않는다.

## 2. 확인한 실제 공개 집계

근거는 최종 clean-split v2 / `ANNOTATION_PATCH_v2_0_2` 결과다. 과거 `results/thesis_final`의 10획득시드 결과와 섞지 않았다. 전체 train은 GC10 1604장, NEU 1258장; R5는561/440장이다.

아래는 **획득 seed2026091901 한 개**의 원문 행을 연결한 표다. 모든 수량은 이미지 포함 수/GT box 수이며 `전체 이미지÷클래스 수`로 추정하지 않았다. AP는 같은9월 YOLOv8n 최종 test AP50–95(%)다. 다른5시드 평균이나10월 모델 비교 dev 결과와 혼합하지 않는다.

공개 구성표를 만든 코드는 이미지마다 클래스 존재를 한 번 세고, 활성 주석 파일의 해당 클래스 box 행을 센다. 최종 test 구성표의 `selected_unique_boxes`와 기존 구성표의 `cumulative_boxes`가240행에서 일치했다. 이는 **공개 표의 일치 확인**이며 원본 annotation의 중복/정확성을 이번에 재검증한 것은 아니다.

### GC10-DET

| 클래스 | 전체 train 이미지/box | Random R5 이미지/box | DINO R5 이미지/box | Random AP | DINO AP | Full AP |
|---|---:|---:|---:|---:|---:|---:|
| punching_hole | 230/230 | 77/77 | 111/111 | 45.299 | 44.655 | 50.151 |
| welding_line | 358/358 | 127/127 | 203/203 | 29.771 | 27.949 | 31.247 |
| crescent_gap | 185/185 | 67/67 | 98/98 | 53.824 | 52.808 | 52.185 |
| water_spot | 217/248 | 72/81 | 91/112 | 31.287 | 32.018 | 37.321 |
| oil_spot | 175/398 | 66/111 | 72/135 | 24.676 | 23.996 | 27.807 |
| silk_spot | 514/619 | 173/214 | 95/130 | 1.658 | 7.973 | 7.520 |
| inclusion | 141/243 | 46/87 | 34/67 | 3.042 | 2.805 | 5.242 |
| rolled_pit | 32/59 | 14/27 | 20/39 | 5.322 | 2.471 | 3.027 |
| crease | 37/52 | 11/18 | 28/41 | 3.871 | 5.480 | 16.677 |
| waist_folding | 98/101 | 43/43 | 19/19 | 22.515 | 14.600 | 14.680 |

### NEU-DET

| 클래스 | 전체 train 이미지/box | Random R5 이미지/box | DINO R5 이미지/box | Random AP | DINO AP | Full AP |
|---|---:|---:|---:|---:|---:|---:|
| crazing | 210/482 | 71/166 | 29/61 | 12.939 | 13.325 | 20.053 |
| inclusion | 268/707 | 84/215 | 119/278 | 35.358 | 31.554 | 36.872 |
| patches | 238/612 | 85/200 | 85/192 | 49.331 | 49.962 | 53.694 |
| pitted_surface | 211/302 | 75/114 | 107/156 | 34.523 | 39.562 | 39.682 |
| rolled-in_scale | 210/440 | 78/150 | 31/59 | 25.252 | 23.091 | 29.693 |
| scratches | 210/384 | 73/125 | 123/220 | 35.794 | 42.581 | 43.005 |

### 이 표로 알 수 있는 것과 없는 것

- GC10 train의 rolled_pit32장/59box, crease37장/52box는 적은 사례의 후보인 반면 silk_spot514장/619box는 이들보다 훨씬 많다. 그런데 Full silk_spot AP도7.520%에 머문다. **모든 취약 클래스가 단순히 사례 수가 적어서 낮다는 설명은 지지되지 않는다.**
- NEU train의 클래스 이미지 포함 수는210–268장이지만 box 수는302–707개다. 동일한 이미지 지원량도 객체 노출량은 다르다. DINO R5 seed1은 crazing29장/61box, scratches123장/220box로 구성 편차가 커진다. 이미지 수만으로 학습 신호를 대체할 수 없다.
- 기존5획득시드 평균에서도 GC rolled_pit은 Random→DINO12.8→20.4장인데 AP6.401→5.443%, NEU inclusion은92.8→116.2장인데33.561→32.189%다. GC silk_spot은180.2→98.0장인데2.534→3.660%다. 이번 seed1 표와 별개의 요약이다. 수량·AP가 항상 같은 방향으로 변하지 않는다.
- 전체 train의 클래스 이미지 포함 수 합은 GC1987(고유 이미지1604), NEU1347(고유 이미지1258)이다. R5 seed1은 GC Random696/DINO771(각561장), NEU466/494(각440장)이다. **공개 집계 정의상 다중클래스 이미지가 존재한다.** 하지만 초과 합을 이미지 수로 나눈 값을 다중클래스 이미지 비율로 쓰면 안 된다. 한 이미지가 몇 개 클래스를 포함하는지, zero-box 이미지가 있는지에 따라 달라진다.
- 전체 box는 GC2493/NEU2927개, R5 seed1 Random852/970개·DINO955/966개다. 이미지당 다중객체도 있으나 정확한 비율은 이미지 단위 자료가 필요하다. box가 많아도 같은 이미지의 반복 결함이면 독립적인 시각적 사례 수가 늘어난 것과 같지 않다.
- 현재 표에는 box 위치·폭/높이·면적·종횡비, 클래스 공동출현 행렬, 실제 결함 모양이 없다. 원본 train label overlay가 맥의 해당 export 경로에 없으므로 이런 분포를 본 것처럼 해석하지 않는다.

## 3. 원인 후보와 구분 가능성

| 후보 | 현재 근거 / 추가로 필요한 근거 | 허용되는 해석 |
|---|---|---|
| 절대 표본 부족 | 작은 클래스의 이미지·box 수 확인; 유효 독립 component·형태 다양성 미확인 | 원인 후보. 충분성 임계값이나 부족의 인과 효과는 미확정 |
| 클래스 불균형 | train pool과 고정 R5의 상대 수량 차이 확인 | 구성 불균형은 기술 가능. 균등 quota가 AP를 높인다는 뜻 아님 |
| 작은 객체·가늘고 긴 box·가장자리 | 현재 aggregate에는 좌표 없음; 활성 train box와 실제 resize 기록 필요 | 크기/위치/종횡비의 연관 후보. bbox 종횡비는 결함의 실제 형태가 아님 |
| 클래스 혼동·누락·위치 오류 | AP만으로 분리 불가; 저장된 예측 box/class/score, 동일 평가 GT·matching 정책 필요 | 가능하면 기존 dev 예측으로 오류 분해. 예측 없으면 미확인으로 남김 |
| 학습 부족 | 기존 epoch 곡선·Full/R5 설정 연결 필요 | 다른 원인과 병존 가능. 표본 수를 보고200epoch를 자동 결정하지 않음 |
| 클래스 간 학습 간섭 | 단일 모델의 클래스별 AP만으로 입증 불가 | 별도 통제 실험이 필요할 수 있으나 이번 최소 범위에는 단일클래스 모델 재학습 등을 추가하지 않음 |

**절대 수량과 상대 불균형은 관측 지표로 구분할 수 있다. 충분한 학습 사례가 실제로 부족한지와 그것이 AP에 미친 인과 효과는 현재 기록만으로 분리할 수 없다.** 클래스 간 AP 상관 하나는 난이도·객체 크기·평가 support·seed를 통제하지 못한다. 새 상관검정/유의성 검정을 늘리지 않고 기존5시드 내 방법 간 변화와 반례를 함께 기술한다.

## 4. Full-data·문헌 비교의 경계

9월의 기존 R5와 Full은 같은 clean split·YOLOv8n·test AP50–95 자료로 **기술적 비교가 가능**하다. 기존 R5_vs_full_data.csv를 그대로 재사용하면 DINO5시드 평균 대 Full은 GC23.280% 대24.586%(−1.305pp), NEU33.063% 대37.167%(−4.103pp)다. Full은 데이터셋별1회이므로5개의 독립 Full 반복처럼 복제하지 않는다.

Full로 늘리는 것은 class count 하나만 바꾸는 조작이 아니다. 선택 구성·전체 사례 수·예산·epoch당 optimizer step 등이 함께 달라진다. 따라서 이 차이는 표본 부족의 순수한 인과 효과가 아니며 Full도 절대적 성능 상한은 아니다. 일부 클래스는 R5가 Full보다 높을 수도 있다.

10월 연구 A의 결과는 DINO R5 seed1·여러 모델·sealed **dev**다. 9월 Full **test**와 직접 빼서 데이터/모델 효과로 해석하지 않는다. 이번 사용자 설명의 공통 모델 선정·RTX5090 환경 구축은 현재 추진 방향으로 받아들이되, 완료됐다고 간주하지 않고 관련 새 run 기록이 생기면 별도 버전으로 연결한다. 기존 A 모델·epoch·optimizer 계획은 변경하지 않는다.

문헌이 Full-data를 썼다면 현재 R5와의 예산 차이는 중요한 비교 제한이다. 그러나 **현재 Full-data도 문헌보다 낮은 문제 전체를 R5 표본 부족으로 설명할 수는 없다.** 데이터 버전·제외/주석 규칙·분할·pretrained·학습/evaluation recipe를 함께 확인해야 한다. 이번에는 문헌 조사나 재학습을 반복하지 않았다.

## 5. 최소 후속 항목 D0 — A·B 공통, 기존 점검의 보완

**우선순위 P1: FF 재현을 앞당기는 조건은 아니며, 새 GPU 학습/새 선택 규칙을 정하기 전에 기존 자료로 수행할 공통 설명 점검.** 기존04의0단계4번을 대체·구체화하는 항목으로 관리한다. 현재 수행한 공개 표 연결을 반복하지 않는다.

| 순서 | 기존 입력과 최소 작업 | 산출물 / GPU |
|---|---|---|
| D0-1, 현재 확인 완료 | 아래 공개 구성/AP 표를 연결하고 seed1 지원량과5시드 반례·Full 비교를 검토 | 본 문서. GPU0 |
| D0-2, 원본 확보 시 P1 | Windows의 활성 `ANNOTATION_PATCH_v2_0_2` train label overlay·고유 train ID/크기·봉인된 Random/DINO R5 ID·component를 읽기 전용 연결. 각 이미지의 distinct class 수/box 수를 먼저 집계한 뒤 다중클래스·다중객체·zero-box 비율, 클래스 공동출현, class별 box/image 비, normalized box area·center·aspect의 q10/median/q90와 유효 component 수를 산출. 기존5시드 집합만 재사용 | 하나의 `class_support_geometry_audit.csv`와 본 문서의 결과/미확인 칸 보완. GPU0, 새 학습0 |
| D0-3, 자료가 이미 있으면 P2 | 같은 split/seed/budget/cache 정의의 K8 봉인 ID와 실행 메타데이터를 연결해 동일 집계 칸만 추가. 현재 K8 raw는 미확인이고 AP는 확인되지 않음 | D0-2의 같은 표에 추가; 없으면 NA, K8 재선택/학습 금지 |
| D0-4, 저장된 dev 예측이 이미 있으면 P2 | 동일 모델·평가 설정의 기존 prediction JSON 등으로 class confusion/미검출/위치 오류를 구분. confidence·IoU matching 정책을 기록 | 본 문서의 원인 후보 보완. 기존 파일 처리만 CPU; 예측이 없으면 보류 |

1차로 파일 연결과 누락 목록까지만 수행한다. 원본이 없거나 source/hash가 안 맞으면 **미확인으로 종료**하며 과거 base label·다른 split로 대체하지 않는다. 대규모 재검산·새 annotation·새 inference·YOLO 학습은 범위 밖이다. 캐시·원본 실험·봉인 결과를 덮어쓰지 않는다.

D0-2의 Full pool GT는 **과거에 완료·봉인된 연구의 사후 진단 공간에서만** 처리한다. 현재/향후 미라벨 pool의 실제 class/bbox를 선택 점수, quota, crop, prototype, hyperparameter 튜닝에 재사용하지 않는다. 결과는 집계 보고서로만 넘기고 selection 입력 파일에 GT를 추가하지 않는다. R4 Patch 진단의24장 샘플을 이 분석을 보고 클래스별로 다시 뽑지 않는다. 기존 test AP는 설명용 과거 기록이며 현재 조건을 고르는 최적화 목표로 삼지 않는다.

## 6. 연구 A·B에 미치는 영향과 중단 기준

- **A:** 모델·학습 조건을 바꾸기 전 ‘데이터 예산/지원량/geometry’라는 설명 축을 보완한다. 데이터 부족을 이유로200epoch·batch·optimizer·5090 학습을 자동 추가하지 않는다.
- **B:** global feature coverage와 클래스 구성·국소 geometry가 다르다는 가설을 더 정확하게 기술할 수 있다. 그러나 클래스 수량만으로 Patch가 해결책이라는 결론이나 GT 기반 균형 선택으로 변경하지 않는다. 현재 R4→R5 재현/제약/정답 분리 계획 유지.
- 이미 있는 클래스 표를 다시 만들거나 단순 AP 상관만 얻는 수준이면 추가 작업하지 않는다. raw label·prediction이 없으면 ‘근거 부족’을 성과로 남긴다. geometry에서 차이가 나도 원인 후보이며 독립 검증 없는 기술적 기여로 포장하지 않는다.
- 확인한 이 브랜치에는 신규 저장소 이전 작업의 구체적인 대상·계획이 없다. 이전을 임의로 만들지 않는다. 향후 별도 이전 계획이 확정되면 이 문서와 아래 source commit/hash만 연구 A·B 공통 인계 목록에 연결한다. 원본 이미지·GT·weights를 이번 커밋에 업로드하지 않는다.

## 근거와 확인 범위

아래 파일은 기존 최종 결과 commit `4f887463c334e62df78debf87c41ac2e588bb129`에서 읽었다. 원문 표 전재·제한된 합계/행 연결만 수행했고 기존 통계 계산·모델 실행은 반복하지 않았다.

- [SELECTION_COMPOSITION.csv](https://github.com/hoyeol-ui/Defect_VLM_Project/blob/4f887463c334e62df78debf87c41ac2e588bb129/results/clean_split_v2_development_20260922/02_tables/SELECTION_COMPOSITION.csv) — SHA256 `b6ac6adf0cfdcebf7207dd7ad3674f5b4011be2cf3e933ad308f3a8728af2e3e`
- [selected_class_composition_by_seed.csv](https://github.com/hoyeol-ui/Defect_VLM_Project/blob/4f887463c334e62df78debf87c41ac2e588bb129/results/final_test_20260925/statistics/selected_class_composition_by_seed.csv) — SHA256 `5986b95da718f8ccd5870409f64370710cee6a68ea9f09ad0942edd5c28e5003`
- [class_ap.csv](https://github.com/hoyeol-ui/Defect_VLM_Project/blob/4f887463c334e62df78debf87c41ac2e588bb129/results/final_test_20260925/raw/class_ap.csv) — SHA256 `e9414058f837be45cf4c5ab0ea47d8bad23cd7b3a74b19ebc21b49a5bb02b081`
- [R5_class_AP_difference.csv](https://github.com/hoyeol-ui/Defect_VLM_Project/blob/4f887463c334e62df78debf87c41ac2e588bb129/results/final_test_20260925/statistics/R5_class_AP_difference.csv) — SHA256 `d8440ad9af0ea2d560339ce657b1b5a26c5bcbced70b6bfde8ebfa28f018a99b`
- [R5_vs_full_data.csv](https://github.com/hoyeol-ui/Defect_VLM_Project/blob/4f887463c334e62df78debf87c41ac2e588bb129/results/final_test_20260925/statistics/R5_vs_full_data.csv) — SHA256 `4910f88cc915ac6a463f45151b7d3b6a97b4733b38d0b5144dc9217b64f25ba3`
- [run_analysis.py](https://github.com/hoyeol-ui/Defect_VLM_Project/blob/4f887463c334e62df78debf87c41ac2e588bb129/results/clean_split_v2_development_20260922/07_reproducibility/run_analysis.py) — SHA256 `386fc0a01c2da62021b65867c6edac681ad9f123dee9dfa845265a51017ef242`

기존 계획 중복 확인: 로컬 연구 검토01 §6, 04 §2 항목4, 06·07의 데이터/annotation 점검과 이번 R4 Patch 진단을 읽었다. 새 질문은 기존 공백을 구체화하는 것으로 연결했으며 별도 대규모 실험 경로를 만들지 않았다.
