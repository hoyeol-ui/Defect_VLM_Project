# 객체 탐지 능동학습 그림·표 관례: 원문 확인 기록

확인일 2026-09-22. 아래 다섯 편의 CVF 공개 원문 PDF를 직접 열어 그림과 본문을 확인했다. PDF 사본과 페이지별 텍스트는 이 폴더 `sources/` 및 `.txt`에 보관한다. **이 다섯 편만으로 모든 선행연구의 보편 관례를 주장하지 않는다.** 특히 방법마다 “budget”의 분모와 비용 단위가 다르다.

| 원문·확인 위치 | x축 / 예산 정의 | y축 / 표시 단위 | 반복·불확실성 | Random 및 Full 기준 | 라운드와 최종 표 |
|---|---|---|---|---|---|
| [Aghdam et al., *Active Learning for Deep Detection Neural Networks*, ICCV 2019, Fig. 5, §4](https://openaccess.thecvf.com/content_ICCV_2019/papers/Aghdam_Active_Learning_for_Deep_Detection_Neural_Networks_ICCV_2019_paper.pdf) | 문제 정의의 B는 선택 **이미지 수**(image-centric). Fig. 5의 범례는 cycle와 전체 미라벨 풀 대비 표기 비율 N을 병기한다. 실제 plot x축은 FPPI이므로 이 그림을 budget-x/AP-y 사례로 인용하지 않는다. | 보행자 miss rate 대 FPPI, 로그 축. mAP/AP가 아니다. | 방법별 5회 반복의 **평균**을 본문에 명시. Fig. 5에 오류막대·band는 보이지 않는다. | Guided random을 별도 패널에 제시. 전체 학습자료 모델은 검은 성능 곡선. | Fig. 5는 cycle 번호와 라벨 비율을 범례에 병기한다. |
| [Pardo et al., *BAOD: Budget-Aware Object Detection*, CVPRW 2021, Fig. 1, §2–3](https://openaccess.thecvf.com/content/CVPR2021W/LXCV/papers/Pardo_BAOD_Budget-Aware_Object_Detection_CVPRW_2021_paper.pdf) | `Budget`: 이미지 수가 아니라 **약/강 annotation의 노력·비용**을 합친 척도. | `Mean Average Precision (%)`, 0–100 계열. Fig. 1의 비용–성능 곡선. | Fig. 1에 band/error bar는 없다. 평균 반복 수·분산 표기는 이 그림에서 확인되지 않는다. | Fig. 1은 여러 감독 방식 곡선과 수평 참조선을 표시. Random Sampling은 §3의 비교 전략으로 정의되지만 Fig. 1 범례의 독립 Random 선은 아니다. | 비용에 따른 곡선을 우선 제시. 이 논문의 비용 축을 우리 이미지 비율 축과 같은 실측 비용으로 해석하면 안 된다. |
| [Lyu et al., *Box-Level Active Detection*, CVPR 2023, Figs. 1, 3; §4](https://openaccess.thecvf.com/content/CVPR2023/papers/Lyu_Box-Level_Active_Detection_CVPR_2023_paper.pdf) | Fig. 1 왼쪽 `#images`, 오른쪽 `#boxes`; Fig. 3 `#boxes`. 이미지 예산과 박스 예산이 다른 질문을 다룸을 직접 보여준다. | VOC `mAP50 (%)`; COCO `mAP (%)`=AP@0.50:0.95 (§4). 0–100 수치. | §4에서 **독립 3회 평균과 SD** 보고를 명시. Figs. 1, 3은 눈에 띄는 band 없이 방법별 곡선. | Random 선 포함; `FullSup`/`Full Sup` 수평 기준선 포함. | 실험 cycle은 방법 절차에 있지만 그림 x축은 실제 주석량. 클래스별 비교도 제시. |
| [Yang et al., *Plug and Play Active Learning for Object Detection*, CVPR 2024, Figs. 4–6, Table 1; §4](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Plug_and_Play_Active_Learning_for_Object_Detection_CVPR_2024_paper.pdf) | Figs. 4–5 `Training Data (%)`; Fig. 6 일부는 `Training images`. | COCO `AP (0.5:0.95)`와 VOC `mAP`, 수치 0–100. 서로 다른 benchmark의 `mAP` 정의를 혼용해서 읽으면 안 된다. | §4는 3개의 다른 초기집합에 대한 **평균 성능**을 명시. Table 1은 예산별 `mAP on % of labelled images`를 값±수치로 병기하나 ±의 통계량 명칭은 확인된 본문만으로 확정하지 않는다. Figs. 4–5에는 band/error bar가 보이지 않는다. | Random 및 Entropy를 명시적 선으로 표시. Figs. 4–5에 Full-data 수평선은 없다. | Fig. 4/5에서 라운드는 x축이 아니며 실제 누적 training data 비율을 x축으로 쓴다. Table 1은 각 budget 열의 수치를 함께 제시. |
| [Yamani et al., *Active Learning for Single-Stage Object Detection in UAV Images*, WACV 2024, Fig. 4, Tables 2, 4, 5; §4](https://openaccess.thecvf.com/content/WACV2024/papers/Yamani_Active_Learning_for_Single-Stage_Object_Detection_in_UAV_Images_WACV_2024_paper.pdf) | Fig. 4는 누적 선택 **이미지 수**(왼쪽)와 객체/박스 수(오른쪽)를 따로 비교. | `mAP@0.5`, 원문 표는 0–1 소수, 그래프도 같은 맥락의 수치. 우리 AP50:95와 직접 비교 불가. | 이 그림과 표에 반복의 SD/CI 또는 band는 확인되지 않는다. | Random과 Whole 모델을 표에서 비교; Fig. 4에 Random 곡선. | Fig. 4의 반복 회차는 x축이 아니며 누적 라벨 수가 x축. Tables 2/4/5는 budget과 최종 성능·클래스 성능을 함께 제시. |

## 우리 논문에 적용할 결정

1. 주 그림 x축은 **`Labeled images (% of training pool)`**로 한다. 실제 정의는 `100 × cumulative labeled images / acquisition pool train N`이다. `Labeling budget`이라는 표현은 통상적이지만 box 수나 사람의 시간·비용을 측정하지 않았으므로 축에서 `images`를 명시한다. 위 Aghdam의 image-centric 정의, Yang의 Training Data %, Lyu의 #images/#boxes 구별이 근거다. `R0`–`R5`는 내부 단계이며 그림 x축에 두지 않는다.
2. x값은 GC10/NEU 각각 원래 정수 `n_r`를 해당 train N으로 나눈 정확한 실수다. 눈금은 10/15/20/25/30/35%로 읽기 쉽게 두고 caption에 실제 n 목록을 쓴다. 라운드별 반올림 오차를 버리고 10/15/...에 데이터를 강제 이동시키지 않는다.
3. y축은 우리 봉인 evaluator의 `metrics/mAP50-95(B)`로 확인된 **`mAP@0.50:0.95 (%)`**. 그림에는 0–100%의 공통 y축을 사용하고, 차이가 작은 경우 paired difference 별도 보조 그림으로 읽는다. mAP50은 별도 보조표로 둔다. Yang의 COCO AP@0.50:0.95와 Lyu의 COCO mAP 정의가 관련 선례다.
4. Random/DINO-F518/Entropy는 같은 세 색·선종류·마커로 모든 그림에 표시한다. 세 방법이 같은 R0 모델을 참조한다. 평균선 주위에는 **5 acquisition seed의 sample SD (ddof=1) band**를 그려 관측 변동을 설명한다. SD band는 문헌의 필수 관례라고 주장하지 않으며, Lyu의 평균+SD 보고와 우리 반복 설계에 맞춘 명시적 선택이다. 통계적 유의성을 뜻하지 않는다.
5. Full-data는 dataset당 1개 모델의 수평 점선으로 `Full-data reference (1 run)`이라고 표시한다. 이를 신뢰구간이나 통계 검정의 반복으로 취급하지 않는다. Aghdam Fig. 5 및 Lyu Figs. 1/3의 전체자료 기준 시각화와 정합한다.
6. 주 곡선과 별도로 예산별 `mean ± SD`, 95% CI, seed별 raw 값을 표에 남긴다. nAULC를 그림과 표로 정리하되, 원문들의 Budget-Average mAP 또는 AULC를 수식이 같은 지표로 오인하지 않는다. 우리 nAULC는 v2 프로토콜의 고정 식을 따른다.

## 해석 경계

이미지 1장을 라벨링하는 실제 비용은 bbox 개수·난도에 따라 달라질 수 있다. Lyu Fig. 1 및 Yamani Fig. 4가 그 차이를 직접 보여준다. 따라서 이번 `35%`는 **이미지 수 기준**이며 “annotation 시간 65% 절감”을 의미하지 않는다. 선행연구의 AP50, VOC mAP, COCO AP50:95를 같은 숫자로 비교하지 않는다.
