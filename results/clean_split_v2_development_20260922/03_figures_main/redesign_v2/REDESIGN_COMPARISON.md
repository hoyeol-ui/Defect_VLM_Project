# 논문 그림 재설계 v2

이번 변경은 **표현만 수정**한다. 원래 `03_figures_main/` 및 `04_figures_appendix/`의 그림, 모든 raw/summary CSV, seed 포함 여부, 통계 검정, 방법 정의는 그대로다. 새 스크립트 `plot_redesign_v2.py`는 확정 CSV를 읽어 이 폴더에만 PNG 300 dpi·벡터 PDF·SVG를 만든다. `DINOv2-FF`는 기존 실험 ID `DINO_F518`의 그림용 명칭이며, DINOv2-small의 F518 입력 특징에 greedy farthest-first를 적용한 동일 arm이다.

| 기존 표현의 문제 | 새 표현과 선택 이유 |
|---|---|
| 데이터셋별 0–50% y축이 결과 구간을 압축 | **본문 그림 1** `MAIN_FIGURE_1_AL_CURVES_COMMON_15_40`: GC10·NEU 2-panel, 공통 15–40% 축. 비교 가능한 동일 척도를 유지하면서 관심 구간을 읽기 쉽게 한다. `AL_CURVES_ALTERNATIVE_DATA_RANGE`는 데이터셋별 16–31/27–41% 축의 검토용 대안이며, 본문에는 공통 축을 권장한다. |
| 큰 그림 내부 제목·Full-data 범례가 공간을 차지 | (a)/(b) 패널명, 세 방법의 공통 범례, 가는 점선의 끝에 Full-data 수치를 직접 표기. Full-data는 데이터셋당 **1회 참고값**이며 오류범위가 없다. |
| 기존 nAULC 그림의 seed 연결선이 방법 간 순차 경로처럼 보임 | `NAULC_POINT_RANGE_ALTERNATIVE`는 연결선을 없애고 5 seed의 작은 점과 평균±SD만 표시했다. 축은 패널별 데이터 범위이므로 패널 사이 높이 자체로 절대값을 비교하지 않는다. **본문에는 `NAULC_MAIN_TABLE.md`를 권장**한다. GC10/NEU의 평균·SD·DINO−Random paired 차이가 표에서 더 정확하고 간결하게 읽힌다. 검토용 점 그림은 부록 후보이며, 기존 paired 그림도 부록 자료로 보존한다. |
| 절대 class AP의 오른쪽 60–100%가 비어 핵심 효과가 가려짐 | **본문 그림 2** `MAIN_FIGURE_2_R5_CLASS_DELTA`: R5에서 DINOv2-FF − Random class AP 평균 차이(%p)를 0선 중심으로 제시한다. GC10 `welding_line` +12.81, `waist_folding` −2.20; NEU `rolled-in_scale` −4.79를 원래 `05_CLASSWISE_RESULTS.csv` 그대로 표시한다. 색/막대는 기술적 차이이며 유의성 표지가 아니다. |
| 세 방법의 절대 class AP도 재현 자료로 필요 | `APPENDIX_GC10_R5_ABSOLUTE_CLASS_AP`, `APPENDIX_NEU_R5_ABSOLUTE_CLASS_AP`에 보존하고 x축은 0에서 시작한다. 상한은 각 데이터셋의 최대 평균+SD를 포함하는 5% 단위로 정했다. |

선·마커의 모양과 색은 전 그림에서 동일하다: Random 회색 원 실선, Entropy 청록 사각형 파선, DINOv2-FF 주황 삼각형 실선. 주 곡선은 실제 누적 라벨 이미지 수를 train acquisition pool로 나눈 정확한 비율에 놓았다. 옅은 띠는 5 acquisition seed의 **표본 SD**이고 신뢰구간이나 유의성 표지가 아니다. R0는 각 seed에서 세 방법이 공유한다. 원래 0–50% 그림은 보관되어 축 선택을 나중에 감사할 수 있다.

## 캡션 초안

**본문 그림 1.** GC10-DET(a)와 NEU-DET(b)의 라벨 이미지 비율에 따른 development mAP@0.50:0.95. x좌표는 train acquisition pool의 실제 누적 라벨 이미지 비율이며, 표기 눈금은 약 10·15·20·25·30·35%다. GC10의 실제 누적 이미지 수는 160·241·321·401·481·561장, NEU는 126·189·252·315·377·440장이다. 각 선은 5 acquisition seed 평균, 옅은 띠는 seed 간 표본 SD다. R0 모델은 세 방법에 공유된다. DINOv2-FF는 F518 입력 특징을 사용한다. Full-data 점선은 각 데이터셋 train 전체로 학습한 **1회** 참고값이다. 두 패널은 공통 15–40% y축을 사용한다. 이 그림은 test 결과가 아니다.

**검토용 대안 그림.** 그림 1과 동일한 수치이며 GC10 16–31%, NEU 27–41%의 서로 다른 y축을 사용한다. 각 데이터셋 안의 방법 간 차이를 더 자세히 보지만, 패널 사이 기울기/높이 비교에는 본문 공통 축보다 부적합하다.

**본문 표 1.** `NAULC_MAIN_TABLE.md`의 한국어 캡션을 사용한다. nAULC의 적분 정의와 exact sign-flip/Holm 결과는 기존 `05_statistics/06_STATISTICAL_TESTS.md`에 둔다. 표의 숫자는 소수 둘째 자리 표시용 반올림뿐이다.

**nAULC 검토용 그림.** 방법별 development nAULC의 5개 acquisition seed 원자료(작은 빈 마커)와 평균±표본 SD(큰 마커·오차막대). 패널별 y축 범위가 다르며 방법 간 seed 연결선은 표시하지 않는다. 점과 오차막대는 통계적 유의성 표지가 아니다.

**본문 그림 2.** 약 35% 라벨 이미지 예산(R5)에서 DINOv2-FF와 Random의 development 클래스별 AP@0.50:0.95 평균 차이. 막대는 5 acquisition seed 평균의 차이(%p)이며 0선이 차이 없음이다. GC10 10개·NEU 6개 클래스를 모두 제시한다. 클래스별 차이는 기술적 분석이며 개별 유의성 검정을 뜻하지 않는다.

**부록 그림.** GC10-DET와 NEU-DET의 R5 development 클래스별 절대 AP@0.50:0.95. 각 마커는 5 seed 평균, 가로 오차막대는 표본 SD다. 데이터셋별 x축 상한은 다르지만 두 축 모두 0부터 시작한다.

## 선행연구와의 관계

[Yang et al., PPAL, CVPR 2024, Figs. 4–5](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_Plug_and_Play_Active_Learning_for_Object_Detection_CVPR_2024_paper.pdf)는 `Training Data (%)`에 COCO AP를 표시하고 Random·Entropy 곡선을 함께 놓는다. 같은 논문의 성능 곡선도 0부터 시작하지 않는 구간 축을 쓴다. [Lyu et al., Box-Level Active Detection, CVPR 2023, Figs. 1·3](https://openaccess.thecvf.com/content/CVPR2023/papers/Lyu_Box-Level_Active_Detection_CVPR_2023_paper.pdf)은 #images와 #boxes를 구분하고 FullSup 참조선을 사용한다. 이에 따라 우리 축을 `Labeled images (%)`로 정확히 명명하고 Full-data를 단일 참조선으로 둔다. 문헌은 이번 평균±SD 띠나 class-delta 막대의 통계적 정당성을 대신하지 않으며, 이 표현의 정의는 위 캡션에 명시했다. 더 자세한 원문 대조는 `01_literature/01_LITERATURE_FIGURE_CONVENTIONS.md`에 있다.
