# 본문용 nAULC 표 초안

**표 1.** GC10-DET와 NEU-DET의 development nAULC. 방법별 수치는 5개 acquisition seed의 평균 ± 표본 표준편차(%)이며, Δ는 같은 seed끼리 짝지은 DINOv2-FF − Random 평균 차이(%p)다. DINOv2-FF는 F518 입력 특징을 사용한다. 사전 정의된 1차 분석은 아직 봉인된 test에 대해 수행해야 한다.

| Dataset | Random | DINOv2-FF | Entropy | Δ DINOv2-FF − Random |
|---|---:|---:|---:|---:|
| GC10-DET | 21.85 ± 0.44 | 23.69 ± 0.33 | 21.73 ± 0.41 | +1.83 |
| NEU-DET | 31.84 ± 0.81 | 31.92 ± 0.54 | 31.76 ± 0.75 | +0.09 |

수치 출처: `02_tables/03_FINAL_NAULC_RESULTS.csv`와 `05_statistics/PAIRED_NAULC_SUMMARY.csv`. 표시는 소수 둘째 자리로 반올림했다.
