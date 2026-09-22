# clean split v2: development 통계 재계산

**범위:** 현 파일은 `ANNOTATION_PATCH_v2_0_2`의 봉인된 **development** 평가를 5 acquisition seed 단위로 재계산한 탐색적 결과다. `reference/FINAL_MAIN_EXPERIMENT_PROTOCOL_v2_0_KO.md` 9절의 1차 분석 대상은 추후 withheld **test**다. 아래 p값을 최종 test 가설검정 결과로 쓰지 않는다.

## 지표와 단위

각 seed의 R0–R5 mAP@0.50:0.95를 `sealed_dev_metrics.json`에서 읽었다. 실제 누적 이미지 수 `n_r`를 사용하여

`nAULC = Σ[r=1..5] (n_r−n_(r−1)) × (AP_r+AP_(r−1))/2 ÷ (n_5−n_0)`

로 계산했다. AP는 0–1 수치이고, 아래 nAULC는 100을 곱한 **%**다. 두 nAULC의 차이는 **%p**다. Full-data AP로 나누지 않았다. shared R0는 각 seed의 세 방법 곡선에 동일한 1개 모델을 참조했다. 빠진 round 또는 제외한 seed는 없다.

| 데이터셋 | Random nAULC mean±SD | DINO-F518 | Entropy |
|---|---:|---:|---:|
| GC10-DET | 21.85±0.44 | 23.69±0.33 | 21.73±0.41 |
| NEU-DET | 31.84±0.81 | 31.92±0.54 | 31.76±0.75 |

절대 평균의 95% CI는 5개 seed의 sample SD(`ddof=1`)에 `t(0.975, df=4)=2.7764451052`를 곱한 `mean ± t × SD/√5`다. seed 반복·고정 split/learner에 조건부인 기술적 구간이며 범용 데이터셋 모집단의 일반화 보증이 아니다. 원자료와 구간은 `03_FINAL_NAULC_RESULTS.csv`, `RAW_SEED_NAULC.csv`에 있다.

## seed를 맞춘 nAULC 차이

| 데이터셋 | 대비 | 평균 Δ (%p) | 5 seed 승/패 | paired bootstrap 95% 구간 (%p) | exact 양측 p | Holm p |
|---|---|---:|---:|---:|---:|---:|
| GC10 | DINO−Random | +1.832 | 5/0 | [1.480, 2.214] | 0.0625 | 0.25 |
| GC10 | DINO−Entropy | +1.952 | 5/0 | [1.721, 2.223] | 0.0625 | 0.25 |
| NEU | DINO−Random | +0.086 | 3/2 | [−0.314, 0.486] | 0.7500 | 1.00 |
| NEU | DINO−Entropy | +0.159 | 4/1 | [−0.349, 0.561] | 0.5625 | 1.00 |
| GC10 | Entropy−Random (보조) | −0.120 | 2/3 | [−0.398, 0.093] | 0.5625 | 적용 안 함 |
| NEU | Entropy−Random (보조) | −0.073 | 2/3 | [−0.517, 0.345] | 0.7500 | 적용 안 함 |

정확 검정은 각 대비의 5 paired 차이에 가능한 `2^5=32` 부호를 모두 적용하고, `|mean(sign×Δ)| ≥ |observed mean(Δ)|`의 비율을 쓴 양측 sign-flip이다. 부호 교환 가능성, 특히 차이 분포의 대칭성을 가정한다. raw p의 최소 가능값은 **2/32=0.0625**다. GC10/NEU의 DINO−Random 및 DINO−Entropy **네 대비를 하나의 Holm family**로 조정했다. 전체 dataset을 n=10으로 합치지 않았다. Entropy−Random의 p는 참고용이며 사전 정의 주요 family에 포함하지 않았다.

paired percentile bootstrap은 dataset별 고정 RNG seed GC10 `2026091912`, NEU `2026091913`, 대비별 100,000회 5개 seed pair의 복원 추출이다. 표의 구간은 평균 paired Δ의 2.5/97.5 percentile이다. n=5에서 구간이 0을 넘지 않더라도 **exact p 및 Holm 결과를 우회해 유의성을 주장하지 않는다.** 어느 주요 대비도 사전 기준 `p<0.05`에 도달하지 않았다. 비유의가 동등성을 입증하지도 않는다.

round별 raw/paired 차이, 모든 seed의 nAULC 차이, 중앙값·최소/최대·SD·정확 p는 `04_ROUNDWISE_RESULTS.csv`, `PAIRED_NAULC_RAW.csv`, `PAIRED_NAULC_SUMMARY.csv`에 있다. 코드 `run_analysis.py`가 CSV와 이 표의 수치 원천이다.
