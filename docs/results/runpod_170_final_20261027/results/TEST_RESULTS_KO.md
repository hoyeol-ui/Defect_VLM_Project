# 로컬170모델 최종 test 결과

고정된 test 입력: GC10-DET343장, NEU-DET270장. 아래는 mAP50–95(%) 평균±표본표준편차,5 paired seeds.

|데이터셋|예산|Random|DINOv2|Entropy|
|---|---:|---:|---:|---:|
|GC10-DET|10%|19.83 ± 0.60|19.83 ± 0.60|19.83 ± 0.60|
|GC10-DET|15%|22.29 ± 0.91|23.79 ± 1.14|21.95 ± 1.95|
|GC10-DET|20%|23.97 ± 1.49|24.38 ± 1.93|24.78 ± 1.11|
|GC10-DET|25%|23.75 ± 1.83|25.53 ± 1.76|25.21 ± 1.21|
|GC10-DET|30%|24.48 ± 1.39|26.05 ± 1.87|25.94 ± 1.07|
|GC10-DET|35%|24.74 ± 0.37|26.97 ± 1.22|26.21 ± 0.92|
|NEU-DET|10%|26.98 ± 2.22|26.98 ± 2.22|26.98 ± 2.22|
|NEU-DET|15%|28.66 ± 0.94|28.55 ± 1.56|28.01 ± 1.73|
|NEU-DET|20%|29.76 ± 1.23|29.52 ± 1.15|29.23 ± 1.57|
|NEU-DET|25%|30.13 ± 1.09|30.31 ± 2.20|30.14 ± 1.80|
|NEU-DET|30%|31.47 ± 0.87|30.86 ± 0.62|30.87 ± 0.59|
|NEU-DET|35%|31.18 ± 1.72|31.11 ± 0.97|30.62 ± 1.10|

GC10-DET Full: 28.56 ± 0.80%


NEU-DET Full: 34.72 ± 1.16%


nAULC는 실제 이미지 수 구간의 사다리꼴 적분/전체 구간 길이로 계산.4개 paired 비교는 양측 exact sign-flip+Holm. n=5에서 유의성이나 동등성을 과장하지 않는다.

Local Windows RTX3070 CUDA12.6 evaluation differs from Linux Blackwell CUDA12.9 training; no claim of bitwise device equivalence.
Canonical source has documented historical exposure; this test is held out from the new170 training, not a never-seen external dataset.
Five paired seeds cannot yield two-sided exact sign-flip p<.05; do not alter tests to pursue significance.
