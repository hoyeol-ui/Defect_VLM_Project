# 현재 실험 결과 — 2026년 9월 13일

이번 연구에서는 제한된 수의 라벨 이미지로 금속 표면 결함을 탐지할 때, 어떤 표본 선택 기준이 도움이 되는지 확인했습니다. Random 무작위 선택, 고정 DINOv2 특징의 farthest-first 선택, 탐지 confidence 기반 Entropy 선택을 같은 YOLOv8n 조건으로 비교했습니다. 전체 train pool을 사용하는 Full-data도 함께 확인했습니다.

## 완료한 실험

기존 Random/DINO 440회, Entropy 200회, Full-data 4회로 총 644회 학습을 완료했습니다. Entropy 표본 선택도 100/100회 끝났습니다. 신규 Entropy 200개와 Full-data 4개 checkpoint의 original-final 평가를 204/204개 모두 완료하고, 기존 Random/DINO 평가 결과를 재사용해 세 방법을 정리했습니다. 신규 평가 중 중복 실행·실패·재시도·새 학습·fine-tuning은 모두 0회입니다.

아래 original-final 수치는 기존 프로토콜 한 사이클을 끝까지 확인한 **original-split exploratory completion 결과**입니다. 독립적인 논문 주결과로 취급하지 않으며, 기존 historical 결과와 development validation을 구분해 제시합니다.

## 전체 학습곡선과 마지막 R5

아래 nAULC는 R0–R5의 mAP50–95 곡선을 정규화한 값입니다. CSV는 0–1, 표는 % 단위입니다. A/B를 먼저 평균하고 acquisition seed 10개를 비교합니다. 644회를 독립 표본 수로 사용하지 않았습니다.

|Dataset|Random (%)|DINO (%)|Entropy (%)|
|---|---:|---:|---:|
|GC10-DET|24.464|25.045|24.933|
|NEU-DET|35.195|35.773|35.511|

마지막 R5의 original-final mAP50–95는 다음과 같습니다.

|Dataset|Random (%)|DINO (%)|Entropy (%)|
|---|---:|---:|---:|
|GC10-DET|26.744|27.335|27.960|
|NEU-DET|38.023|38.297|38.501|

Random, DINO, Entropy 세 방법을 비교했을 때, 두 데이터셋 모두 전체 학습곡선에서는 DINO가 가장 높았고, 마지막 R5에서는 Entropy가 가장 높았습니다. 다만 방법 간 차이는 크지 않았으며, DINO와 Entropy 중 한 방법이 계속 우세하게 나타나지는 않았습니다.

## 전체 train pool을 사용한 참고 성능

Full-data original-final mAP50–95입니다. 각 데이터셋의 A/B 실제 학습 두 번(n=2)을 그대로 보여주고, mean은 두 결과의 산술평균입니다.

|Dataset|A (%)|B (%)|mean (%)|
|---|---:|---:|---:|
|GC10-DET|31.026|33.049|32.037|
|NEU-DET|42.181|43.749|42.965|

R5는 GC10 561/1605장, NEU 441/1259장을 사용했습니다. 약 35%의 train image를 사용했을 때 Full-data 대비 약 83–90%의 성능이 관측됐습니다. 아래는 R5 mAP50–95를 Full A/B 평균으로 나눈 descriptive ratio입니다.

|Dataset|Random (%)|DINO (%)|Entropy (%)|
|---|---:|---:|---:|
|GC10-DET|83.48|85.32|87.27|
|NEU-DET|88.50|89.13|89.61|

Full-data와 AL R5 사이에는 절대 성능 차이가 남아 있습니다. 이미지 수 비율과 실제 라벨링 비용은 같지 않으므로 비용 절감률로 해석하지 않습니다.

## 차이를 어느 정도로 해석할 수 있나

|Dataset / contrast|차이 (pp)|95% CI (pp)|Holm p|
|---|---:|---:|---:|
|GC10-DET / Entropy - Random|+0.469|[+0.215, +0.733]|0.054688|
|GC10-DET / DINO - Entropy|+0.113|[-0.258, +0.481]|0.529297|
|NEU-DET / Entropy - Random|+0.316|[-0.057, +0.671]|0.416016|
|NEU-DET / DINO - Entropy|+0.262|[-0.040, +0.559]|0.416016|

이 표는 새 original-final exploratory contrast 4개입니다. Seed별 A/B 평균을 짝지어 비교하고, paired bootstrap 100,000회와 exact sign-flip 1,024패턴, 이 4개 비교 내 Holm 보정을 사용했습니다. Holm-adjusted p < .05는 0개입니다. DINO와 Entropy 중 한 방법이 통계적으로 일관되게 우세했다고 보기는 어렵습니다. 두 방법의 동등성을 확인한 것은 아닙니다.

기존 historical DINO−Random nAULC 차이 GC10 +0.581pp, NEU +0.578pp는 당시 결과로 보존합니다. 기존 통계 family와 새 exploratory 비교를 하나의 동일 검정으로 합치지 않았습니다.

## Development validation은 별도 결과

기존 validation nAULC는 GC10 Random/DINO/Entropy 27.002/27.393/27.649%, NEU 35.189/35.981/35.902%입니다. Development validation Full-data 평균은 GC10 34.717%, NEU 42.896%입니다. 이 값들은 위 original-final 수치와 다른 평가 기준이므로 별도로 읽어야 합니다.

이번 미팅에서는 지금까지 나온 결과를 바탕으로, 현재의 비교 범위와 결과 해석으로 논문을 정리해도 되는지, 추가 실험이 더 필요한지 교수님 의견을 듣고 싶습니다.

## 근거와 재검산

[Original-final 공개 CSV·그림 안내](ORIGINAL_FINAL_PUBLIC_RESULTS_20260913.md) · [Development validation](../results/development_20260913/EXPERIMENT_RESULT_BRIEF.md) · [Historical 결과](../results/thesis_final/README.md).

`python scripts/verify_original_final_20260913.py`로 공개 표의 산술과 집계를 다시 확인할 수 있습니다. 이번 업데이트에서는 새 training·inference·fine-tuning을 실행하지 않았습니다.
