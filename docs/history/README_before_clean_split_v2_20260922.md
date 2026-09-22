# 소량 라벨 환경에서 특징 기반 능동학습을 활용한 금속 표면 결함 객체 탐지

라벨을 붙일 수 있는 이미지 수가 제한될 때, 어떤 이미지를 고르면 금속 표면 결함을 더 잘 탐지할 수 있을까요? GC10-DET와 NEU-DET에서 **Random**, **Frozen DINOv2 feature + farthest-first**, **Detection-confidence Entropy**를 같은 YOLOv8n 학습 조건으로 비교했습니다. **Full-data**는 전체 train pool을 사용했을 때의 참고 성능입니다.

## 현재 상태 — 2026-09-13

|실험|완료 상태|
|---|---|
|Historical Random / DINO|440 physical training COMPLETE, 기존 평가 재사용|
|Entropy|200/200 training, 100/100 acquisition COMPLETE|
|Full-data|4/4 training COMPLETE: GC10 A/B, NEU A/B|
|Total physical training|644 = 440 + 200 + 4|
|Original-final new evaluation|204/204 COMPLETE: Entropy 200 + Full-data 4|

신규 평가의 중복 실행·실패·재시도는 모두 0회이고, 평가 중 새 학습과 fine-tuning도 0회입니다. 공통 R0는 재사용했으며 새 학습으로 중복 계산하지 않았습니다. 이번 문서 업데이트에서는 training·inference·fine-tuning을 추가로 실행하지 않았습니다.

아래 original-final 수치는 기존 프로토콜 한 사이클을 끝까지 확인한 **original-split exploratory completion 결과**입니다. 독립적인 논문 주결과로 취급하지 않으며, 기존 historical 결과와 development validation을 구분해 제시합니다.

## Original-final 3-way result

R0–R5 전체 곡선을 요약한 nAULC입니다. CSV의 0–1 값을 아래에서는 100배하여 %로 표시합니다. 각 seed/round에서 learner A/B를 먼저 평균하고, acquisition seed 10개를 비교 단위로 사용했습니다. 644회는 학습 횟수이며 통계 표본 수가 아닙니다.

|Dataset|Random (%)|DINO (%)|Entropy (%)|
|---|---:|---:|---:|
|GC10-DET|24.464|25.045|24.933|
|NEU-DET|35.195|35.773|35.511|

두 데이터셋 모두 관측 평균은 DINO > Entropy > Random입니다. [6행 요약 CSV](results/original_split_exploratory_completion_20260913/three_way_original_final_summary.csv) · [60개 seedwise nAULC](results/original_split_exploratory_completion_20260913/original_final_naulc.csv) · [budget curve 그림](results/original_split_exploratory_completion_20260913/figures/three_way_original_final_budget_curves.png).

## R5 result

마지막 R5의 original-final mAP50–95입니다. GC10은 561/1605장(34.95%), NEU는 441/1259장(35.03%)을 사용했습니다.

|Dataset|Random (%)|DINO (%)|Entropy (%)|
|---|---:|---:|---:|
|GC10-DET|26.744|27.335|27.960|
|NEU-DET|38.023|38.297|38.501|

마지막 R5에서는 두 데이터셋 모두 Entropy > DINO > Random으로 관측됐습니다. 전체 곡선의 순위와 구분해서 읽어야 합니다.

## Full-data result

전체 train pool로 학습한 original-final mAP50–95 참고 성능입니다. Dataset별 A/B 실제 두 번의 학습 결과(**n=2**)와 산술평균이며, AL의 acquisition seed 10개처럼 취급하지 않습니다.

|Dataset|A (%)|B (%)|mean (%)|
|---|---:|---:|---:|
|GC10-DET|31.026|33.049|32.037|
|NEU-DET|42.181|43.749|42.965|

[Full A/B 원값과 평균](results/original_split_exploratory_completion_20260913/full_data_original_final_results.csv)

R5 / Full A/B 평균의 descriptive ratio는 다음과 같습니다.

|Dataset|Random (%)|DINO (%)|Entropy (%)|
|---|---:|---:|---:|
|GC10-DET|83.48|85.32|87.27|
|NEU-DET|88.50|89.13|89.61|

약 35%의 train image를 사용한 R5에서 Full-data 대비 약 83–90% 수준의 성능이 관측됐습니다. Full-data와 AL R5 사이에는 절대 성능 차이가 남아 있습니다. 이미지 수 비율을 실제 라벨링 비용 절감률로 환산하지 않습니다. [비율 원표](results/original_split_exploratory_completion_20260913/R5_vs_Full_descriptive_ratios.csv).

## Development validation result

기존 development validation 결과는 별도로 보존합니다. 아래 nAULC와 Full 참고값은 validation 기준이며 위 original-final 수치와 섞어 비교하지 않습니다.

|Dataset|Random nAULC (%)|DINO nAULC (%)|Entropy nAULC (%)|Full A/B mean mAP50–95 (%)|
|---|---:|---:|---:|---:|
|GC10-DET|27.002|27.393|27.649|34.717|
|NEU-DET|35.189|35.981|35.902|42.896|

![Three-way development validation curves](results/development_20260913/figures/figure_2_development_curves.png)

선은 seed 10개의 A/B 평균을 다시 평균한 값이고 음영은 seed SD입니다. Validation은 best checkpoint 선택에 사용됐습니다. [Development 결과 기록](results/development_20260913/EXPERIMENT_RESULT_BRIEF.md) · [paired 비교](results/development_20260913/tables/paired_nAULC_comparison.md) · [Full validation 원표](results/development_20260913/tables/full_data_reference_summary.md) · [선택 구성 진단](results/development_20260913/tables/selection_composition_per_query.csv).

## Result interpretation

Random, DINO, Entropy 세 방법을 비교했을 때, 두 데이터셋 모두 전체 학습곡선에서는 DINO가 가장 높았고, 마지막 R5에서는 Entropy가 가장 높았습니다. 다만 방법 간 차이는 크지 않았으며, DINO와 Entropy 중 한 방법이 계속 우세하게 나타나지는 않았습니다.

새 original-final exploratory contrast 4개의 결과는 아래와 같습니다. A/B 평균 후 seed n=10을 기준으로 paired bootstrap 100,000회와 exact sign-flip 1,024패턴을 사용했고, Holm 보정은 이 4개 비교에만 적용했습니다.

|Dataset / contrast|차이 (pp)|95% CI (pp)|Holm p|
|---|---:|---:|---:|
|GC10-DET / Entropy - Random|+0.469|[+0.215, +0.733]|0.054688|
|GC10-DET / DINO - Entropy|+0.113|[-0.258, +0.481]|0.529297|
|NEU-DET / Entropy - Random|+0.316|[-0.057, +0.671]|0.416016|
|NEU-DET / DINO - Entropy|+0.262|[-0.040, +0.559]|0.416016|

Holm-adjusted p < .05인 비교는 0개입니다. DINO와 Entropy 중 한 방법이 통계적으로 일관되게 우세했다고 보기는 어렵고, 동등성을 확인한 것도 아닙니다. [통계 CSV](results/original_split_exploratory_completion_20260913/original_final_statistics.csv).

기존 historical DINO−Random 결과는 GC10 +0.581pp, NEU +0.578pp입니다. 기존 분석의 family와 통계 결과는 그대로 보존했으며, 새 exploratory contrast의 검정과 합치지 않았습니다. [Historical 결과](results/thesis_final/README.md) · [기존 primary 표](results/thesis_final/tables/locked_final_primary.csv).

## Files / reproduction

1. [최신 결과 요약](docs/CURRENT_EXPERIMENT_RESULT_20260913.md)
2. [Original-final 공개 표 안내](docs/ORIGINAL_FINAL_PUBLIC_RESULTS_20260913.md)와 [그림](results/original_split_exploratory_completion_20260913/figures/)
3. [Development 분석](results/development_20260913/EXPERIMENT_RESULT_BRIEF.md)와 [원표](results/development_20260913/tables/)
4. [Historical Random/DINO 결과](results/thesis_final/README.md)
5. [Controlled 440 protocol](docs/collaboration/CONTROLLED_440_PROTOCOL.md) / [Entropy protocol](entropy_extension_v1/official_release_v1/protocol.json)
6. [전체 파일 안내](THESIS_FILE_INDEX.md) / [재현성 문서](docs/collaboration/REPRODUCIBILITY.md)

```sh
python scripts/verify_original_final_20260913.py
python scripts/verify_public_results.py
python scripts/verify_development_20260913.py
```

첫 명령은 공개 CSV로 nAULC·R5·Full 평균·비율·paired 차이와 exact p/Holm을 재검산합니다. GPU, checkpoint, raw dataset을 읽지 않습니다. 기존 두 verifier는 각각 historical 및 development 공개 자료를 확인합니다. Development의 당시 navigation 해시는 Git commit `077503ca`의 원본과 대조하므로 해당 이력이 포함된 Git clone이 필요합니다. 원래 날짜의 문서와 실험 기록은 당시 범위대로 보존합니다.
