# Original-final 공개 결과 파일 — 2026년 9월 13일

최신 해석은 [현재 실험 결과](CURRENT_EXPERIMENT_RESULT_20260913.md)에서 읽을 수 있습니다. 이 자료는 original-split exploratory completion의 관측 결과이며 독립적인 논문 주결과로 취급하지 않습니다. 기존 CSV와 그림의 scope/metadata는 그대로 보존했습니다.

## 표

- [three_way_original_final_summary.csv](../results/original_split_exploratory_completion_20260913/three_way_original_final_summary.csv) — 방법별 nAULC/R5 요약 6행.
- [original_final_naulc.csv](../results/original_split_exploratory_completion_20260913/original_final_naulc.csv) — acquisition seed별 nAULC 60행.
- [full_data_original_final_results.csv](../results/original_split_exploratory_completion_20260913/full_data_original_final_results.csv) — Full A/B 실제 4행 + 평균 2행.
- [original_final_budget_curves.csv](../results/original_split_exploratory_completion_20260913/original_final_budget_curves.csv) — dataset/method/round별 곡선 36행.
- [R5_vs_Full_descriptive_ratios.csv](../results/original_split_exploratory_completion_20260913/R5_vs_Full_descriptive_ratios.csv) — R5/Full descriptive ratio 6행.
- [original_final_statistics.csv](../results/original_split_exploratory_completion_20260913/original_final_statistics.csv) — 새 exploratory contrast 4행.
- [original_final_round_AB_means_360.csv](../results/original_split_exploratory_completion_20260913/original_final_round_AB_means_360.csv) — seed/round별 A/B 평균 360행.
- [original_final_paired_deltas.csv](../results/original_split_exploratory_completion_20260913/original_final_paired_deltas.csv) — paired seed 차이 40행.
- [entropy_original_final_results.csv](../results/original_split_exploratory_completion_20260913/entropy_original_final_results.csv) — 신규 Entropy 평가 200행.
- [per_class_original_final_results.csv](../results/original_split_exploratory_completion_20260913/per_class_original_final_results.csv) — 클래스별 관측 결과 1,632행.

## 그림

- [figures/GC10-DET_original_final_budget_curve.png](../results/original_split_exploratory_completion_20260913/figures/GC10-DET_original_final_budget_curve.png)
- [figures/NEU-DET_original_final_budget_curve.png](../results/original_split_exploratory_completion_20260913/figures/NEU-DET_original_final_budget_curve.png)
- [figures/three_way_original_final_budget_curves.pdf](../results/original_split_exploratory_completion_20260913/figures/three_way_original_final_budget_curves.pdf)
- [figures/three_way_original_final_budget_curves.png](../results/original_split_exploratory_completion_20260913/figures/three_way_original_final_budget_curves.png)
- [figures/three_way_original_final_budget_curves.svg](../results/original_split_exploratory_completion_20260913/figures/three_way_original_final_budget_curves.svg)

## 공개 표로 재검산

```sh
python scripts/verify_original_final_20260913.py
```

표준 Python만으로 A/B 평균 이후의 곡선→nAULC→3-way 요약, Full 평균, R5 비율, paired delta, exact sign-flip/Holm을 확인합니다. 모델 평가 재실행은 포함하지 않습니다. 선택한 파일의 SHA-256은 [공개 파일 목록](ORIGINAL_FINAL_PUBLIC_FILES_20260913.json)에 기록했습니다. CSV에서 mean 행은 추가 inference가 아니며 Full은 dataset별 실제 learner 2개입니다. 표의 0–1 값에 100을 곱해야 %가 됩니다.

Historical 자료와 development validation 자료는 각각 원래 분석 기록으로 보존합니다.
