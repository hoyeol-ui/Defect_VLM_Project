# Final Experiment Academic Evaluation — 2026-08-28

## 한 문장 판정

이 실험은 **confirmatory primary 기준으로 성공**했다. Frozen DINO farthest-first selection은 Random보다 GC10-DET와 NEU-DET 모두에서 10–35% budget 구간의 locked-final mAP50-95 AULC를 약 **+0.58 percentage point** 높였다. 그러나 이 결과는 크기가 작고, final-budget 우월성은 불명확하며, NEU에서는 최약 class가 일관되게 악화됐다. 따라서 가장 강한 논문 주장은 “DINO가 보편적으로 우월하다”가 아니라 **selection이 평균 downstream learning으로 번역될 수 있지만, 그 번역은 budget과 class composition에 따라 이득과 손실을 동시에 만든다**는 것이다.

## 1. Confirmatory primary

| Dataset | Random AULC | Frozen DINO AULC | DINO − Random | 95% acquisition bootstrap CI | seed direction | exact sign-flip p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| GC10-DET | 0.244643 | 0.250454 | +0.005812 (+0.581 pp) | [+0.003970, +0.007576] | 10/10 positive | 0.001953 | 0.003906 |
| NEU-DET | 0.351950 | 0.357731 | +0.005782 (+0.578 pp) | [+0.003100, +0.008119] | 9/10 positive | 0.003906 | 0.003906 |

- Primary unit은 dataset별 acquisition seed `n=10`이며, 각 acquisition 안에서 training seed A/B를 먼저 평균했다.
- 100,000회 paired acquisition bootstrap, exact `2^10` sign-flip test, 두 dataset Holm correction을 사전 고정된 방식대로 적용했다.
- GC10와 NEU를 `n=20`으로 합치지 않았다.
- 효과의 상대 크기는 Random AULC 대비 GC10 약 2.38%, NEU 약 1.64%다. 통계적으로는 명확하지만 절대 크기는 **modest**다.
- 판정은 protocol-bounded Case A다. 두 dataset에서 positive paired downstream difference가 확립됐지만, DINO의 보편적 우월성은 아니다.

## 2. Learning-curve 해석

효과는 마지막 budget 한 점보다 **초·중반 label efficiency**에서 강하다.

| Dataset | R1 (15%) | R2 (20%) | R3 (25%) | R4 (30%) | R5 (35%) |
|---|---:|---:|---:|---:|---:|
| GC10-DET | +0.246 pp | +1.142 pp | +0.522 pp | +0.700 pp | +0.590 pp |
| NEU-DET | +0.812 pp | +1.091 pp | +0.388 pp | +0.463 pp | +0.274 pp |

- 두 dataset 모두 R2에서 차이가 가장 크고 이후 감소한다.
- R5 final-budget mAP50-95 CI는 GC10 `[-0.010,+1.155] pp`, NEU `[-0.225,+0.743] pp`로 0을 포함한다.
- 따라서 허용되는 주장은 “10–35% 구간의 평균 학습 효율(AULC)이 향상됐다”다.
- “35%에서 확실히 더 높은 최종 성능을 얻었다”는 주장은 허용되지 않는다.

## 3. Selection → composition → downstream translation

### GC10-DET

R5에서 Frozen DINO는 Random보다 rare-target image를 평균 14.7장 더 포함했고, total instances는 46.6개, normalized class-incidence entropy는 0.0207 증가했다. 이에 대응해 frozen rare-class macro AULC는 `+0.673 pp`, CI `[+0.302,+1.068] pp`, 9/10 positive였다.

그러나 final class AP는 균일하지 않았다. `crease +3.975 pp`, `welding_line +5.538 pp`, `crescent_gap +2.713 pp`인 반면 `silk_spot −3.353 pp`, `inclusion −1.176 pp`, `rolled_pit −1.254 pp`였다. Rare composition을 더 확보했다는 사실이 모든 rare class의 downstream 향상을 보장하지는 않았다.

### NEU-DET

R5에서 Frozen DINO의 평균 image incidence는 Random 대비 다음처럼 이동했다.

| Class | Random mean | DINO mean | DINO − Random |
|---|---:|---:|---:|
| crazing | 73.8 | 32.3 | −41.5 |
| inclusion | 93.3 | 98.4 | +5.1 |
| patches | 83.3 | 81.9 | −1.4 |
| pitted_surface | 79.0 | 110.6 | +31.6 |
| rolled-in_scale | 68.8 | 33.7 | −35.1 |
| scratches | 75.1 | 132.7 | +57.6 |

Normalized class-incidence entropy는 `−0.061`, total instances는 `−54.9`였다. 이 composition shift와 final class AP 방향도 대응했다: `scratches +5.449 pp`, `pitted_surface +2.236 pp`, `crazing −3.118 pp`, `rolled-in_scale −3.336 pp`.

NEU worst-class endpoint는 거의 모든 curve에서 `crazing`이었고, DINO − Random은 AULC `−1.488 pp` (CI `[−1.978,−1.002]`, 0/10 positive), final `−3.118 pp` (CI `[−3.822,−2.372]`, 0/10 positive)였다. 이는 formal primary를 뒤집는 결과는 아니지만, **평균 성능 향상과 worst-class safety가 동시에 성립하지 않았다**는 중요한 secondary finding이다.

## 4. 440회가 필요했는가

결론은 **과학적으로 정당화됐다**다. 더 적은 실행으로 같은 숫자를 우연히 볼 수는 있었지만, 같은 수준의 주장을 방어하기는 어려웠다.

- Acquisition seed 10개는 selector가 어떤 pool realization을 만나느냐에 따른 변이를 primary uncertainty로 직접 추정하게 했다.
- Training seed 2개는 learner noise를 acquisition 안에서 평균하게 했다. A/B AULC의 평균 절대 차이는 GC10 Random `0.440 pp`, GC10 DINO `0.368 pp`, NEU Random `0.242 pp`, NEU DINO `0.266 pp`였다. 이는 primary effect `약 0.58 pp`에 비해 무시할 수 없는 크기다.
- Training seed별 사후 진단에서도 평균 effect는 모두 양수였다: GC10 A `+0.392 pp` (9/10), GC10 B `+0.770 pp` (10/10), NEU A `+0.518 pp` (9/10), NEU B `+0.638 pp` (9/10). 즉 결론은 특정 learner seed 하나에만 의존하지 않았다.
- Shared Round0를 실제로 한 번만 평가해 480이 아니라 440 scientific cells로 줄인 설계도 적절했다.
- 누적 training runtime은 67.937 GPU-hours, locked-final evaluation은 72.10분이었다. 이 비용은 작은 effect를 과장하기 위한 비용이 아니라, 작은 effect가 stochastic artifact가 아님을 확인한 비용이다.

## 5. 연구 성공도

| 평가 축 | 판정 | 이유 |
|---|---|---|
| Frozen confirmatory primary | 성공 | 두 dataset 모두 CI > 0, Holm p=0.003906 |
| 실용적 effect magnitude | 제한적 성공 | +0.58 pp AULC; final-budget CI는 0 포함 |
| Acquisition reproducibility | 강함 | GC10 10/10, NEU 9/10 |
| Learner-seed robustness | 양호 | A/B 각각에서도 방향 유지, 단 A/B 차이는 작지 않음 |
| Subgroup safety | 실패/경고 | NEU worst class 0/10, class AP −3.118 pp |
| 새 selector 알고리즘 novelty | 약함 | global frozen DINO farthest-first는 단순한 diversity baseline |
| 석사학위 연구 가치 | 강함 | prefreeze, independent final lock, paired seed hierarchy, full curve, raw registry, negative subgroup 보존 |
| SOTA algorithm paper 가능성 | 낮음 | detector-aware uncertainty/diversity 및 다수 강한 baseline이 없음 |
| empirical/reproducibility paper 가능성 | 있음 | 산업 결함 detection에서 translation과 class-composition trade-off를 통제된 반복으로 제시 |

## 6. 선행연구 대비 위치

- Munjal et al. (CVPR 2022)은 동일 조건에서도 AL의 Random 대비 gain이 일관되지 않고 stochasticity가 결론을 흔들 수 있음을 보였다. 이번 연구는 그 문제의식에 맞춰 acquisition과 learner seed를 분리했으며, 이번 특정 protocol에서는 small but reproducible positive effect를 얻었다.
- PPAL (CVPR 2024)은 detection의 classification/localization difficulty와 multi-object category-conditioned similarity를 직접 모델링한다. 이번 Frozen DINO는 global image representation diversity이므로 알고리즘 정교함이나 SOTA를 경쟁하는 연구가 아니다.
- Gonzalez et al. (2026)은 실제 산업 porosity dataset에서 uncertainty, structured diversity, random을 결합하고 6 repetitions와 independent test set으로 label-efficiency를 평가했다. 이번 연구는 public industrial defect datasets 두 개, acquisition 10 × learner 2, 6-point curve를 통해 variance decomposition과 subgroup translation을 더 엄격하게 다루지만, annotation-time cost와 proprietary real-line validation은 제공하지 않는다.
- Foundation-model feature가 detection에 유용할 수 있다는 연구는 존재하지만, 그것이 곧 AL selection의 class-safe downstream gain을 보장하지 않는다. 이번 NEU 결과가 그 간극을 직접 보여준다.

## 7. 논문에서 허용되는 핵심 주장

> Under a prospectively frozen 10–35% image-budget protocol, frozen DINO diversity selection produced a small but acquisition-reproducible improvement in downstream detection AULC over Random on both GC10-DET and NEU-DET. However, the gain was concentrated at early-to-mid budgets and did not guarantee final-budget or worst-class improvement; on NEU-DET, composition shifted away from crazing and rolled-in-scale, producing a marked worst-class loss.

한국어로는 다음이 가장 안전하다.

> 사전 고정된 산업 결함 객체검출 AL protocol에서 Frozen DINO diversity는 Random보다 두 데이터셋의 downstream AULC를 작지만 재현성 있게 향상시켰다. 그러나 이득은 초·중반 budget에 집중됐고 class composition을 편향시켜 최약 클래스 성능을 악화시킬 수 있었다. 따라서 selection-stage의 대표성·발견 신호, 평균 downstream 효율, class-level safety는 분리해 평가해야 한다.

## 8. 금지하거나 약화해야 할 주장

- “DINO가 Random보다 항상 우월하다.”
- “35% budget에서 확실히 더 높은 final performance를 달성했다.”
- “모든 class 또는 rare class가 개선됐다.”
- “annotation cost를 절감했다.” 실제 annotation time/cost를 측정하지 않았다.
- “산업 현장 전체에 일반화된다.” 두 public dataset, YOLOv8n, 한 selector family에 한정된다.
- “새로운 SOTA selector를 제안했다.” 이 연구의 기여는 selector novelty보다 controlled empirical evidence다.

## 9. 남은 한계와 다음 실험 우선순위

추가 대규모 training을 즉시 할 필요는 없다. 먼저 이 결과로 논문을 완성하는 것이 우선이다. 후속 실험이 필요하다면 순서는 다음이 합리적이다.

1. 동일 440을 반복하지 말고, NEU class-composition safeguard를 갖춘 stratified/coverage-aware DINO를 별도 future-work protocol로 설계한다.
2. Entropy/Core-set 또는 detector-aware baseline을 소규모가 아니라 새로 동결된 독립 비교로 추가해야만 algorithm-paper 주장이 가능하다.
3. 실제 annotation time을 측정하지 않는 한 cost-saving claim은 보류한다.
4. detector family 또는 dataset을 하나 더 추가하면 external validity가 개선되지만, 현재 석사 논문의 완료 조건은 아니다.

## 10. 무결성과 데이터 주석 메모

- Training audit: `440/440`, missing/duplicate/corrupt/mismatch/seed replacement/config change/boundary violation `0`.
- Final evaluation audit: `PASS`, `440/440`, development substitution `0`, post-final retuning `0`.
- NEU locked-final의 `crazing_120.xml`에는 동일 bounding box가 한 번 중복되어 있다. 캐시 raw label은 830 lines, unique label은 829이며, Ultralytics가 매 평가에서 동일 duplicate 1개를 제거했다. 모든 model/strategy에 공통인 deterministic handling이므로 paired contrast를 차별적으로 편향시키지 않지만, dataset-quality limitation으로 공개한다.

## 11. 정본 파일

- `CONFIRMATORY_FINAL_ANALYSIS_REPORT.md`: 결과 표와 protocol-bounded 해석
- `RAW_CONFIRMATORY_RESULTS_MASTER_FINAL_20260828.md`: raw 결과 정본과 SHA-256 registry
- `results/raw_locked_final_metrics_conceptual_480.csv`: strategy × budget conceptual rows
- `results/curve_level_aulc_80.csv`: 80개 완전한 6-point curve
- `results/primary_acquisition_deltas_20.csv`: primary 원 단위 20개
- `results/selection_composition_raw_240.csv`: selection/composition 원 상태
- `figures/confirmatory_primary_results.*`: primary publication figure
- `figures/confirmatory_classwise_results.*`: class-wise descriptive figure

## 참고한 주요 선행연구

- Munjal et al., *Towards Robust and Reproducible Active Learning Using Neural Networks*, CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Munjal_Towards_Robust_and_Reproducible_Active_Learning_Using_Neural_Networks_CVPR_2022_paper.html
- Yang et al., *Plug and Play Active Learning for Object Detection*, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Plug_and_Play_Active_Learning_for_Object_Detection_CVPR_2024_paper.html
- Gonzalez et al., *Active learning for industrial defect detection: a study on hybrid sampling strategies*, IJAMT 2026: https://link.springer.com/article/10.1007/s00170-025-17378-7
- Han et al., *Few-Shot Object Detection with Foundation Models*, CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Han_Few-Shot_Object_Detection_with_Foundation_Models_CVPR_2024_paper.html
