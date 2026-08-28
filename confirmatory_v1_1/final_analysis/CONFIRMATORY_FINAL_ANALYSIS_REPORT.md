# Random vs Frozen DINO Controlled Confirmatory Final Report

- Protocol: `confirmatory-random-dino-v1.1`
- Primary unit: acquisition seed, dataset당 `n=10`
- Learner repeats: acquisition 안에서 training seed A/B 평균
- Primary: six-budget locked-final mAP50-95 AULC
- Raw final evaluations: `440/440`
- Missing/interpolated primary point: `0`
- Development substitution: `0`

## 결론

**Case A. 평가된 두 dataset/protocol에서 Frozen DINO의 positive paired downstream difference가 확립됐다.**

이 결론은 DINO의 보편적 우월성·열등성이나 두 방법의 통계적 동등성을 뜻하지 않는다. 평가 범위는 GC10-DET와 NEU-DET, 약 10–35% image budget, Frozen DINO global representation diversity, YOLOv8n cold-start protocol이다.

## Primary result

| Dataset | DINO − Random AULC | 95% acquisition bootstrap CI | wins/losses | exact p | Holm p |
|---|---:|---:|---:|---:|---:|
| GC10-DET | +0.581 pp | [+0.397 pp, +0.758 pp] | 10/0 | 0.001953 | 0.003906 |
| NEU-DET | +0.578 pp | [+0.310 pp, +0.812 pp] | 9/1 | 0.003906 | 0.003906 |

### Acquisition-seed raw paired effects

| Dataset | acquisition | seed | Random AULC | DINO AULC | DINO − Random |
|---|---:|---:|---:|---:|---:|
| GC10-DET | a00 | 118406135 | 0.242222 | 0.244148 | +0.193 pp |
| GC10-DET | a01 | 1544253486 | 0.243776 | 0.250479 | +0.670 pp |
| GC10-DET | a02 | 1207468566 | 0.255739 | 0.264369 | +0.863 pp |
| GC10-DET | a03 | 1047824994 | 0.237676 | 0.246299 | +0.862 pp |
| GC10-DET | a04 | 489338187 | 0.240225 | 0.242923 | +0.270 pp |
| GC10-DET | a05 | 1247201608 | 0.246716 | 0.254026 | +0.731 pp |
| GC10-DET | a06 | 1159501161 | 0.244529 | 0.251605 | +0.708 pp |
| GC10-DET | a07 | 1248489736 | 0.249099 | 0.252691 | +0.359 pp |
| GC10-DET | a08 | 1498171501 | 0.242118 | 0.252130 | +1.001 pp |
| GC10-DET | a09 | 1117821898 | 0.244327 | 0.245874 | +0.155 pp |
| NEU-DET | a00 | 556094945 | 0.342822 | 0.347543 | +0.472 pp |
| NEU-DET | a01 | 1401936841 | 0.353611 | 0.362891 | +0.928 pp |
| NEU-DET | a02 | 308869329 | 0.357355 | 0.363527 | +0.617 pp |
| NEU-DET | a03 | 1765023412 | 0.359061 | 0.363220 | +0.416 pp |
| NEU-DET | a04 | 1863187687 | 0.343290 | 0.355766 | +1.248 pp |
| NEU-DET | a05 | 1566547639 | 0.348665 | 0.355099 | +0.643 pp |
| NEU-DET | a06 | 1039657086 | 0.351805 | 0.357006 | +0.520 pp |
| NEU-DET | a07 | 1644540782 | 0.348499 | 0.353540 | +0.504 pp |
| NEU-DET | a08 | 1157893727 | 0.359269 | 0.355272 | -0.400 pp |
| NEU-DET | a09 | 1944238630 | 0.355119 | 0.363448 | +0.833 pp |

## Budget curve

| Dataset | Round | Labeled fraction | Random mAP50-95 | DINO mAP50-95 | paired delta (95% CI) |
|---|---:|---:|---:|---:|---:|
| GC10-DET | R0 | 10.0% | 20.51% | 20.51% | +0.000 pp [+0.000 pp, +0.000 pp] |
| GC10-DET | R1 | 15.0% | 22.91% | 23.16% | +0.246 pp [-0.374 pp, +0.855 pp] |
| GC10-DET | R2 | 20.0% | 23.95% | 25.09% | +1.142 pp [+0.658 pp, +1.626 pp] |
| GC10-DET | R3 | 25.0% | 25.19% | 25.71% | +0.522 pp [-0.025 pp, +1.045 pp] |
| GC10-DET | R4 | 30.0% | 26.65% | 27.35% | +0.700 pp [+0.328 pp, +1.107 pp] |
| GC10-DET | R5 | 35.0% | 26.74% | 27.33% | +0.590 pp [-0.010 pp, +1.155 pp] |
| NEU-DET | R0 | 10.0% | 30.62% | 30.62% | +0.000 pp [+0.000 pp, +0.000 pp] |
| NEU-DET | R1 | 15.0% | 33.06% | 33.87% | +0.812 pp [+0.227 pp, +1.338 pp] |
| NEU-DET | R2 | 20.0% | 34.72% | 35.82% | +1.091 pp [+0.513 pp, +1.653 pp] |
| NEU-DET | R3 | 25.0% | 36.63% | 37.02% | +0.388 pp [-0.052 pp, +0.781 pp] |
| NEU-DET | R4 | 30.0% | 37.24% | 37.70% | +0.463 pp [-0.085 pp, +0.996 pp] |
| NEU-DET | R5 | 35.0% | 38.02% | 38.30% | +0.274 pp [-0.225 pp, +0.743 pp] |

## Secondary and subgroup endpoints

Secondary 결과는 primary를 rescue하는 formal claim으로 사용하지 않는다.

| Dataset | Endpoint | DINO − Random | 95% CI | wins/losses |
|---|---|---:|---:|---:|
| GC10-DET | `final_map50_95` | +0.590 pp | [-0.010 pp, +1.155 pp] | 8/2 |
| NEU-DET | `final_map50_95` | +0.274 pp | [-0.225 pp, +0.743 pp] | 7/3 |
| GC10-DET | `aulc_recall` | +0.723 pp | [+0.258 pp, +1.251 pp] | 8/2 |
| NEU-DET | `aulc_recall` | +0.412 pp | [-0.012 pp, +0.808 pp] | 7/3 |
| GC10-DET | `final_recall` | +0.657 pp | [-1.240 pp, +2.388 pp] | 5/5 |
| NEU-DET | `final_recall` | +0.410 pp | [-0.635 pp, +1.582 pp] | 4/6 |
| GC10-DET | `aulc_gc10_rare_macro_ap5095` | +0.673 pp | [+0.302 pp, +1.068 pp] | 9/1 |
| GC10-DET | `final_gc10_rare_macro_ap5095` | +0.388 pp | [-0.603 pp, +1.262 pp] | 7/3 |
| NEU-DET | `aulc_neu_macro_ap5095` | +0.578 pp | [+0.310 pp, +0.812 pp] | 9/1 |
| NEU-DET | `final_neu_macro_ap5095` | +0.274 pp | [-0.225 pp, +0.743 pp] | 7/3 |
| NEU-DET | `aulc_neu_worst_class_ap5095` | -1.488 pp | [-1.978 pp, -1.002 pp] | 0/10 |
| NEU-DET | `final_neu_worst_class_ap5095` | -3.118 pp | [-3.822 pp, -2.372 pp] | 0/10 |

### Class-wise final-budget effects

| Dataset | Class | DINO − Random AP50-95 | 95% CI |
|---|---|---:|---:|
| GC10-DET | punching_hole | -0.214 pp | [-1.383 pp, +0.766 pp] |
| GC10-DET | welding_line | +5.538 pp | [+1.509 pp, +9.520 pp] |
| GC10-DET | crescent_gap | +2.713 pp | [+1.550 pp, +3.964 pp] |
| GC10-DET | water_spot | +1.391 pp | [+0.220 pp, +2.627 pp] |
| GC10-DET | oil_spot | -0.161 pp | [-0.904 pp, +0.519 pp] |
| GC10-DET | silk_spot | -3.353 pp | [-4.083 pp, -2.563 pp] |
| GC10-DET | inclusion | -1.176 pp | [-1.816 pp, -0.621 pp] |
| GC10-DET | rolled_pit | -1.254 pp | [-2.298 pp, -0.334 pp] |
| GC10-DET | crease | +3.975 pp | [+2.244 pp, +5.770 pp] |
| GC10-DET | waist_folding | -1.557 pp | [-3.751 pp, +0.461 pp] |
| NEU-DET | crazing | -3.118 pp | [-3.822 pp, -2.372 pp] |
| NEU-DET | inclusion | +0.220 pp | [-1.038 pp, +1.614 pp] |
| NEU-DET | patches | +0.190 pp | [-0.528 pp, +1.026 pp] |
| NEU-DET | pitted_surface | +2.236 pp | [+1.170 pp, +3.336 pp] |
| NEU-DET | rolled-in_scale | -3.336 pp | [-4.993 pp, -1.432 pp] |
| NEU-DET | scratches | +5.449 pp | [+3.956 pp, +7.318 pp] |

## Selection-stage discovery/composition

| Dataset | Round | Metric | DINO − Random | 95% CI |
|---|---:|---|---:|---:|
| GC10-DET | R5 | `gc10_rare_target_image_count` | +14.7000 images | [+11.7000, +17.5000] |
| GC10-DET | R5 | `normalized_shannon_image_incidence` | +0.0207 | [+0.0167, +0.0245] |
| GC10-DET | R5 | `total_instances` | +46.6000 | [+38.1000, +55.4000] |
| NEU-DET | R5 | `normalized_shannon_image_incidence` | -0.0610 | [-0.0625, -0.0596] |
| NEU-DET | R5 | `total_instances` | -54.9000 | [-68.0000, -42.5000] |
| NEU-DET | R5 | `class_image_count_0_crazing` | -41.5000 images | [-44.4000, -38.1000] |
| NEU-DET | R5 | `class_image_count_1_inclusion` | +5.1000 images | [+1.9000, +8.4000] |
| NEU-DET | R5 | `class_image_count_2_patches` | -1.4000 images | [-5.5000, +3.0000] |
| NEU-DET | R5 | `class_image_count_3_pitted_surface` | +31.6000 images | [+28.5000, +34.8000] |
| NEU-DET | R5 | `class_image_count_4_rolled-in_scale` | -35.1000 images | [-39.0000, -31.4000] |
| NEU-DET | R5 | `class_image_count_5_scratches` | +57.6000 images | [+53.2000, +61.6000] |

## Learner variation

| Dataset | Strategy | mean |A−B| AULC | median |A−B| | max |A−B| |
|---|---|---:|---:|---:|
| GC10-DET | Random-PCG64-v1 | +0.440 pp | +0.466 pp | +1.032 pp |
| GC10-DET | FrozenDINO-FF-v1 | +0.368 pp | +0.210 pp | +1.534 pp |
| NEU-DET | Random-PCG64-v1 | +0.242 pp | +0.208 pp | +0.590 pp |
| NEU-DET | FrozenDINO-FF-v1 | +0.266 pp | +0.131 pp | +0.727 pp |

## Historical evidence와의 bounded comparison

- Historical NEU fixed-set `+0.016236 (5/5)`는 training stability이고, 새 acquisition 10개의 historical estimate `+0.007019`, CI `[-0.005211,+0.019678]`는 acquisition reproducibility 미확립이었다. 이번 controlled result는 어느 쪽도 새 n으로 합치지 않고 방향과 구조만 비교한다.
- Historical GC10 q20은 overall `+0.017378`과 rare macro `−0.019877`이 공존했다. 이번 결과에서도 overall과 frozen rare subgroup을 반드시 함께 해석한다.
- MPDD/VisA discovery, KSDD2 threshold, DeepPCB STOP은 다른 task/estimand이며 confirmatory effect에 pooling하지 않는다.

## 냉정한 연구 평가 기준

1. **Selector 논문으로서:** Frozen DINO 자체의 novelty나 보편적 superiority를 주장할 수 없다.
2. **석사학위 연구로서:** result-independent freeze, acquisition/training seed 분리, full learning curve, subgroup, raw registry와 negative-result 보존은 강한 장점이다.
3. **핵심 학술 기여:** candidate signal → discovery → composition → reproducibility → downstream learning을 동일 개념으로 취급하면 안 된다는 산업 결함 AL의 empirical evidence다.
4. **해석 한계:** 두 dataset, YOLOv8n 한 family, global DINO image representation, image-count budget, training seed 2개에 한정된다.
5. **금지 주장:** annotation-time 절감, 산업 현장 일반화, 완전한 untouched test, DINO의 universal superiority/equivalence.

## Reproducibility artifacts

- `results/raw_locked_final_metrics_conceptual_480.csv`
- `results/curve_level_aulc_80.csv`
- `results/primary_acquisition_deltas_20.csv`
- `results/selection_composition_raw_240.csv`
- `results/CONFIRMATORY_ANALYSIS_INTEGRITY_AND_SUMMARY.json`
- `figures/confirmatory_primary_results.*`
- `figures/confirmatory_classwise_results.*`
