# Research context handoff — 2026-07-12

이 문서는 현재 Codex 대화 맥락이 사라져도 Defect VLM Active Learning 연구를 이어갈 수 있도록 남기는 연구 핸드오프다.

## 1. 현재 연구 상태 한 줄 요약

초기 아이디어였던 “VLM explanation consistency만으로 Random보다 나은 GT-free Active Learning을 만들 수 있다”는 현재까지의 실험에서는 지지되지 않았다. 그러나 실패 원인을 분리하면서 `DINO visual diversity`, `detector-aware uncertainty`, `instance richness`, `class balance`를 결합하는 방향이 더 타당하다는 근거를 얻었다.

가장 최신 후보인 V10b는 seed42에서는 Random을 이겼지만, seed43~46 독립 one-cycle 검증에서는 평균 mAP50-95 차이가 `+0.000949`에 그쳤다. 따라서 V10b는 “가능성 있는 방향”이지 “Random 대비 일반화 우위가 검증된 최종 방법”은 아니다.

## 2. 실험 흐름 요약

### V3/V5 계열

- VLM explanation/consistency 기반 GT-free active learning 가능성을 탐색했다.
- 초기에는 결과가 좋아 보이는 구간이 있었지만, dataset split, evaluation size, protocol confound를 충분히 고정하지 못했다.
- 이후 분석에서 “작고 흔들리는 validation”과 “데이터 구성 confound”가 큰 위험이라는 점이 드러났다.

### V7

- full-curve, multi-seed, root-cause audit를 강화했다.
- 중요한 발견:
  - row-order sampling 문제를 canonical sampling으로 교정했다.
  - split overlap은 주요 원인이 아니었다.
  - mixed GC10/NEU 프로토콜에서 클래스와 데이터셋 구성이 심하게 섞여 해석이 불안정했다.
- 결론:
  - VLM consistency-only 전략은 Random 대비 안정적 우위를 보이지 못했다.

### V8 NEU-only

- 데이터셋 confound를 줄이기 위해 NEU-only로 전환했다.
- Random, DINO Visual, Consistency를 비교했다.
- 결과:
  - 전체 성능은 protocol 정리 후 상승했다.
  - DINO Visual은 Consistency보다 낫지만 Random을 안정적으로 이기지는 못했다.
- 해석:
  - visual diversity는 유효한 보조 신호일 수 있지만 단독 acquisition strategy로는 부족했다.

### V9 detector-aware pivot

- detector uncertainty, DINO diversity, balance를 결합하는 방향으로 pivot했다.
- 초기 V9는 “어려운 샘플”을 너무 많이 고르는 문제가 있었다.
- 결과적으로 학습 가능한 정보량이 아니라 hard/noisy sample 쪽으로 치우쳐 성능이 애매했다.

### V9b instance-rich DINO balanced

- “어려운 샘플”보다 “학습 가능한 instance-rich 샘플”이 중요하다는 가설로 score를 수정했다.
- V9b는 V9보다 방향이 좋아졌지만, 5-seed pilot에서는 Random을 안정적으로 이기지 못했다.
- 중요한 교훈:
  - pseudo instance count는 도움이 된다.
  - 하지만 너무 instance-rich로 몰리면 다양성과 recall 측면에서 문제가 생길 수 있다.

### V10 NEU large-pool smoke

- NEU 전체 1,800장을 기준으로 더 큰 pool/eval 구조를 도입했다.
- 주요 설계:
  - acquisition pool: 900
  - development eval: 300
  - final test: 300 locked
  - initial budget: 60
  - query size: 30
  - one-cycle smoke first
- seed42 결과:
  - Round0 mAP50-95: 0.306094
  - Random mAP50-95: 0.329700
  - V9b mAP50-95: 0.317276
- 해석:
  - 데이터 설계는 더 안정적이었다.
  - V9b는 rich sample을 고르지만 Random보다 낮았다.

### V10b frozen-weight selection

V10b는 V9b의 과한 instance-rich 편향을 줄이고, uncertainty/DINO/balance/instance를 고정 가중치로 조합했다.

```text
detector_uncertainty:     0.25
dino_visual_distance:    0.35
predicted_class_deficit: 0.15
pseudo_instance_count:   0.25
```

seed42 single training 결과:

```text
Random mAP50-95: 0.329700
V9b mAP50-95:    0.317276
V10b mAP50-95:   0.340866

V10b - Random: +0.011166
V10b - V9b:    +0.023590
```

per-class audit:

```text
V10b vs Random AP50-95:
crazing:        +0.0244
inclusion:      -0.0596
patches:        +0.0247
pitted_surface: +0.0496
rolled-in-scale:-0.0138
scratches:      +0.0417
```

V10b는 V9b 대비 6개 NEU 클래스 모두에서 개선되었다. 하지만 Random 대비로는 class-wise 손실도 있었다.

### V10b seed43~46 independent one-cycle

목적:

- seed42에서 좋아 보인 결과가 acquisition seed에 과적합된 것인지 확인한다.
- V9b는 제외하고 Random vs frozen V10b만 비교한다.
- final test는 계속 사용하지 않는다.

결과:

```text
Completed seeds: [43, 44, 45, 46]
Failed seeds:    []
Random mean mAP50-95: 0.299207
V10b mean mAP50-95:   0.300157
paired mean diff:     +0.000949
V10b wins/losses/ties: 3/1/0
precision mean diff:  +0.023781
recall mean diff:     -0.026170
final test used:      False
method weights frozen: True
```

해석:

- V10b는 Random과 거의 동률이다.
- wins/losses는 3/1로 좋아 보이지만 평균 차이는 매우 작다.
- precision은 좋아지고 recall은 떨어지는 경향이 있다.
- 따라서 V10b는 “Random을 이겼다”가 아니라 “Random 근처까지 온 detector-aware candidate”로 기록해야 한다.

## 3. 왜 Random을 이기기 어려웠나

현재 연구에서 Random이 강했던 이유는 다음으로 해석한다.

1. NEU 데이터의 결함 클래스가 작고 제한적이라, 어느 정도 균형 잡힌 random sampling도 빠르게 유효 instance를 확보한다.
2. query size 30에서는 active selection의 이점이 노이즈와 seed variation에 묻히기 쉽다.
3. detector-aware score가 pseudo-label에 의존하므로 Round0 detector의 bias를 그대로 물려받을 수 있다.
4. instance-rich sample은 precision에는 도움을 주지만, 다양성과 recall을 희생할 수 있다.
5. GT-free 조건에서는 실제 missed defect를 직접 알 수 없기 때문에 “좋은 어려움”과 “학습 불가능한 어려움”을 분리하기 어렵다.

## 4. 지금까지 고정한 중요한 원칙

- final test는 마지막까지 locked.
- development eval만 보고 방법을 고른다.
- Random baseline은 반드시 충분히 강하게 둔다.
- seed42 하나만 보고 결론 내리지 않는다.
- “좋아 보이는 결과”보다 implementation integrity audit를 먼저 본다.
- GT-free strategy와 oracle/GT-based strategy 명칭을 분리한다.
- raw `runs/`는 GitHub에 올리지 않고, curated result만 `docs/results/`에 올린다.

## 5. GitHub에 보존된 핵심 artifact

문서:

- `docs/preview.html`
- `docs/vlm_gt_free_al_workflow.html`
- `docs/final_detector_aware_pivot_protocol_20260712.md`
- `docs/v8_neu_only_5seed_result_log_20260712.md`
- `docs/v9_detector_aware_reimplementation_plan_20260712.md`
- `docs/research_context_handoff_20260712.md`
- `docs/macbook_handoff_guide_20260712.md`
- `docs/continuation_playbook_20260712.md`

보고서:

- `docs/results/v10b_seed42_documentation_20260712_215841/V10b_Seed42_Development_Gate_Updated_Report.docx`
- `docs/results/v10b_seed42_documentation_20260712_215841/V10b_Seed42_Development_Gate_Updated_Report.md`
- `docs/results/v10b_seed42_documentation_20260712_215841/figures/`
- `docs/results/v10b_seed42_documentation_20260712_215841/tables/`

실행/감사 코드:

- `scripts/02_active_learning/run_v10_neu_large_pool_smoke.py`
- `scripts/02_active_learning/probe_v10b_selection_from_existing_v10.py`
- `scripts/02_active_learning/train_v10b_from_existing_selection.py`
- `scripts/02_active_learning/recover_v10_per_class_audit.py`
- `scripts/02_active_learning/run_v10b_multiseed_onecycle.py`
- `scripts/02_active_learning/build_v10b_seed42_documentation.py`
- `scripts/docs/check_handoff_package.py`
- `scripts/docs/macbook_open_docs.sh`

## 6. 다음 실험으로 넘어간다면

추천 방향은 V10b를 그대로 밀어붙이는 것이 아니라, V10b의 recall penalty를 줄이는 쪽이다.

가능한 다음 후보:

1. V10c recall-aware candidate
   - no-box를 완전히 배제하지 않고 제한적으로 허용한다.
   - pseudo instance count 상한을 조금 완화한다.
   - DINO distance를 유지하되 과도한 precision 편향을 줄인다.

2. query size sensitivity
   - query size 30에서 차이가 너무 작다면 60도 비교한다.
   - 단, annotation budget claim이 달라지므로 명확히 분리해야 한다.

3. longer curve
   - one-cycle에서 동률이라면 round 2~4에서 누적 이점이 있는지 확인한다.
   - 그러나 final test는 여전히 사용하지 않는다.

4. statistical reporting
   - paired seed diff
   - win/loss/tie
   - bootstrap CI
   - per-class AP delta
   - precision/recall trade-off

## 7. 현재 결론 문장 후보

논문/보고서에서는 다음 표현이 안전하다.

> V10b improved substantially over the previous detector-aware V9b variant and achieved near-parity with a strong random baseline under independent seed validation. However, the average gain over Random remained marginal, and the method exhibited a precision-recall trade-off. We therefore treat V10b as a promising detector-aware direction rather than a validated replacement for Random.

한국어로는:

> V10b는 이전 detector-aware V9b보다 명확히 개선되었고 독립 seed 검증에서 Random과 거의 동률 수준까지 도달했다. 그러나 Random 대비 평균 이득은 매우 작았으며 recall 감소가 동반되었다. 따라서 V10b는 최종 방법이라기보다, recall 보정이 필요한 유망한 detector-aware 방향으로 해석하는 것이 타당하다.

