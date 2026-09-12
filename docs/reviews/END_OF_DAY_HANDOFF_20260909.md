# 다음 세션 handoff — 2026-09-09

최신 공유 브랜치: `research/entropy-a00-audit-20260908`. 9월 8일 저녁 실험과 감사에 이어 9월 9일 00시대에 CPU 사후 진단을 마쳤다. 다음 담당자는 루트 README와 [geometry 보고서](dino_class_geometry_posthoc_20260909/README.md)를 먼저 읽는다.

## 1. 확정된 thesis 방향

제한된 라벨 예산의 산업 결함 객체 탐지에서 **사전학습 DINOv2 + farthest-first**의 평균 학습 효용과 취약 클래스 성능을 함께 설명한다. Entropy는 detector uncertainty 계열 비교군이다. 사전학습 DINOv2 / Pretrained DINOv2로 설명하며, 본 실험에서 encoder 가중치를 추가 학습하지 않았다는 사실도 함께 쓴다. 과거 식별자와 원고 본문은 수정하지 않았다.

## 2. Historical 440

기존 공식 440-cell 결과와 평가 기록은 완료 상태다. 이번 작업은 source cache·acquisition manifest·기존 선택 구성 집계만 분석에 사용했다. 분석 입력 hash는 모두 유지됐고, historical guard 1,348개와 기존 보호 metadata 2,574개, 총 3,922개 경로의 size/mtime도 유지됐다. 새로운 440-checkpoint 전수 hash 감사나 locked-test 결과 재평가를 수행했다고 보고하지 않는다.

## 3. DINOv2 validity evidence

기존 전체/클래스별 1-NN, 5-NN, 거리 분석은 재사용했다. 새로 full label-set permutation 10,000회를 수행했다.

| Dataset | Observed 1-NN | Chance mean | Null 95% 범위 | Difference |
|---|---:|---:|---:|---:|
| GC10-DET | 81.25% | 21.07% | 18.82–23.30% | +60.17 pp |
| NEU-DET | 99.21% | 18.96% | 16.44–21.53% | +80.25 pp |

라벨 하나라도 겹치면 match다. 두 데이터셋 모두 multi-label 이미지를 포함한다. 위 구간은 null distribution percentile이며 관측치 CI가 아니다.

## 4. NEU class composition / AP

기존 R5 선택 편차는 crazing −41.5장, rolled-in_scale −35.1장, patches −1.4장, inclusion +5.1장, pitted_surface +31.6장, scratches +57.6장이다. 기존 보고서 AP 차이와 Spearman 약 0.943의 연관성을 보였던 상태다. 이번에는 AP/test metric을 입력하지 않았다. 선택 count CI 5개와 AP CI 4개가 0을 배제한다는 차이, 그리고 post-hoc 지위를 유지한다.

## 5. 이번 geometry 진단

**전체 PARTIAL SUPPORT.** 사전 고정한 네 지표를 모두 보존했다.

| R5 geometry ↔ 선택 편차 rho | Pair median (primary) | Within-class 5-NN | Centroid median | Radius p90 |
|---|---:|---:|---:|---:|
| GC10-DET | 0.661 | 0.758 | 0.661 | 0.552 |
| NEU-DET | 1.000 | 0.886 | 0.943 | 0.543 |

NEU의 primary 거리 순위와 선택 편차 순위가 정확히 일치한다. GC10은 같은 방향이지만 punching_hole·rolled_pit·silk_spot 등의 예외가 있다. class support/co-occurrence, class 사이 위치, R0 reference coverage까지 통제한 분석은 아니다. 같은 geometry와 누적 round를 반복 집계한 것이므로 R1–R5를 독립 실험 5개로 세지 않는다.

## 6. Entropy official 상태

- Training: **20/200 COMPLETE**, 나머지 **180은 attempt 0**.
- Acquisition: **10/100 COMPLETE**, 나머지 **90은 attempt 0**.
- 완료 범위: GC10 a00 및 NEU a00의 R1–R5, A/B.
- **INTERIM DESCRIPTIVE ONLY / NOT FINAL — 1/10 acquisition trajectories per dataset**.
- 현재 승인된 실행 범위는 a00 두 trajectory다. 나머지 180개는 새로운 명시적 실행 승인이 필요하다.
- integration 2셀은 official에 재사용하지 않았다.
- 잠긴 test는 계속 비활성이다. class-wise/worst-class Entropy 결과는 아직 생성되지 않았다.
- 잔여 실행시간은 기존 RTX 3070 실측 계획치 약 **29.48 GPU시간**, 운영 범위 26.5–35.4시간이다. 통계적 CI가 아니며 이번 CPU 작업으로 다시 추정하지 않았다.

## 7. Full-data reference

[별도 4-run 계획](../design/full_data_reference_runplan_20260908.md)이 준비돼 있다. GC10 A/B, NEU A/B, 사전 지정 a00 learner seed를 쓰는 reference이며 acquisition selector가 아니다. 지정된 run output 폴더는 없고 이번에 시작하지 않았다. 성능의 보장된 upper bound라고 부르지 않는다.

## 8. 다음 작업 우선순위 1–5

1. 이번 geometry/chance와 예외를 검토자에게 공유하고 논문 3.10.4의 post-hoc 진단 위치에 맞춰 반영 범위를 결정한다. GPU 불필요.
2. 남은 Entropy 180 cells의 범위를 명시적으로 승인한 뒤 기존 freeze/seed/config를 유지해 독립 Windows runner로 실행한다. integration/a00 재사용 규칙을 바꾸지 않는다.
3. Entropy 공식 완료 후 integrity audit를 먼저 하고, 별도 평가 승인에 따라 기존 checkpoint의 class-wise AP·dynamic worst-class AP·해당 curve nAULC를 분석한다. 추가 학습 없이 checkpoint 평가가 가능하며 locked-test는 승인 전 접근하지 않는다.
4. 사전 계획된 100% reference 4 runs를 별도 승인·queue에서 실행한다. 결과가 없는데 참조 성능을 인용하지 않는다.
5. 허용된 범위의 원고/발표 작업에서 숫자 → 그래프 → 나중의 이미지 예시 순서를 유지한다. 이번에는 원고나 PPT를 작성·수정하지 않았다.

## 9. 유지해야 할 protocol

| Artifact | SHA-256 |
|---|---|
| Entropy engine freeze | `0d3deb6656b9b22ddbb8d63f17ef6721adac4f3120560528c44954f8713d00fb` |
| Entropy official freeze | `e40842787a93d2e06304f8da7eea5ea93f2125c7bd15982bbce6610903505e6a` |
| a00 authorization | `1cf0b12328878f222be2a3824e945f21e51620240f16723db51b15ef959e3936` |
| Official registry snapshot | `e610be1d3ea2e740be26123606137572b454164d91f21a251d9c0ff9a56d73e3` |
| 이번 geometry config | `f52d491815387bc19bf981bc2e8eb7f9549ca192a61ffb964f86b2f1a545447b` |

Entropy는 policy learner A validation-best checkpoint만 사용한다. A/B는 같은 누적 manifest에서 각각 사전 지정 seed, sealed YOLOv8n으로 cold-start한다. binary entropy·SUM·no-detection=0·conf=0.25·iou=0.70·max_det=300·canonical ID tie-break를 유지한다. geometry/chance는 해석용이므로 selector에 입력하지 않는다.

## 10. 아직 주장하면 안 되는 것

compactness 인과 메커니즘의 완전한 증명, 모든 클래스 보호, 사전학습 DINOv2의 모든 산업 데이터 적합성, annotation 시간·비용 절감, Entropy 최종 우위, 낮은 budget에서의 보편적 방법 순위, full-data reference의 관측 성능. 이번 rho=1.000은 NEU의 클래스 6개에 대한 geometry↔선택 구성 관계이며 기존 선택 구성↔AP rho=0.943과 혼동하지 않는다.

추가 GPU 학습 실행 없음 / historical artifacts 수정 없음 / locked-test 접근 없음
