# Controlled Validation Protocol Decision v1

- 결정일: 2026-08-22 KST
- 상태: **FROZEN CANDIDATE — NOT EXECUTED — NO 440 AUTHORITY**
- Phase 2A readiness: **READY_FOR_GPU_SMOKE**
- 기준 커밋: `71489c238d3a44316d4721d6a62c963ebd919894`
- 격리 브랜치: `experiment/confirmatory-random-dino-v1`

## 1. 최종 결정

GC10-DET와 NEU-DET에서 **Random-PCG64-v1 대 FrozenDINO-FF-v1**의 2-arm controlled validation을 최우선 protocol candidate로 동결한다. 이 block의 역할은 새 selector의 novelty를 만드는 것이 아니라, selection-stage의 관찰이 budget 전반의 downstream detector learning으로 번역되는지를 동일 조건에서 확인하는 것이다.

440 detector jobs는 과학적 설계 수로 유지하지만, 이 문서와 YAML은 실행 승인이 아니다. 이번 Phase에서 허용되는 다음 단계는 Phase 2B GPU smoke와 runtime/storage benchmark뿐이다.

## 2. 동결한 연구 질문

> 동일한 object-detection Active Learning protocol에서 DINO representation diversity 기반 표본 선택이 Random sampling과 비교하여 budget 전반의 downstream detector learning performance에 어떤 차이를 만드는가?

목표는 DINO superiority 입증이 아니다. DINO 우세, 명확한 차이 미확립, Random 우세, dataset별 반대 방향을 모두 허용한다.

## 3. Primary estimand

Primary는 **two-training-seed averaged, six-budget mAP50-95 AULC**다.

1. dataset × acquisition seed × strategy × training seed별 six-point AULC를 계산한다.
2. 같은 acquisition/strategy의 두 training-seed AULC를 산술평균한다.
3. 같은 acquisition 안에서 `FrozenDINO-FF-v1 - Random-PCG64-v1` paired contrast를 만든다.
4. acquisition seed 10개가 primary unit다.
5. GC10과 NEU를 `n=20`으로 단순 합치지 않고 dataset별로 먼저 보고한다.

## 4. 설계 결정표

| 항목 | 동결 결정 | 이유 |
|---|---|---|
| Strategies | Random + Frozen DINO 2-arm | uncertainty formula 탐색을 제거하고 acquisition 10 × training 2 full curve 유지 |
| DINO 성격 | historical lineage + prospectively cleaned implementation | historical embedding/revision seal이 없어 exact binary replication 불가 |
| Split | 공통 deterministic 70/10/20 | historical split이 dataset별로 이질적이며 동일 protocol 비교가 목적 |
| Identity | canonical RGB pixel SHA-256 | filename/class text 제거와 중복 JPEG encoding 검출 |
| Budget | fixed acquisition pool의 10% + 5% × 5, half-up | 두 dataset에 같은 비율 규칙과 six-point AULC 적용 |
| Acquisition seeds | dataset당 새 deterministic 10개 | favorable/unfavorable historical seed 선택 방지 |
| Training seeds | acquisition당 2개, strategy paired | learner variation을 acquisition 안에 nested |
| Learner | pinned YOLOv8n, every-budget cold start | acquisition effect와 continual-training effect 분리 |
| Primary | budget-averaged mAP50-95 AULC | final point 하나가 아닌 전체 learning curve 유지 |
| Final | 외부 authorization 전 hard lock | development와 final 역할 분리 |

## 5. 440 graph

한 dataset/acquisition/training-seed cell은 `1 shared Round0 + 2 strategies × 5 post rounds = 11` jobs다.

`2 datasets × 10 acquisitions × 2 training seeds × 11 = 440`

- shared Round0: 40
- Random post-round: 200
- Frozen DINO post-round: 200
- unique cumulative selected states: 220
- final evaluation: Phase 2A에서 0

## 6. Evidence basis와 해석 경계

동결 전 `RAW_EXPERIMENT_RESULTS_MASTER_20260822.md`, NEU addendum/source index/recovery audit, two-arm decision 및 440 compute audit, current-state/narrative/decision-items를 읽었다. 기존 결과는 protocol 선택 후의 새 결과가 아니며 새 confirmatory n에 합치지 않는다.

특히 다음을 유지한다.

- NEU의 fixed-set stability와 acquisition reproducibility는 다른 estimand다.
- GC10 q20 overall positive와 rare negative는 같이 보존한다.
- MPDD/VisA discovery는 downstream evidence가 아니다.
- KSDD2/DeepPCB의 threshold/STOP는 결과 삭제 근거가 아니다.

## 7. 범위 밖

이 protocol은 uncertainty/VLM selector family의 상대 순위, universal DINO superiority, annotation-time 절감, 산업 현장 일반화를 답하지 않는다. 지도교수의 범위 승인을 받았다고 표현하지 않는다.

정본 YAML: [`confirmatory_protocol_v1.yaml`](./confirmatory_protocol_v1.yaml), SHA-256 `fa6ad82f5f6b282b1fde5529af41cb54eb5471d3f2950f00d32f491b8ed0c663`.

