# Phase 2A Readiness Audit

## 판정

**READY_FOR_GPU_SMOKE**

이 판정은 Phase 2B tiny CUDA smoke와 runtime/storage benchmark를 수행할 준비가 됐다는 뜻이다. 440 full experiment 또는 final evaluation의 승인이 아니다.

## Gate checklist

| Gate | Result |
|---|---|
| exact base commit isolated worktree | PASS |
| historical evidence left unchanged | PASS |
| RQ/primary/claim frozen | PASS |
| Random exact RNG/order frozen | PASS |
| FrozenDINO model/processor/tensor/precision/FF frozen | PASS |
| prospective GT-free code boundary | PASS |
| identity duplicate/conflict audit | PASS |
| split sizes and all-class support | PASS |
| budget counts/rounding | PASS |
| acquisition/training seed separation | PASS |
| YOLO lifecycle exact candidate | PASS |
| AULC/statistical/subgroup definition | PASS |
| final evaluation hard lock | PASS |
| CPU preflight | 14 PASS / 0 FAIL |
| full operational feasibility | UNKNOWN — Phase 2B target |

## Immutable candidate artifact

- YAML: [`confirmatory_protocol_v1.yaml`](./confirmatory_protocol_v1.yaml)
- SHA-256: `fa6ad82f5f6b282b1fde5529af41cb54eb5471d3f2950f00d32f491b8ed0c663`
- status inside YAML: `FROZEN_CANDIDATE_NOT_EXECUTED`

YAML을 수정하면 SHA를 다시 봉인하고 readiness를 재감사해야 한다.

## Remaining technical risks

1. **Operational scale unknown:** T0...T5, total GPU-hours, retry rate와 storage가 측정되지 않았다.
2. **Runtime compatibility:** 2022 YOLOv8n checkpoint와 Ultralytics 8.4.90 조합은 load audit만 했고 tiny CUDA train/metric extraction은 아직 안 했다.
3. **DINO full-cache determinism:** one-image CPU spec은 통과했지만 two full CUDA FP32 cache의 exact hash/completeness는 아직 없다.
4. **GC10 rare development precision:** all rare classes가 남지만 class 8/9 development support가 각 5 images라 subgroup CI가 넓을 수 있다.
5. **Public-data lock limitation:** final membership은 prospectively lock됐지만 public GT 자체는 과거 연구에서 이미 분석됐다.

## Phase 2B required tasks

1. Protocol/YAML/code/model/data hashes를 먼저 재확인한다.
2. Acquisition split만 이용해 GC10/NEU DINO FP32 cache 2개를 만들고 count/dim/NaN/zero/hash validation 후 seal한다.
3. Final registry access guard가 계속 fail하는지 smoke 시작 전 재확인한다.
4. Prespecified tiny acquisition/development fixture에서 YOLO CUDA cold-start smoke를 실행해 checkpoint load, per-class extraction, best/last lifecycle을 검증한다.
5. 실제 `T0...T5`, DINO cache time, evaluation time, peak GPU/RAM, artifact storage를 측정한다.
6. Smoke failure는 outcome과 분리된 attempt registry에 기록하고 protocol field 변경이 필요하면 440을 시작하지 말고 재승인한다.
7. 사용자·지도교수 scope 및 operational feasibility 결정을 받은 뒤에만 별도 full-run authorization을 만든다.

## Explicit hold

- full 440: HOLD
- final evaluation: LOCKED
- professor approval: NOT CLAIMED
- remote push: NOT PERFORMED
- existing Raw Master modification: NOT PERFORMED

