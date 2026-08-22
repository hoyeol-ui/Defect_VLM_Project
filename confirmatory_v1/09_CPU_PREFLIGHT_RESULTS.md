# CPU Preflight Results

- 실행일: 2026-08-22 KST
- 결과: **14 PASS / 0 FAIL**
- GPU compute: 0
- detector training: 0
- full DINO cache: 0
- final evaluation: 0

## 1. Required 14 tests

실행 명령:

```powershell
python -m pytest -q confirmatory_v1\tests\test_cpu_preflight.py
```

Observed output: `14 passed in 0.12s` (최초 구현 직후 실행도 `14 passed`).

| # | Test | Result |
|---:|---|---|
| 1 | same acquisition seed → identical initial | PASS |
| 2 | Random/DINO shared initial | PASS |
| 3 | different acquisition seeds differ | PASS |
| 4 | training seed absent from acquisition API/trajectory | PASS |
| 5 | GT/class/bbox selector columns fail | PASS |
| 6 | sample ID opaque SHA only | PASS |
| 7 | split overlap/duplicate SHA fail | PASS |
| 8 | DINO farthest-first toy geometry | PASS |
| 9 | exact tie → sample ID ascending | PASS |
| 10 | trapezoid AULC hand calculation | PASS |
| 11 | query/cumulative count grid | PASS |
| 12 | final lock without authorization | PASS |
| 13 | missing embedding fails | PASS |
| 14 | NaN and zero vector fail | PASS |

Test file: [`test_cpu_preflight.py`](./tests/test_cpu_preflight.py), SHA-256 `4e5e19463f753eb949d493dc729b69ca63107a72da101aac1ac2806e1096d316`.

## 2. DINO tiny CPU spec smoke

Recovered snapshot과 합성 RGB 이미지 한 장만 사용했다.

- `BitImageProcessor`
- input `[1,3,224,224]` float32
- exact `pooler_output [1,384]` float32
- finite PASS
- row-wise L2 후 norm `1.0`

전체 dataset image, development/final image 또는 historical output을 embedding하지 않았다.

## 3. Split builder integrity

동일 source에서 builder를 두 번 실행한 뒤 다음 SHA가 byte-identical했다.

- summary: `727c6b79ab1046a40ed272a6d347d14808dd00173937591c83eae855e4d0d2e2`
- hash ledger: `5975afafc5c871bee2aa2256303cfd63f402e4be6f40d70d45b7231114a2a991`
- zero class-support split cells: 0
- duplicate/overlap canonical identity: 0

## 4. Environment audit

| Component | Version |
|---|---|
| Python | 3.11.9 |
| NumPy | 2.4.6 |
| Pillow | 12.3.0 |
| PyTorch | 2.12.1+cu126 |
| Transformers | 5.13.0 |
| Ultralytics | 8.4.90 |
| GPU/driver read-only observation | RTX 3070 8 GB / 560.94 |

`nvidia-smi`와 package metadata만 읽었으며 CUDA training/inference smoke는 실행하지 않았다.

## 5. Preflight scope

PASS는 deterministic/GT-free/integrity primitives와 DINO tiny fixture가 작동한다는 뜻이다. Full DINO cache hash, CUDA memory, YOLO runtime, T0...T5, storage는 Phase 2B에서만 확인한다.

