# FrozenDINO-FF-v1 Method Specification

- 상태: **PROSPECTIVELY FROZEN CANDIDATE — FULL CACHE NOT BUILT**
- 내부명: `ConfirmatoryDINOFFV1`
- 정확한 표현: **historical DINO farthest-first lineage를 기반으로 결과 확인 전에 구현과 input boundary를 prospectively cleaned and frozen한 selector**
- 금지 표현: historical exact binary reproduction

## 1. Model과 processor

| Field | Frozen value |
|---|---|
| Model ID | `facebook/dinov2-small` |
| Snapshot | `ed25f3a31f01632728cabb09d1542f84ab7b0056` |
| `config.json` SHA-256 | `1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d` |
| `preprocessor_config.json` SHA-256 | `14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828` |
| `model.safetensors` SHA-256 | `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` |
| Loader | `AutoImageProcessor` + `AutoModel`, `local_files_only=True` |
| Processor class | `BitImageProcessor` |

Preprocessing은 RGB 변환, shortest edge 256 bicubic resize, 224×224 center crop, `1/255` rescale, ImageNet mean `[0.485, 0.456, 0.406]`와 std `[0.229, 0.224, 0.225]` normalize다.

## 2. Representation과 precision

- exact tensor: `outputs.pooler_output`; fallback 금지
- shape: `[batch, 384]`; 다른 차원은 hard fail
- inference: `model.eval()`, `torch.inference_mode()`
- precision: model/input/output FP32, autocast off, TF32 off
- deterministic policy: deterministic algorithms on, cuDNN benchmark off
- Phase 2B full-cache batch size: 32, device `cuda:0`
- CPU/GPU output은 float32 CPU array로 옮긴 뒤 row-wise L2 normalize
- NaN, infinity, norm `<=1e-12`는 hard fail

Historical V7은 CUDA AMP forward 후 float32 저장이었다. 이번 FP32 정책은 bitwise historical replication을 주장하지 않고 forward precision 자유도를 제거하기 위한 prospective cleaning이다.

## 3. Class-neutral identity

`sample_id`는 다음 canonical payload의 SHA-256이다.

```text
CONFIRMATORY_RGB_U8_V1\0 || width_be32 || height_be32 || RGB_U8_pixels
```

raw-byte SHA는 source integrity에만 보존한다. Filename, directory, class, bbox는 sample ID와 tie-break에 들어가지 않는다. Canonical RGB hash를 택한 이유는 GC10에서 픽셀은 같지만 JPEG bytes가 다른 12쌍이 실제 발견됐기 때문이다.

## 4. Cache completeness

dataset당 acquisition universe 하나에 대해 cache 하나만 만든다. ordered row는 `sample_id` ascending이다. 다음 중 하나라도 있으면 selection을 시작하지 않는다.

- expected manifest count와 row count 불일치
- duplicate/missing/extra sample ID
- 384 이외 차원
- NaN/infinity/zero vector
- snapshot/processor/model/code hash mismatch

On-demand embedding repair와 seed별 rebuild는 금지한다. Phase 2B에서 GC10/NEU cache array, ordered manifest, config, artifact/code hash와 최종 cache SHA를 함께 봉인한다.

## 5. Farthest-first

L2-normalized embedding `z`에 대해 candidate score는 다음이다.

```text
d(x,R) = 1 - max_{r in R} cosine(z_x, z_r)
```

`R`은 현재 cumulative labeled set과 같은 query batch에서 이미 고른 sample이다. 최대 distance sample을 하나 선택한 즉시 `R`에 넣고 다시 계산한다. Query size가 `q`면 이를 `q`번 반복한다. Exact score tie는 absolute tolerance `1e-12` 안에서 opaque `sample_id` ascending으로 푼다.

Training seed, detector checkpoint, class/bbox, development/final metric은 DINO trajectory 입력이 아니다.

## 6. Phase 2A actual verification

합성 RGB 이미지 1장에 recovered snapshot을 CPU/FP32로 실제 통과시켰다.

| Check | Result |
|---|---|
| processor | `BitImageProcessor` |
| pixel tensor | `[1,3,224,224]`, `torch.float32` |
| `pooler_output` | `[1,384]`, `torch.float32` |
| finite | PASS |
| post-normalization norm | `1.0` |

이는 full dataset embedding이 아니며 selector 결과를 생성하지 않았다.

구현: [`frozen_dino_ff_v1.py`](./src/confirmatory_v1/frozen_dino_ff_v1.py), SHA-256 `cb98419adc5df72e798b22f9b4cace3e1e0fb026bb415d62ff9dd37c515a63cc`.

