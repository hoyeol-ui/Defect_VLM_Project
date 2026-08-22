# YOLO Lifecycle Freeze

- learner: **YOLOv8n**
- lifecycle: **independent cold start at every budget**
- operational feasibility: **UNKNOWN until Phase 2B benchmark**
- detector training in Phase 2A: **0 jobs**

## 1. Historical/current audit

| Field | Historical V7 | Historical GC10 q20 | Confirmatory v1 |
|---|---|---|---|
| checkpoint | `yolov8n.pt` | local `yolov8n.pt` | same sealed local artifact |
| epochs | 100 | 75 | 100 |
| imgsz / batch | 640 / 8 | 640 / 16 | 640 / 8 |
| start | cold | cold | cold every budget |
| optimizer | implicit Ultralytics default | explicit `auto` | explicit `SGD` |
| checkpoint outcome | best | last | best, with last retained |
| cache | RAM | false | false |
| validation | development during train | separate development after train | development during train + frozen best evaluation |

Current environment의 Ultralytics 8.4.90은 `optimizer=auto`가 iteration 수에 따라 AdamW 또는 MuSGD를 고른다. 이 동작은 historical version과 달라질 수 있으므로 v1에서는 `SGD`를 직접 고정했다. 이는 결과를 본 tuning이 아니라 version-dependent hidden choice 제거다.

## 2. Model seal

- source artifact: `C:/Users/user/Desktop/vlm/Defect_VLM_Project/yolov8n.pt`
- bytes: `6,549,796`
- SHA-256: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`
- embedded artifact version: Ultralytics `8.0.0.dev0`
- runtime package: Ultralytics `8.4.90`

각 job은 artifact를 새로 load하고 `resume=False`로 시작한다. 이전 budget의 best/last weight를 다음 budget에 전달하지 않는다. Dataset head initialization과 training stochasticity만 frozen training seed에 의해 달라진다.

## 3. Exact train configuration

| Field | Value |
|---|---|
| epochs / patience | 100 / 100 |
| imgsz / batch | 640 / 8 |
| device / workers | `0` / 4 |
| cache / AMP | false / true |
| deterministic | true |
| optimizer | SGD, Nesterov as implemented by pinned Ultralytics |
| lr0 / lrf / cosine | 0.01 / 0.01 / false |
| momentum / weight decay | 0.937 / 0.0005 |
| nbs | 64 |
| warmup | 3 epochs, momentum 0.8, bias LR 0.1 |
| close mosaic | final 10 epochs |
| rect / single-class / fraction | false / false / 1.0 |

Augmentation은 `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`, `degrees=0`, `translate=0.1`, `scale=0.5`, `shear=0`, `perspective=0`, `flipud=0`, `fliplr=0.5`, `bgr=0`, `mosaic=1`, `mixup=0`, `cutmix=0`, `copy_paste=0`으로 고정한다.

## 4. Validation/checkpoint

- training-time validation: development split, every Ultralytics scheduled epoch
- primary checkpoint: development fitness 기준 `best.pt`
- final epoch `last.pt`도 raw audit artifact로 보존
- primary/secondary metric extraction: sealed `best.pt`에서 development를 동일 config로 평가
- evaluation: `conf=0.001`, `iou=0.7`, `max_det=300`, plots/save_json false
- final split: authorization 전 dataset YAML/registry loader 단계에서 hard fail
- retention: best, last, args, results, aggregate/per-class metrics, log

## 5. Round0 sharing

같은 dataset × acquisition seed × training seed에서 initial manifest, checkpoint, learner config와 code hash가 같을 때 Random/DINO가 Round0 training artifact를 공유한다. 두 training seeds 사이에는 공유하지 않는다.

## 6. Retry

Infrastructure failure만 같은 scientific cell ID 아래 attempt를 늘려 재실행한다. Seed/config/package/model/manifest를 바꾸는 retry는 금지하며 별도 protocol 재승인 대상이다. Result를 보고 checkpoint, seed, epoch 또는 augmentation을 바꾸지 않는다.

## 7. 남은 operational question

CPU-only Phase 2A는 job당 `T0...T5`, 8 GB GPU memory, worker/cache throughput, checkpoint storage를 측정하지 않았다. 이 값은 Phase 2B tiny CUDA smoke에서 protocol을 바꾸지 않고 측정한다.

