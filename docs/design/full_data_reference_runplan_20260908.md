# 100% labeled-data reference run plan

PREPARED ONLY. No training or evaluation executed. This is a reference, not a selector or a statistical bound.

Two datasets × two learner seeds = four runs. Use the first existing A/B learner pair per dataset, selected by index only. Acquisition seed is not a repeat axis here. This explicit seed subset is a preparation assumption to review before execution.

| Dataset | Repeat | Learner seed | Train images |
|---|---|---:|---:|
| GC10-DET | A | 1356694077 | 1605 |
| GC10-DET | B | 850898698 | 1605 |
| NEU-DET | A | 838661549 | 1259 |
| NEU-DET | B | 2073899285 | 1259 |

Each dataset.yaml points to the complete sealed acquisition pool and the same fixed development validation list. It has no test key. No annotation file was opened to prepare these manifests. Train configs retain every recorded R0 training argument, changing only seed, data, name and project; save_dir is runtime-generated.

## Dormant training commands

Run only in a separately authorized session after checkpoint/config/data hash checks and a new four-run registry freeze. Do not pass these configs to the historical 440-cell runner.

```powershell
& "C:/Users/user/Desktop/vlm/Defect_VLM_Project/.python311/python.exe" -c "import yaml; from ultralytics import YOLO; c=yaml.safe_load(open('C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/docs/design/full_data_reference_20260908/GC10-DET/train_A.yaml')); YOLO(c['model']).train(**c)"
& "C:/Users/user/Desktop/vlm/Defect_VLM_Project/.python311/python.exe" -c "import yaml; from ultralytics import YOLO; c=yaml.safe_load(open('C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/docs/design/full_data_reference_20260908/GC10-DET/train_B.yaml')); YOLO(c['model']).train(**c)"
& "C:/Users/user/Desktop/vlm/Defect_VLM_Project/.python311/python.exe" -c "import yaml; from ultralytics import YOLO; c=yaml.safe_load(open('C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/docs/design/full_data_reference_20260908/NEU-DET/train_A.yaml')); YOLO(c['model']).train(**c)"
& "C:/Users/user/Desktop/vlm/Defect_VLM_Project/.python311/python.exe" -c "import yaml; from ultralytics import YOLO; c=yaml.safe_load(open('C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/docs/design/full_data_reference_20260908/NEU-DET/train_B.yaml')); YOLO(c['model']).train(**c)"
```

## Dormant validation commands

```powershell
& "C:/Users/user/Desktop/vlm/Defect_VLM_Project/.python311/python.exe" -c "from ultralytics import YOLO; YOLO('C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/runs/full_data_reference_20260908/GC10-DET_full_reference_A/weights/best.pt').val(data='C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/docs/design/full_data_reference_20260908/GC10-DET/dataset.yaml', split='val', imgsz=640, batch=8, device=0, workers=4, conf=0.001, iou=0.7, max_det=300, plots=False, save_json=False, verbose=False, project='C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/runs/full_data_reference_20260908/GC10-DET_full_reference_A', name='sealed_development_eval')"
& "C:/Users/user/Desktop/vlm/Defect_VLM_Project/.python311/python.exe" -c "from ultralytics import YOLO; YOLO('C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/runs/full_data_reference_20260908/GC10-DET_full_reference_B/weights/best.pt').val(data='C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/docs/design/full_data_reference_20260908/GC10-DET/dataset.yaml', split='val', imgsz=640, batch=8, device=0, workers=4, conf=0.001, iou=0.7, max_det=300, plots=False, save_json=False, verbose=False, project='C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/runs/full_data_reference_20260908/GC10-DET_full_reference_B', name='sealed_development_eval')"
& "C:/Users/user/Desktop/vlm/Defect_VLM_Project/.python311/python.exe" -c "from ultralytics import YOLO; YOLO('C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/runs/full_data_reference_20260908/NEU-DET_full_reference_A/weights/best.pt').val(data='C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/docs/design/full_data_reference_20260908/NEU-DET/dataset.yaml', split='val', imgsz=640, batch=8, device=0, workers=4, conf=0.001, iou=0.7, max_det=300, plots=False, save_json=False, verbose=False, project='C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/runs/full_data_reference_20260908/NEU-DET_full_reference_A', name='sealed_development_eval')"
& "C:/Users/user/Desktop/vlm/Defect_VLM_Project/.python311/python.exe" -c "from ultralytics import YOLO; YOLO('C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/runs/full_data_reference_20260908/NEU-DET_full_reference_B/weights/best.pt').val(data='C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/docs/design/full_data_reference_20260908/NEU-DET/dataset.yaml', split='val', imgsz=640, batch=8, device=0, workers=4, conf=0.001, iou=0.7, max_det=300, plots=False, save_json=False, verbose=False, project='C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/runs/full_data_reference_20260908/NEU-DET_full_reference_B', name='sealed_development_eval')"
```

## Locked-test evaluation preparation

The historical evaluation launcher requires exactly 440 historical cells and its own authorization seal. It cannot be reused for these four new runs. A separate four-cell checkpoint registry and explicit evaluation authorization are required. The following API calls describe the intended post-authorization evaluation; they do not provide the missing guard/authorization wrapper. Existing locked-final YAML paths are listed by filename only; no final data or outcome was loaded in this session.

```powershell
& "C:/Users/user/Desktop/vlm/Defect_VLM_Project/.python311/python.exe" -c "from ultralytics import YOLO; YOLO('C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/runs/full_data_reference_20260908/GC10-DET_full_reference_A/weights/best.pt').val(data='C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/confirmatory_v1_1/final_analysis/final_evaluation/data_cache/GC10-DET/locked_final.yaml', split='val', imgsz=640, batch=8, device=0, workers=4, conf=0.001, iou=0.7, max_det=300, plots=False, save_json=False, verbose=False, project='C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/runs/full_data_reference_20260908/GC10-DET_full_reference_A', name='authorized_locked_test_eval')"
& "C:/Users/user/Desktop/vlm/Defect_VLM_Project/.python311/python.exe" -c "from ultralytics import YOLO; YOLO('C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/runs/full_data_reference_20260908/GC10-DET_full_reference_B/weights/best.pt').val(data='C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/confirmatory_v1_1/final_analysis/final_evaluation/data_cache/GC10-DET/locked_final.yaml', split='val', imgsz=640, batch=8, device=0, workers=4, conf=0.001, iou=0.7, max_det=300, plots=False, save_json=False, verbose=False, project='C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/runs/full_data_reference_20260908/GC10-DET_full_reference_B', name='authorized_locked_test_eval')"
& "C:/Users/user/Desktop/vlm/Defect_VLM_Project/.python311/python.exe" -c "from ultralytics import YOLO; YOLO('C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/runs/full_data_reference_20260908/NEU-DET_full_reference_A/weights/best.pt').val(data='C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/confirmatory_v1_1/final_analysis/final_evaluation/data_cache/NEU-DET/locked_final.yaml', split='val', imgsz=640, batch=8, device=0, workers=4, conf=0.001, iou=0.7, max_det=300, plots=False, save_json=False, verbose=False, project='C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/runs/full_data_reference_20260908/NEU-DET_full_reference_A', name='authorized_locked_test_eval')"
& "C:/Users/user/Desktop/vlm/Defect_VLM_Project/.python311/python.exe" -c "from ultralytics import YOLO; YOLO('C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/runs/full_data_reference_20260908/NEU-DET_full_reference_B/weights/best.pt').val(data='C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/confirmatory_v1_1/final_analysis/final_evaluation/data_cache/NEU-DET/locked_final.yaml', split='val', imgsz=640, batch=8, device=0, workers=4, conf=0.001, iou=0.7, max_det=300, plots=False, save_json=False, verbose=False, project='C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/runs/full_data_reference_20260908/NEU-DET_full_reference_B', name='authorized_locked_test_eval')"
```

The locked-final adapter uses its `val` key for the locked-test image list; verify that mapping and all split hashes inside the new authorization wrapper before use.

## Exact paths and hashes

See `full_data_reference_20260908/runplan.json` for all config, manifest, checkpoint-output paths and SHA-256 values. Pretrained model, 100 epochs, SGD, batch 8, 640px, augmentation, cold start and validation-selected best.pt remain identical to the archived R0 args. Expected last.pt is retained for audit only. Four independent cold starts; no warm-start or acquisition.
