# DCAL-XAI

One compact experiment folder for detector-coupled active learning and
grounded selection explanations.

## Files

- `config.json`: frozen experiment parameters and gates
- `protocol.md`: hypotheses, roles, stopping rules, and allowed claims
- `core.py`: label-free difficulty, diversity, and explanation logic
- `run.py`: audit, acquisition, and confirmation stages
- `test_core.py`: synthetic unit tests; no training

All generated artifacts go to `runs/dcal_xai/`; no additional script folders
are created.

## Safe execution order

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe scripts\04_dcal_xai\test_core.py

.\.venv\Scripts\python.exe scripts\04_dcal_xai\run.py audit

.\.venv\Scripts\python.exe scripts\04_dcal_xai\run.py acquire --dry-run
```

The commands above do not train a model. Review `runs/dcal_xai/gc10_r1/audit.json`
and `acquisition_plan.json` before any training.

One-seed smoke acquisition, written to an isolated output:

```powershell
.\.venv\Scripts\python.exe scripts\04_dcal_xai\run.py acquire `
  --seeds 0 `
  --output-dir runs\dcal_xai\smoke_s0
```

Main five-seed acquisition:

```powershell
.\.venv\Scripts\python.exe scripts\04_dcal_xai\run.py acquire
```

Development confirmation is blocked unless the main five-seed selection gate
passes:

```powershell
.\.venv\Scripts\python.exe scripts\04_dcal_xai\run.py confirm
```

The runner never reads `gc10_final_test_locked.csv`.

## V2: data scope and learner-capacity reset

The R1 detector-difficulty selector failed its selection gate. V2 therefore
does not tune another uncertainty formula. It first audits the initial labeled
budget, then separates data-policy effects from detector-capacity effects.

Selection-only budget audit (no training):

```powershell
.\.venv\Scripts\python.exe scripts\04_dcal_xai\plan_v2_budget.py
.\.venv\Scripts\python.exe scripts\04_dcal_xai\plan_v2_budget_extended.py
```

The frozen result is `DINOClusterCoverageK40` at 140 initial labels. Audit the
five new acquisition realizations without training:

```powershell
.\.venv\Scripts\python.exe scripts\04_dcal_xai\run_v2_backbone.py audit `
  --output-dir runs\dcal_xai\v2_backbone_main
```

Before the smoke run, make sure both frozen pretrained weights exist:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8s.pt')"
```

Run and inspect the isolated four-model technical smoke before the main run:

```powershell
.\.venv\Scripts\python.exe scripts\04_dcal_xai\run_v2_backbone.py smoke `
  --output-dir runs\dcal_xai\v2_backbone_smoke

Get-Content runs\dcal_xai\v2_backbone_smoke\smoke_validation.json
Import-Csv runs\dcal_xai\v2_backbone_smoke\smoke_metrics.csv | Format-Table -AutoSize
```

After inspecting a smoke `PASS`, run the frozen 30-model YOLOv8n screen:

```powershell
.\.venv\Scripts\python.exe scripts\04_dcal_xai\run_v2_backbone.py screen `
  --output-dir runs\dcal_xai\v2_backbone_main
```

Inspect `v8n_screen_decision.md`. Only a screen `PASS` permits expansion to the
remaining YOLOv8s models and completion of the 60-model factorial:

```powershell
.\.venv\Scripts\python.exe scripts\04_dcal_xai\run_v2_backbone.py main `
  --output-dir runs\dcal_xai\v2_backbone_main
```

Neither V2 runner reads or evaluates the locked final test. A main `PASS`
authorizes only a detector error-association audit, not another AL round.

## V2.3: detector-signal validity without new training

V2.3 uses only the 15 existing `Random140 x YOLOv8n` checkpoints. It seals
development predictions without GT and joins GT only during post-hoc analysis.

```powershell
.\.venv\Scripts\python.exe scripts\04_dcal_xai\test_v2_detector_signal.py
.\.venv\Scripts\python.exe scripts\04_dcal_xai\run_v2_detector_signal_validity.py audit
.\.venv\Scripts\python.exe scripts\04_dcal_xai\run_v2_detector_signal_validity.py predict
.\.venv\Scripts\python.exe scripts\04_dcal_xai\run_v2_detector_signal_validity.py analyze
```

The frozen primary gate is documented in
`v2_detector_signal_validity_protocol.md`. Secondary signals cannot replace a
failed primary result after inspection. This runner performs no training and
never reads the locked final test.
