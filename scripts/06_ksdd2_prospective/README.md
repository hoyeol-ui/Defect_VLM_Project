# KSDD2 prospective authorization branch

This folder implements the frozen G1-G4 design in `protocol.md`. Raw outputs
are written under `runs/ksdd2_patchcore_prospective_v1/` and remain ignored by
Git.

For the full Korean operator guide, decision interpretation, and slide-ready
wording, see `OPERATOR_RUNBOOK_KO.md`.

## Important boundary

Run `prepare` and inspect the manifests first. Do not run `freeze` until the
current evidence-freeze changes have been intentionally committed. Official
freeze refuses a dirty worktree.

## Safe setup and Stage 0

```powershell
cd "C:\Users\user\Desktop\vlm\Defect_VLM_Project"

.\.python311\python.exe -m pip install `
  -r scripts\06_ksdd2_prospective\requirements.txt

.\.python311\python.exe scripts\06_ksdd2_prospective\run.py download --extract
.\.python311\python.exe scripts\06_ksdd2_prospective\run.py prepare
.\.python311\python.exe scripts\06_ksdd2_prospective\run.py verify-pre-freeze
```

After reviewing and committing the protocol:

```powershell
.\.python311\python.exe scripts\06_ksdd2_prospective\run.py freeze
```

## Sequential G1-G4 execution

```powershell
# G1: seed-0 reference, development score, then post-hoc GT audit
.\.python311\python.exe scripts\06_ksdd2_prospective\run.py reveal-initial --seed 0
.\.python311\python.exe scripts\06_ksdd2_prospective\run.py score-g1
.\.python311\python.exe scripts\06_ksdd2_prospective\run.py analyze-g1

# Runs only when g1_decision.json is PASS.
.\.python311\python.exe scripts\06_ksdd2_prospective\run.py score-selection --all-seeds
.\.python311\python.exe scripts\06_ksdd2_prospective\run.py select --all-seeds
.\.python311\python.exe scripts\06_ksdd2_prospective\run.py analyze-g2-g4
.\.python311\python.exe scripts\06_ksdd2_prospective\run.py verify-run
```

The final command prints `STOP`, `DISCOVERY_ONLY_STOP`, or `BOUNDED_SCREEN`.
Nothing in this folder trains a detector or reads the official test image or
mask content.

## Tests

```powershell
.\.python311\python.exe scripts\06_ksdd2_prospective\test_core.py
.\.python311\python.exe scripts\06_ksdd2_prospective\smoke_patchcore.py
```
