# Full 440 Training Authorization

- User authorization: **GO**
- Authorized scope: remaining 434 detector training jobs
- Reusable completed cells: 6
- Planned scientific training cells: 440
- Actual final evaluation: **FORBIDDEN / LOCKED**
- Protocol: v1.1, SHA-256 `6836f452d22c3de26dfa2680c45d0b5cb8f925b1a9274789c4ef81a1770f0b10`

Each failed infrastructure attempt must retry the same cell with the same
dataset, acquisition seed, training seed, manifest, checkpoint, code,
environment and learner configuration. Seed replacement and silent deletion are
forbidden. Scientific outcomes stay hidden during execution.

After 440/440 training completion, the runner performs only an integrity audit.
It must not create final authorization or evaluate the final split.

