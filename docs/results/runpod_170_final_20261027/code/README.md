# Archived scientific source reference

Training source and local final-test source are preserved for method inspection. These are source references, not an independently runnable public evaluation kit. The original absolute paths, package seals, data, weights and GPU authorization dependencies are intentionally not reconstructed. Do not execute these files to repeat the completed final test. The supported local command in this export is `../scripts/verify_export.py`, which uses only saved JSON and SHA-256 with the Python standard library.

`protocol/EFFECTIVE_CONFIG.json` documents the actual overrides: batch32 and the five learner seeds supersede technical base defaults. Final YOLO26 evaluation uses its one-to-one NMS-free path; acquisition CPU NMS is separately defined.
