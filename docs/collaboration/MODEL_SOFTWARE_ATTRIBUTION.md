# Model and Software Attribution

No external model weights are redistributed on this branch. Names, revisions, roles, hashes, and upstream terms are recorded so collaborators can obtain them from their owners.

## Final controlled study

| Component | Exact use | Role | Version / seal | Upstream / license |
|---|---|---|---|---|
| DINOv2 Small | `facebook/dinov2-small` | frozen image feature extractor for acquisition only | HF revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`; safetensors SHA-256 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` | https://huggingface.co/facebook/dinov2-small and https://github.com/facebookresearch/dinov2; model card/repository identify Apache-2.0 for this checkpoint/code |
| YOLOv8n | `yolov8n.pt` | downstream object detector, cold-start at every budget | Ultralytics `8.4.90`; checkpoint SHA-256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`; embedded upstream version `8.0.0.dev0` | https://github.com/ultralytics/ultralytics; AGPL-3.0 open-source license in current upstream, with separate enterprise licensing offered. Users must verify terms applicable to the exact version and deployment. |
| Transformers | model/processor loading for DINO and historical VLM/OWL-ViT | library | controlled environment `5.13.0` | https://github.com/huggingface/transformers; Apache-2.0 |
| PyTorch | training and inference | framework | `2.12.1+cu126`, CUDA build 12.6, cuDNN 91002 | https://pytorch.org; consult upstream license for the installed build |
| NumPy | PCG64 selection/statistics | library | `2.4.6` | https://numpy.org; BSD-3-Clause |
| Pillow | image decoding/preprocessing | library | `12.3.0` | https://python-pillow.org; HPND-style Pillow license |

## Historical evidence models

| Model | Exact identifier | Historical role | Upstream terms |
|---|---|---|---|
| Qwen2-VL | `Qwen/Qwen2-VL-2B-Instruct` | VLM explanations, structured validity, forced compliance, and grounded prompt experiments | https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct; model card identifies Apache-2.0 |
| Sentence-BERT | `sentence-transformers/all-MiniLM-L6-v2` | explanation/response semantic similarity | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2; model card identifies Apache-2.0 |
| OWL-ViT | `google/owlvit-base-patch32` | open-vocabulary pseudo boxes for pseudo-groundedness diagnostics | https://huggingface.co/google/owlvit-base-patch32; model card identifies Apache-2.0 |
| Additional Qwen-family checkpoints | Qwen3-VL-2B and Qwen2.5-VL-3B variants named in the paired historical comparison | cross-model compliance/grounding gate only | Exact Hub repository IDs/revisions are not consistently preserved in the public historical tree: `VERIFY REQUIRED`; no weights included |
| timm Wide ResNet-50-2 | `timm/wide_resnet50_2.racm_in1k` | KSDD2 PatchCore feature extraction | obtain through timm/Hugging Face and verify exact checkpoint terms |

Historical model comparisons failed their frozen compliance/grounding gates and are not part of the final controlled selector claim.

## Reuse and attribution rules

1. Download weights from the upstream owner/model hub; do not copy them into this repository.
2. Verify the license at the exact revision used. A repository's current license can change and may not cover third-party training data or every derivative checkpoint.
3. Cite the model paper and software used in the manuscript.
4. Ultralytics licensing is especially relevant to distribution or commercial deployment. This research release reports academic use and does not grant a downstream YOLO license.
5. Dataset terms are separate from model/software terms; see `DATASETS.md`.
