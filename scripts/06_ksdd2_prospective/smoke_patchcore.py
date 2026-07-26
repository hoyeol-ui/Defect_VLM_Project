"""Small CPU/GPU smoke test for the anomalib PatchCore adapter."""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config, read_csv, verify_seal
from patchcore_scoring import score_rows


def main() -> None:
    config = copy.deepcopy(load_config())
    config["patchcore"].update(
        {
            "backbone": "resnet18",
            "layers": ["layer2", "layer3"],
            "coreset_sampling_ratio": 0.5,
            "num_neighbors": 1,
            "image_size": [64, 64],
            "train_batch_size": 2,
            "predict_batch_size": 2,
            "num_workers": 0,
        }
    )
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        rows = []
        rng = np.random.default_rng(260726)
        for index in range(6):
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[:, :, :] = 64 + index
            image[16:32, 16:32] += rng.integers(0, 10, size=(16, 16, 3), dtype=np.uint8)
            path = root / f"image_{index}.png"
            Image.fromarray(image).save(path)
            rows.append({"sample_id": str(index), "image_path": str(path)})
        output = root / "scores.csv"
        result = score_rows(
            config,
            reference_rows=rows[:2],
            target_rows=rows[2:],
            seed=0,
            output_path=output,
            pretrained_override=False,
        )
        scores = read_csv(output)
        if len(scores) != 4 or not verify_seal(output):
            raise RuntimeError("PatchCore smoke output was incomplete or unsealed")
        if any(set(row) != {"sample_id", "image_path", "patchcore_score", "acquisition_seed", "score_rank"} for row in scores):
            raise RuntimeError("PatchCore smoke output schema changed")
        print(
            {
                "status": "PASS",
                "reference_count": result["reference_count"],
                "target_count": result["target_count"],
                "sealed": True,
            }
        )


if __name__ == "__main__":
    main()
