"""Run the frozen one-prompt forced-binary VLM compliance probe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_LOCKED = [
    PROJECT_ROOT / "runs" / "evaluation_protocol_v7" / "eval_protocol_20260711_173723" / "final_test_v7.csv",
    PROJECT_ROOT / "runs" / "gc10_taxonomy_protocol" / "gc10_protocol_20260715" / "gc10_final_test_locked.csv",
]
PROMPT_ID = "forced_binary_grounding_v2_p1"
PROMPT = """Inspect only the pixels in this industrial surface image. The image may or may not contain a visible surface defect.

Return exactly one JSON object with exactly these four keys and no markdown:
defect_present, bbox_norm, appearance, confidence.

Rules:
- defect_present must be the JSON boolean true or false. Never use null, unknown, or abstain.
- If defect_present is true, bbox_norm must contain four numeric coordinates [x1,y1,x2,y2] between 0 and 1 around the clearest anomalous region.
- If defect_present is false, bbox_norm must be null.
- confidence must be a numeric value between 0 and 1.
- No example values are provided. Make every value from the current image rather than copying the instruction.
- Do not infer from filenames, folders, metadata, or class names. Use visible pixels only."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_locked_names(paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                for key in ("image_name", "filename", "image_path", "resolved_image_path"):
                    value = str(row.get(key, "")).strip()
                    if value:
                        names.add(Path(value).name.casefold())
    return names


def load_model(
    model_id: str,
    revision: str | None,
    offload_folder: Path,
    gpu_max_memory: str,
    cpu_max_memory: str,
):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        revision=revision,
        dtype="auto",
        device_map="auto",
        max_memory={0: gpu_max_memory, "cpu": cpu_max_memory},
        offload_folder=str(offload_folder),
        low_cpu_mem_usage=True,
    )
    model.eval()
    return processor, model


def query_model(processor, model, image_path: str, prompt: str, max_new_tokens: int) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    input_device = next(model.parameters()).device
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    trimmed = [out[len(inp) :] for inp, out in zip(inputs["input_ids"], generated)]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def model_revision(model_id: str, revision: str | None) -> str | None:
    try:
        from huggingface_hub import model_info

        return str(model_info(model_id, revision=revision).sha)
    except Exception:
        return None


def read_manifest(path: Path, locked: set[str]) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, raw in enumerate(csv.DictReader(handle), start=1):
            image_id = str(raw.get("image_id", "")).strip()
            image_path = Path(str(raw.get("image_path", "")).strip()).resolve()
            split_role = str(raw.get("split_role", "")).strip()
            if not image_id or not image_path.is_file():
                raise RuntimeError(f"Invalid manifest row {index}: {image_id} {image_path}")
            if "final" in split_role.casefold() or image_path.name.casefold() in locked:
                raise RuntimeError(f"Forbidden final/locked view: {image_id}")
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "dataset": str(raw.get("dataset", "GC10-DET")),
                    "split_role": split_role or "development_paired_oracle_compliance_audit",
                }
            )
    if len({row["image_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate image_id in manifest")
    return rows


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    values = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                values.add(str(json.loads(line).get("image_id", "")))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--locked-manifest", type=Path, action="append", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--gpu-max-memory", default="6.5GiB")
    parser.add_argument("--cpu-max-memory", default="24GiB")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    locked_paths = args.locked_manifest or DEFAULT_LOCKED
    rows = read_manifest(args.manifest, read_locked_names(locked_paths))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_hash = sha256_bytes(PROMPT.encode("utf-8"))
    plan_path = args.output_dir / "forced_compliance_prompt_plan.csv"
    with plan_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["image_id", "image_path", "dataset", "split_role", "prompt_id", "prompt_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "prompt_id": PROMPT_ID, "prompt_sha256": prompt_hash})
    config = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "model_id": args.model_id,
        "requested_revision": args.revision,
        "resolved_model_revision": args.revision if args.dry_run else model_revision(args.model_id, args.revision),
        "prompt_id": PROMPT_ID,
        "prompt_sha256": prompt_hash,
        "prompt_text": PROMPT,
        "views": len(rows),
        "planned_responses": len(rows),
        "deterministic_decoding": True,
        "gpu_max_memory": args.gpu_max_memory,
        "cpu_max_memory": args.cpu_max_memory,
        "gt_visible_to_model": False,
        "final_test_allowed": False,
        "dry_run": args.dry_run,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.dry_run:
        print(f"[DRY-RUN] views={len(rows)} planned_responses={len(rows)}")
        print(f"[PLAN] {plan_path}")
        return

    output_path = args.output_dir / "forced_compliance_responses.jsonl"
    done = completed_ids(output_path)
    offload_folder = args.output_dir / "model_offload"
    offload_folder.mkdir(parents=True, exist_ok=True)
    processor, model = load_model(
        args.model_id,
        args.revision,
        offload_folder,
        args.gpu_max_memory,
        args.cpu_max_memory,
    )
    with output_path.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            if row["image_id"] in done:
                continue
            raw_response = query_model(
                processor, model, row["image_path"], PROMPT, args.max_new_tokens
            )
            record = {
                **row,
                "prompt_id": PROMPT_ID,
                "prompt_sha256": prompt_hash,
                "model_id": args.model_id,
                "decoding": "deterministic",
                "raw_response": raw_response,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(rows)}] {row['image_id']}")
    print(f"[DONE] {output_path}")


if __name__ == "__main__":
    main()
