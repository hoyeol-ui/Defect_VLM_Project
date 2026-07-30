"""Generate frozen multi-prompt structured VLM responses without using GT.

The default mode is ``--dry-run`` friendly: it validates the manifest, checks
locked-final exclusions, and writes the exact image/prompt plan without loading
the VLM. Actual inference is deterministic and resumable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from validity_prompt_family import prompt_family_hash, prompt_records  # noqa: E402


DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_LOCKED_MANIFESTS = [
    PROJECT_ROOT
    / "runs"
    / "evaluation_protocol_v7"
    / "eval_protocol_20260711_173723"
    / "final_test_v7.csv",
    PROJECT_ROOT
    / "runs"
    / "gc10_taxonomy_protocol"
    / "gc10_protocol_20260715"
    / "gc10_final_test_locked.csv",
]


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


def first_present(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def read_manifest(
    path: Path,
    default_dataset: str,
    default_split_role: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, raw in enumerate(csv.DictReader(handle), start=1):
            image_path = first_present(raw, ("image_path", "resolved_image_path", "path"))
            image_id = first_present(raw, ("image_id", "sample_id", "image_name", "filename"))
            dataset = first_present(raw, ("dataset", "dataset_type")) or default_dataset
            split_role = first_present(raw, ("split_role", "protocol_split")) or default_split_role
            if not image_path:
                raise ValueError(f"Manifest row {index} has no image path")
            if not image_id:
                image_id = Path(image_path).stem
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": str(Path(image_path).expanduser().resolve()),
                    "dataset": dataset or "unknown",
                    "split_role": split_role or "unknown",
                }
            )
    return rows


def validate_rows(rows: list[dict[str, str]], locked_names: set[str]) -> None:
    seen: set[str] = set()
    errors: list[str] = []
    for row in rows:
        image_id = row["image_id"]
        role = row["split_role"].strip().casefold().replace("-", "_")
        name = Path(row["image_path"]).name.casefold()
        if image_id in seen:
            errors.append(f"duplicate image_id: {image_id}")
        seen.add(image_id)
        if "final" in role or role in {"locked", "test_locked"}:
            errors.append(f"forbidden split_role for {image_id}: {row['split_role']}")
        if name in locked_names:
            errors.append(f"locked-final filename for {image_id}: {name}")
        if not Path(row["image_path"]).is_file():
            errors.append(f"missing image for {image_id}: {row['image_path']}")
    if errors:
        preview = "\n".join(f"- {e}" for e in errors[:30])
        raise RuntimeError(f"Manifest validation failed ({len(errors)} issues):\n{preview}")


def exclude_locked_rows(
    rows: list[dict[str, str]], locked_names: set[str]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    eligible: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    for row in rows:
        role = row["split_role"].strip().casefold().replace("-", "_")
        name = Path(row["image_path"]).name.casefold()
        reasons: list[str] = []
        if name in locked_names:
            reasons.append("filename_in_locked_final_manifest")
        if "final" in role or role in {"locked", "test_locked"}:
            reasons.append(f"forbidden_split_role:{row['split_role']}")
        if reasons:
            excluded.append({**row, "exclusion_reason": "|".join(reasons)})
        else:
            eligible.append(row)
    return eligible, excluded


def read_completed_pairs(path: Path) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                completed.add((str(row.get("image_id")), str(row.get("prompt_id"))))
    return completed


def load_model(model_id: str):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
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
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--default-dataset", default="unknown")
    parser.add_argument("--default-split-role", default="development_pilot")
    parser.add_argument("--locked-manifest", type=Path, action="append", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=20260715)
    parser.add_argument("--prompt-limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    locked_paths = args.locked_manifest or DEFAULT_LOCKED_MANIFESTS
    locked_names = read_locked_names(locked_paths)
    source_rows = read_manifest(args.manifest, args.default_dataset, args.default_split_role)
    eligible_rows, excluded_rows = exclude_locked_rows(source_rows, locked_names)
    rows = eligible_rows
    if args.max_images is not None:
        sampler = random.Random(args.sample_seed)
        rows = sampler.sample(rows, k=min(args.max_images, len(rows)))
    if not rows:
        raise RuntimeError("No eligible non-final images remain after exclusion")
    validate_rows(rows, locked_names)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exclusion_path = args.output_dir / "locked_final_exclusions.csv"
    if excluded_rows:
        with exclusion_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(excluded_rows[0].keys()))
            writer.writeheader()
            writer.writerows(excluded_rows)
    else:
        exclusion_path.write_text(
            "image_id,image_path,dataset,split_role,exclusion_reason\n",
            encoding="utf-8-sig",
        )
    prompts = prompt_records()
    if args.prompt_limit is not None:
        if args.prompt_limit < 1:
            raise ValueError("--prompt-limit must be at least 1")
        prompts = prompts[: args.prompt_limit]
    plan_rows = [
        {**row, **prompt}
        for row in rows
        for prompt in prompts
    ]
    plan_path = args.output_dir / "structured_vlm_prompt_plan.csv"
    with plan_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plan_rows[0].keys()))
        writer.writeheader()
        writer.writerows(plan_rows)

    config = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "locked_manifests": [str(p.resolve()) for p in locked_paths],
        "model_id": args.model_id,
        "prompt_family_hash": prompt_family_hash(),
        "source_manifest_rows": len(source_rows),
        "eligible_after_locked_final_exclusion": len(eligible_rows),
        "locked_final_exclusions": len(excluded_rows),
        "images": len(rows),
        "prompts_per_image": len(prompts),
        "planned_responses": len(plan_rows),
        "sample_seed": args.sample_seed,
        "deterministic_decoding": True,
        "gt_visible_to_model": False,
        "final_test_allowed": False,
        "dry_run": args.dry_run,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.dry_run:
        print(f"[DRY-RUN] images={len(rows)} planned_responses={len(plan_rows)}")
        print(f"[PLAN] {plan_path}")
        return

    output_path = args.output_dir / "structured_vlm_responses.jsonl"
    completed = read_completed_pairs(output_path)
    processor, model = load_model(args.model_id)
    with output_path.open("a", encoding="utf-8") as handle:
        for index, item in enumerate(plan_rows, start=1):
            key = (item["image_id"], item["prompt_id"])
            if key in completed:
                continue
            raw_response = query_model(
                processor,
                model,
                item["image_path"],
                item["prompt_text"],
                args.max_new_tokens,
            )
            record = {
                key: item[key]
                for key in (
                    "image_id", "image_path", "dataset", "split_role",
                    "prompt_id", "prompt_family_id", "prompt_family_hash",
                )
            }
            record.update(
                {
                    "model_id": args.model_id,
                    "decoding": "deterministic",
                    "raw_response": raw_response,
                }
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(plan_rows)}] {item['image_id']} {item['prompt_id']}")
    print(f"[DONE] {output_path}")


if __name__ == "__main__":
    main()
