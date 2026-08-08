"""Validate one explicit frozen-run config before any downstream action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from defect_al.data.identity import compute_file_sha256
from defect_al.preflight import build_preflight_report


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# Frozen-run preflight: {report['run_id']}",
        "",
        f"- Verdict: **{report['status']}**",
        f"- Harness version: `{report['harness_version']}`",
        f"- Config SHA-256: `{report['config_sha256']}`",
        f"- Git commit: `{report['git_commit']}`",
        f"- Git dirty: `{report['git_dirty']}`",
        "",
        "## Manifest seals and identities",
        "",
        "| split | rows | canonical IDs | image SHA-256s | manifest SHA-256 |",
        "|---|---:|---:|---:|---|",
    ]
    for split, item in report["manifests"].items():
        lines.append(
            f"| {split} | {item['row_count']} | {item['unique_canonical_ids']} | "
            f"{item['unique_image_sha256s']} | `{item['manifest_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Pairwise leakage checks",
            "",
            "| left | right | identity | overlap | example |",
            "|---|---|---|---:|---|",
        ]
    )
    for item in report["split_overlaps"]:
        lines.append(
            f"| {item['left_split']} | {item['right_split']} | {item['identity_type']} | "
            f"{item['overlap_count']} | {item['example'] or ''} |"
        )
    lines.extend(
        [
            "",
            "Preflight performs no selection, training, inference, or model evaluation.",
            "A PASS only allows a separately authorized downstream protocol to continue.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()

    report = build_preflight_report(args.config, project_root=args.project_root)
    report["preflight_cli_sha256"] = compute_file_sha256(Path(__file__).resolve())
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preflight_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "preflight_report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "run_id": report["run_id"],
        "output_dir": str(output_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
