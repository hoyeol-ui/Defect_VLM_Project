#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

python3 scripts/docs/check_handoff_package.py

if [[ "${1:-}" == "--check-only" ]]; then
  exit 0
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo
  echo "This helper opens files with macOS 'open'. On this OS, use the printed paths manually."
  exit 0
fi

open "docs/preview.html"
open "docs/vlm_gt_free_al_workflow.html"
open "docs/macbook_handoff_guide_20260712.md"
open "docs/research_context_handoff_20260712.md"
open "docs/results/v10b_seed42_documentation_20260712_215841/V10b_Seed42_Development_Gate_Updated_Report.docx"
open "docs/results/v10b_seed42_documentation_20260712_215841/V10b_Seed42_Development_Gate_Updated_Report.md"

echo
echo "Opened the MacBook documentation handoff package."
