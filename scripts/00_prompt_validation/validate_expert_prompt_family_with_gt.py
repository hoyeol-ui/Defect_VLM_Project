"""
GT-based prompt-family pilot validation wrapper.

This script validates whether the expert prompt family can elicit location,
scale, and appearance descriptions aligned with benchmark GT annotations. This
is a pilot validation step only. GT information from this script must not be
used for active learning acquisition.

The implementation imports the legacy compatibility script instead of copying
its model-loading and evaluation logic.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from grounded_prompt_experiment import main  # noqa: E402


if __name__ == "__main__":
    main()
