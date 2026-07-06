"""
Direction-fix experiment launcher guide.

This script intentionally does not duplicate the full YOLO training runner.
Instead, it prints reproducible commands for the v4 direction-fix experiments
now supported by run_al_yolo_ablation_v3_minimal.py via environment variables.

Recommended flow:
1) Generate priority sensitivity CSVs.
2) Run dry selection checks.
3) Run YOLO retraining for selected promising variants.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--priority-sensitivity-dir",
        default="outputs/priority_sensitivity_*",
        help="Directory or glob containing penalty_*/priority_scores_pseudo.csv variants.",
    )
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated AL seeds.")
    parser.add_argument("--dry-run-first", action="store_true", help="Print dry-run commands before train commands.")
    return parser.parse_args()


def main():
    args = parse_args()
    runner = "python scripts/02_active_learning/run_al_yolo_ablation_v3_minimal.py"
    core = "Random,ConsistencyOnly,GroundednessOnlySoft,CombinedSoftPenalty,LowPrioritySoft"
    ablation = "Random,ConsistencyOnly,CombinedNoPenalty,CombinedNoGroundedness,CombinedWeighted,CombinedRankCalibrated"
    balanced = "Random,RandomClassBalanced,CombinedSoftPenalty,CombinedSoftPenaltyClassBalanced,LowPrioritySoft,LowPrioritySoftClassBalanced"

    print("=" * 88)
    print("V4 direction-fix experiments are run through the v3 runner with env vars.")
    print("=" * 88)
    print()
    print("[0] Generate priority sensitivity variants")
    print("python scripts/01_score_generation/make_priority_scores_sensitivity.py \\")
    print("  --penalties 0,0.1,0.2,0.5,1.0 \\")
    print("  --groundedness-weights 0,0.25,0.5,1.0")
    print()
    print("[1] Missing-box penalty sensitivity")
    print("# Replace <CSV> with outputs/priority_sensitivity_*/penalty_*/priority_scores_pseudo.csv")
    if args.dry_run_first:
        print(f"AL_DRY_RUN_ONLY=1 AL_SEEDS={args.seeds} AL_PRIORITY_CSV=<CSV> AL_STRATEGIES={core} {runner}")
    print(f"AL_SEEDS={args.seeds} AL_PRIORITY_CSV=<CSV> AL_STRATEGIES={core} {runner}")
    print()
    print("[2] Groundedness and calibration ablation")
    if args.dry_run_first:
        print(f"AL_DRY_RUN_ONLY=1 AL_SEEDS={args.seeds} AL_STRATEGIES={ablation} {runner}")
    print(f"AL_SEEDS={args.seeds} AL_STRATEGIES={ablation} AL_WEIGHTED_BETA=0.5 AL_WEIGHTED_GAMMA=0.2 {runner}")
    print()
    print("[3] Class-balanced direction check")
    if args.dry_run_first:
        print(f"AL_DRY_RUN_ONLY=1 AL_SEEDS={args.seeds} AL_STRATEGIES={balanced} {runner}")
    print(f"AL_SEEDS={args.seeds} AL_STRATEGIES={balanced} {runner}")
    print()
    print("[4] Seed expansion after cheap audits look stable")
    print(f"AL_SEEDS=0,1,2,3,4,42,43,44,100,123 AL_STRATEGIES={core} {runner}")
    print()
    print("[5] Post-hoc audits to run after each AL result")
    print("python scripts/03_analysis/analyze_selection_direction_issue.py --run-dir <RUN_DIR> --priority-dir <PRIORITY_DIR>")
    print("python scripts/03_analysis/audit_no_pseudo_box_gt_presence.py --run-dir <RUN_DIR> --priority-dir <PRIORITY_DIR>")
    print("python scripts/03_analysis/analyze_selection_enrichment.py --run-dir <RUN_DIR> --priority-dir <PRIORITY_DIR>")
    print("python scripts/03_analysis/analyze_round_pool_state.py --run-dir <RUN_DIR> --priority-dir <PRIORITY_DIR>")


if __name__ == "__main__":
    main()
