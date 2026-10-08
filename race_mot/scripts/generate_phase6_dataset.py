"""Generate and audit the frozen Phase 6 MOT17 paired-rollout dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from race_mot.evaluation.phase6_dataset import generate_phase6_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence", action="append", nargs=2, metavar=("SEQUENCE_DIR", "BASELINE_RUN_DIR"), required=True,
        help="training-role MOT sequence and its frozen detector-every-frame run directory",
    )
    parser.add_argument("--roles", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prior-history-samples", type=int, default=48)
    args = parser.parse_args()
    if args.prior_history_samples < 1:
        parser.error("--prior-history-samples must be positive")
    manifest = generate_phase6_dataset(
        sequences=[(Path(sequence), Path(run_dir)) for sequence, run_dir in args.sequence],
        roles_path=args.roles,
        protocol_path=args.protocol,
        output_root=args.output_root,
        prior_history_samples_per_sequence=args.prior_history_samples,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

