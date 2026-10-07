"""Command line for the initial RACE-MOT device-feasibility milestone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from race_mot.config import load_config
from race_mot.evaluation.mot import group_by_frame, read_gt
from race_mot.inventory import collect_inventory
from race_mot.stream_probe import probe_source


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="race-mot")
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser(
        "inventory", help="collect read-only Jetson metadata"
    )
    inventory.add_argument(
        "--output", type=Path, help="JSON output path; defaults to stdout"
    )

    probe = commands.add_parser(
        "probe", help="measure read/decode behavior without saving frames"
    )
    probe.add_argument("--input", required=True, help="local video path or RTSP URL")
    probe.add_argument("--duration-sec", type=float, default=30.0)
    probe.add_argument(
        "--output", type=Path, help="JSON output path; defaults to stdout"
    )

    validate = commands.add_parser(
        "validate-config", help="validate a local JSON run configuration"
    )
    validate.add_argument("--config", required=True, type=Path)

    inspect = commands.add_parser(
        "inspect-mot", help="inspect a MOTChallenge gt.txt file"
    )
    inspect.add_argument("--gt", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "inventory":
            report = collect_inventory()
        elif args.command == "probe":
            report = probe_source(args.input, args.duration_sec)
        elif args.command == "validate-config":
            config = load_config(args.config)
            report = {
                "valid": True,
                "mode": config.mode,
                "source_kind": config.source_kind,
                "detector_family": config.detector_family,
                "input_size": list(config.input_size),
                "precision": config.precision,
                "batch_size": config.batch_size,
                "policy": {
                    "threshold_tau": config.threshold_tau,
                    "max_consecutive_skips": config.max_consecutive_skips,
                },
            }
        else:
            boxes = read_gt(args.gt)
            frames = group_by_frame(boxes)
            report = {
                "path": str(args.gt),
                "boxes": len(boxes),
                "frames": len(frames),
                "first_frame": min(frames) if frames else None,
                "last_frame": max(frames) if frames else None,
                "identities": len({box.identity for box in boxes}),
                "class_filter": "person (class_id=1), visibility > 0",
            }
        rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        output = getattr(args, "output", None)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
            print(f"Wrote report: {output}")
        else:
            print(rendered, end="")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"race-mot: error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
