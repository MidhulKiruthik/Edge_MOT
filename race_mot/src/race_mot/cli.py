"""Command line for the initial RACE-MOT device-feasibility milestone."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from race_mot.config import load_config
from race_mot.detector_smoke import DetectorSmokeConfig, run_detector_smoke
from race_mot.evaluation.mot import group_by_frame, read_gt
from race_mot.inventory import collect_inventory
from race_mot.stream_probe import probe_mot_sequence, probe_source


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
    source = probe.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="local video path or public input URL")
    source.add_argument("--input-env", help="environment variable containing the private input URL")
    probe.add_argument("--duration-sec", type=float, default=30.0)
    probe.add_argument(
        "--output", type=Path, help="JSON output path; defaults to stdout"
    )

    mot_probe = commands.add_parser(
        "mot-probe", help="decode a bounded MOT image-sequence prefix without saving frames"
    )
    mot_probe.add_argument("--sequence", required=True, type=Path)
    mot_probe.add_argument("--frames", type=int, default=100)
    mot_probe.add_argument("--output", type=Path, help="JSON output path; defaults to stdout")

    smoke = commands.add_parser(
        "detector-smoke", help="run a bounded no-save TensorRT detector feasibility check"
    )
    smoke_source = smoke.add_mutually_exclusive_group(required=True)
    smoke_source.add_argument("--input", help="local video path or public input URL")
    smoke_source.add_argument("--input-env", help="environment variable containing the private input URL")
    smoke_source.add_argument("--mot-sequence", type=Path, help="local MOT image-sequence directory")
    smoke.add_argument("--engine", required=True, type=Path)
    smoke.add_argument("--frames", type=int, default=10)
    smoke.add_argument("--confidence", type=float, default=0.3)
    smoke.add_argument("--nms-iou", type=float, default=0.45)
    smoke.add_argument("--output", type=Path, help="JSON output path; defaults to stdout")

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
        output = getattr(args, "output", None)
        if output and output.exists():
            raise ValueError("report already exists; choose a new output path")
        if args.command == "inventory":
            report = collect_inventory()
        elif args.command == "probe":
            source = os.environ.get(args.input_env) if args.input_env else args.input
            if not source:
                raise ValueError("input environment variable is unset or empty")
            report = probe_source(source, args.duration_sec)
        elif args.command == "mot-probe":
            report = probe_mot_sequence(args.sequence, args.frames)
        elif args.command == "detector-smoke":
            source = args.mot_sequence if args.mot_sequence else (os.environ.get(args.input_env) if args.input_env else args.input)
            if not source:
                raise ValueError("input environment variable is unset or empty")
            report = run_detector_smoke(
                source,
                DetectorSmokeConfig(
                    args.engine,
                    args.frames,
                    args.confidence,
                    args.nms_iou,
                ),
            )
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
            with output.open("x", encoding="utf-8") as handle:
                handle.write(rendered)
            print(f"Wrote report: {output}")
        else:
            print(rendered, end="")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"race-mot: error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
