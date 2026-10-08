"""Command line for the initial RACE-MOT device-feasibility milestone."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from race_mot.config import load_config
from race_mot.dashboard import serve_dashboard
from race_mot.detector_smoke import DetectorSmokeConfig, run_detector_smoke
from race_mot.application import SequentialBaseline
from race_mot.detectors.yolox_tensorrt import TensorRTYoloXDetector
from race_mot.evaluation.mot import group_by_frame, read_gt
from race_mot.evaluation.tracking_metrics import evaluate_tracking
from race_mot.evaluation.phase6_dataset import generate_phase6_dataset
from race_mot.inventory import collect_inventory
from race_mot.phone_live import run_reconnect_check
from race_mot.run_manifest import build_manifest, write_manifest
from race_mot.stream_probe import probe_mot_sequence, probe_source
from race_mot.sources.mot_sequence import MotSequenceSource, read_sequence_info
from race_mot.sources.video import VideoSource
from race_mot.trackers.bytetrack import ByteTrackAdapter


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

    reconnect = commands.add_parser(
        "phone-reconnect", help="repeat bounded no-save phone stream open/read/reconnect cycles"
    )
    reconnect_source = reconnect.add_mutually_exclusive_group(required=True)
    reconnect_source.add_argument("--input", help="private phone stream URL")
    reconnect_source.add_argument("--input-env", help="environment variable containing the private stream URL")
    reconnect.add_argument("--cycles", type=int, default=3)
    reconnect.add_argument("--frames-per-cycle", type=int, default=30)
    reconnect.add_argument("--pause-sec", type=float, default=1.0)
    reconnect.add_argument("--output", type=Path, help="JSON output path; defaults to stdout")

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
    smoke.add_argument("--reference-onnx", type=Path)
    smoke.add_argument(
        "--ground-truth-visibility-min",
        type=float,
        default=0.0,
        help="minimum MOT visibility retained for the diagnostic (default: 0)",
    )
    smoke.add_argument("--output", type=Path, help="JSON output path; defaults to stdout")

    validate = commands.add_parser(
        "validate-config", help="validate a local JSON run configuration"
    )
    validate.add_argument("--config", required=True, type=Path)

    inspect = commands.add_parser(
        "inspect-mot", help="inspect a MOTChallenge gt.txt file"
    )
    inspect.add_argument("--gt", required=True, type=Path)

    manifest = commands.add_parser(
        "manifest", help="write a redacted immutable run manifest"
    )
    manifest.add_argument("--config", required=True, type=Path)
    manifest.add_argument("--source", required=True)
    manifest.add_argument("--source-kind", required=True, choices=("rtsp", "video_file", "mot_sequence"))
    manifest.add_argument("--run-id", required=True)
    manifest.add_argument("--output", required=True, type=Path)

    baseline = commands.add_parser(
        "run-baseline", help="run the deterministic detector-every-frame local replay"
    )
    baseline.add_argument("--sequence", required=True, type=Path)
    baseline.add_argument("--engine", required=True, type=Path)
    baseline.add_argument("--config", required=True, type=Path)
    baseline.add_argument("--output", required=True, type=Path)
    baseline.add_argument("--frames", type=int, default=100)
    baseline.add_argument("--confidence", type=float, default=0.10)
    baseline.add_argument("--nms-iou", type=float, default=0.45)
    baseline.add_argument("--run-id")

    phone_run = commands.add_parser(
        "phone-run", help="run a bounded no-save FP32 detector/tracker pass on the live phone stream"
    )
    phone_source = phone_run.add_mutually_exclusive_group(required=True)
    phone_source.add_argument("--input", help="private phone stream URL")
    phone_source.add_argument("--input-env", help="environment variable containing the private stream URL")
    phone_run.add_argument("--engine", required=True, type=Path)
    phone_run.add_argument("--config", required=True, type=Path)
    phone_run.add_argument("--output", required=True, type=Path)
    phone_run.add_argument("--frames", type=int, default=100)
    phone_run.add_argument("--confidence", type=float, default=0.10)
    phone_run.add_argument("--nms-iou", type=float, default=0.45)
    phone_run.add_argument("--run-id")

    replay = commands.add_parser(
        "replay-check", help="repeat a local baseline replay and compare output hashes"
    )
    replay.add_argument("--sequence", required=True, type=Path)
    replay.add_argument("--engine", required=True, type=Path)
    replay.add_argument("--config", required=True, type=Path)
    replay.add_argument("--output-root", required=True, type=Path)
    replay.add_argument("--frames", type=int, default=20)
    replay.add_argument("--repeats", type=int, default=2)
    replay.add_argument("--confidence", type=float, default=0.10)
    replay.add_argument("--nms-iou", type=float, default=0.45)

    evaluate = commands.add_parser(
        "evaluate-baseline", help="evaluate a baseline JSONL log against local MOT ground truth"
    )
    evaluate.add_argument("--sequence", required=True, type=Path)
    evaluate.add_argument("--frame-log", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--visibility-min", type=float, default=0.20)
    evaluate.add_argument("--iou", type=float, default=0.50)

    phase5 = commands.add_parser(
        "phase5-baseline", help="run repeated full local-MOT baseline measurements and metrics"
    )
    phase5.add_argument("--sequence", required=True, type=Path, action="append")
    phase5.add_argument("--engine", required=True, type=Path)
    phase5.add_argument("--config", required=True, type=Path)
    phase5.add_argument("--output-root", required=True, type=Path)
    phase5.add_argument("--repeats", type=int, default=2)
    phase5.add_argument("--frames", type=int, help="bounded prefix per sequence; omit for full sequence")
    phase5.add_argument("--confidence", type=float, default=0.10)
    phase5.add_argument("--nms-iou", type=float, default=0.45)
    phase5.add_argument("--visibility-min", type=float, default=0.20)
    phase5.add_argument("--deadline-ms", type=float, default=None)

    dashboard = commands.add_parser(
        "dashboard", help="serve a local redacted dashboard for a replay run"
    )
    dashboard.add_argument("--run-dir", required=True, type=Path)
    dashboard.add_argument(
        "--host", default="127.0.0.1",
        help="bind address; use a trusted-LAN address only when authorized",
    )
    dashboard.add_argument("--port", type=int, default=8765)

    phase6 = commands.add_parser(
        "phase6-dataset", help="generate and audit the frozen MOT17 paired-rollout dataset"
    )
    phase6.add_argument(
        "--sequence", action="append", nargs=2, metavar=("SEQUENCE_DIR", "BASELINE_RUN_DIR"), required=True,
    )
    phase6.add_argument("--roles", required=True, type=Path)
    phase6.add_argument("--protocol", required=True, type=Path)
    phase6.add_argument("--output-root", required=True, type=Path)
    phase6.add_argument("--prior-history-samples", type=int, default=48)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        output = getattr(args, "output", None)
        if output and output.exists() and args.command not in {"run-baseline"}:
            raise ValueError("report already exists; choose a new output path")
        if args.command == "inventory":
            report = collect_inventory()
        elif args.command == "probe":
            source = os.environ.get(args.input_env) if args.input_env else args.input
            if not source:
                raise ValueError("input environment variable is unset or empty")
            report = probe_source(source, args.duration_sec)
        elif args.command == "phone-reconnect":
            source = os.environ.get(args.input_env) if args.input_env else args.input
            if not source:
                raise ValueError("input environment variable is unset or empty")
            report = run_reconnect_check(
                source,
                cycles=args.cycles,
                frames_per_cycle=args.frames_per_cycle,
                pause_sec=args.pause_sec,
            )
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
                    args.reference_onnx,
                    args.ground_truth_visibility_min,
                ),
            )
        elif args.command == "phone-run":
            source = os.environ.get(args.input_env) if args.input_env else args.input
            if not source:
                raise ValueError("input environment variable is unset or empty")
            if args.frames < 1:
                raise ValueError("frames must be positive")
            source_reader = VideoSource(
                source,
                run_id=args.run_id or "phone-live-run",
                max_frames=args.frames,
            )
            detector = TensorRTYoloXDetector(
                args.engine,
                confidence_threshold=args.confidence,
                nms_iou_threshold=args.nms_iou,
            )
            result = SequentialBaseline(detector, ByteTrackAdapter()).run(
                source_reader,
                output_dir=args.output,
                config_path=args.config,
                source_name=source,
                source_kind=source_reader.source_kind.value,
                run_id=args.run_id,
                max_frames=args.frames,
                runtime_metadata={
                    "phone_capture": True,
                    "raw_video_persistence": False,
                    "measurement_boundary": "phone_live_demo",
                },
            )
            report = result.as_dict()
        elif args.command == "validate-config":
            config = load_config(args.config)
            report = {
                "valid": True,
                "schema_version": config.schema_version,
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
                "measurement_boundary": config.measurement_boundary,
                "annotated_export": config.annotated_export,
                "deadline_ms": config.deadline_ms,
                "warmup_frames": config.warmup_frames,
                "repetitions": config.repetitions,
                "limits_file": str(config.limits_file) if config.limits_file else None,
            }
        elif args.command == "manifest":
            config = load_config(args.config)
            manifest = build_manifest(
                run_id=args.run_id,
                config_path=args.config,
                source=args.source,
                source_kind=args.source_kind,
                model={
                    "family": config.detector_family,
                    "engine_path": config.engine_path.name if config.engine_path else None,
                    "precision": config.precision,
                    "input_size": list(config.input_size),
                    "batch_size": config.batch_size,
                },
                tracker={"family": "bytetrack", "implementation_version": config.tracker_version},
            )
            write_manifest(args.output, manifest)
            print(f"Wrote manifest: {args.output}")
            return 0
        elif args.command == "run-baseline":
            config = load_config(args.config)
            if config.source_kind != "mot_sequence":
                raise ValueError("run-baseline currently accepts a mot_sequence config")
            detector = TensorRTYoloXDetector(
                args.engine,
                checkpoint_path=(args.engine.with_name("yolox_tiny.onnx") if args.engine.with_name("yolox_tiny.onnx").is_file() else None),
                confidence_threshold=args.confidence,
                nms_iou_threshold=args.nms_iou,
            )
            tracker = ByteTrackAdapter()
            runner = SequentialBaseline(detector, tracker)
            result = runner.run(
                MotSequenceSource(args.sequence, run_id=args.run_id or "baseline"),
                output_dir=args.output,
                config_path=args.config,
                source_name=args.sequence.name,
                source_kind="mot_sequence",
                run_id=args.run_id,
                max_frames=args.frames,
            )
            print(json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n", end="")
            return 0 if result.status == "completed" else 2
        elif args.command == "replay-check":
            config = load_config(args.config)
            if args.repeats < 2:
                raise ValueError("repeats must be at least 2")
            if args.output_root.exists() and any(args.output_root.iterdir()):
                raise ValueError("output-root must be empty or absent")
            args.output_root.mkdir(parents=True, exist_ok=True)
            results = []
            for repeat in range(1, args.repeats + 1):
                detector = TensorRTYoloXDetector(
                    args.engine,
                    checkpoint_path=(args.engine.with_name("yolox_tiny.onnx") if args.engine.with_name("yolox_tiny.onnx").is_file() else None),
                    confidence_threshold=args.confidence,
                    nms_iou_threshold=args.nms_iou,
                )
                runner = SequentialBaseline(detector, ByteTrackAdapter())
                result = runner.run(
                    MotSequenceSource(args.sequence, run_id=f"replay-{repeat}"),
                    output_dir=args.output_root / f"repeat-{repeat:02d}",
                    config_path=args.config,
                    source_name=args.sequence.name,
                    source_kind="mot_sequence",
                    run_id=f"replay-{repeat}",
                    max_frames=args.frames,
                )
                results.append(result.as_dict())
            comparable = {
                "frames_processed": {item["frames_processed"] for item in results},
                "detections": {item["detections"] for item in results},
                "tracks": {item["tracks"] for item in results},
                "frame_records_sha256": {item["frame_records_sha256"] for item in results},
                "detection_records_sha256": {item["detection_records_sha256"] for item in results},
                "track_records_sha256": {item["track_records_sha256"] for item in results},
            }
            report = {
                "schema_version": 1,
                "status": "passed" if all(len(values) == 1 for values in comparable.values()) and all(item["status"] == "completed" for item in results) else "failed",
                "repeats": results,
                "equality": {key: len(values) == 1 for key, values in comparable.items()},
                "phone_capture": False,
            }
            rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
            (args.output_root / "replay_check.json").write_text(rendered, encoding="utf-8")
            print(rendered, end="")
            return 0 if report["status"] == "passed" else 2
        elif args.command == "evaluate-baseline":
            report = evaluate_tracking(
                args.sequence,
                args.frame_log,
                visibility_min=args.visibility_min,
                iou_threshold=args.iou,
            )
        elif args.command == "phase5-baseline":
            config = load_config(args.config)
            if args.repeats < 1:
                raise ValueError("repeats must be positive")
            if args.output_root.exists() and any(args.output_root.iterdir()):
                raise ValueError("output-root must be empty or absent")
            args.output_root.mkdir(parents=True, exist_ok=True)
            runs: list[dict[str, object]] = []
            for repeat in range(1, args.repeats + 1):
                for sequence in args.sequence:
                    info = read_sequence_info(sequence)
                    run_dir = args.output_root / f"repeat-{repeat:02d}-{sequence.name}"
                    detector = TensorRTYoloXDetector(
                        args.engine,
                        checkpoint_path=(args.engine.with_name("yolox_tiny.onnx") if args.engine.with_name("yolox_tiny.onnx").is_file() else None),
                        confidence_threshold=args.confidence,
                        nms_iou_threshold=args.nms_iou,
                    )
                    runner = SequentialBaseline(detector, ByteTrackAdapter())
                    result = runner.run(
                        MotSequenceSource(sequence, run_id=f"phase5-{repeat:02d}-{sequence.name}"),
                        output_dir=run_dir,
                        config_path=args.config,
                        source_name=sequence.name,
                        source_kind="mot_sequence",
                        run_id=f"phase5-{repeat:02d}-{sequence.name}",
                        max_frames=args.frames,
                        runtime_metadata={
                            "phase": 5,
                            "repeat": repeat,
                            "sequence_frame_rate": info.frame_rate,
                            "sequence_dimensions": [info.width, info.height],
                            "deadline_ms": args.deadline_ms,
                            "measurement_boundary": config.measurement_boundary,
                            "external_energy_meter": False,
                        },
                    )
                    metric = evaluate_tracking(
                        sequence,
                        run_dir / "frame_log.jsonl",
                        visibility_min=args.visibility_min,
                    ) if result.status == "completed" else {"status": "unavailable", "reason": result.error}
                    metric_path = run_dir / "metrics.json"
                    metric_path.write_text(json.dumps(metric, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    runs.append({"repeat": repeat, "sequence": sequence.name, "summary": result.as_dict(), "metrics": metric})
            completed = [item for item in runs if item["summary"]["status"] == "completed"]
            report = {
                "schema_version": 1,
                "phase": 5,
                "status": "completed" if len(completed) == len(runs) else "failed",
                "sequences": [sequence.name for sequence in args.sequence],
                "repeats": args.repeats,
                "frames_per_sequence": args.frames,
                "detector": {"confidence_threshold": args.confidence, "nms_iou_threshold": args.nms_iou},
                "visibility_min": args.visibility_min,
                "deadline_ms": args.deadline_ms,
                "external_energy_meter": False,
                "energy_status": "unavailable_external_meter_not_connected",
                "runs": runs,
                "phone_capture": False,
            }
            rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
            (args.output_root / "phase5_baseline.json").write_text(rendered, encoding="utf-8")
            print(rendered, end="")
            return 0 if report["status"] == "completed" else 2
        elif args.command == "dashboard":
            if not args.run_dir.is_dir():
                raise ValueError(f"run directory does not exist: {args.run_dir}")
            if not 1 <= args.port <= 65535:
                raise ValueError("port must be in [1, 65535]")
            serve_dashboard(args.run_dir, args.host, args.port)
            return 0
        elif args.command == "phase6-dataset":
            if args.prior_history_samples < 1:
                raise ValueError("prior-history-samples must be positive")
            manifest = generate_phase6_dataset(
                sequences=[(Path(sequence), Path(run_dir)) for sequence, run_dir in args.sequence],
                roles_path=args.roles,
                protocol_path=args.protocol,
                output_root=args.output_root,
                prior_history_samples_per_sequence=args.prior_history_samples,
            )
            print(json.dumps(manifest, indent=2, sort_keys=True) + "\n", end="")
            return 0
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
