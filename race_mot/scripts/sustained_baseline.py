"""Run the complete local detector/tracker baseline with Jetson telemetry."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from race_mot.application import SequentialBaseline
from race_mot.detectors.yolox_tensorrt import TensorRTYoloXDetector
from race_mot.inventory import collect_inventory
from race_mot.sources.mot_sequence import MotSequenceSource
from race_mot.trackers.bytetrack import ByteTrackAdapter


def _cycling_source(sequence: Path, run_id: str, duration_sec: float):
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        for packet in MotSequenceSource(sequence, run_id=run_id):
            if time.monotonic() >= deadline:
                return
            yield packet


def _telemetry_summary(path: Path) -> dict[str, object]:
    temperatures: list[float] = []
    ram_used: list[int] = []
    ram_total: list[int] = []
    gpu: list[int] = []
    vdd_in: list[int] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.search(r"RAM (\d+)/(\d+)MB", line)
            if match:
                ram_used.append(int(match.group(1))); ram_total.append(int(match.group(2)))
            match = re.search(r"tj@([0-9.]+)C/([0-9.]+)C", line)
            if match:
                temperatures.extend([float(match.group(1)), float(match.group(2))])
            match = re.search(r"GR3D_FREQ (\d+)%", line)
            if match:
                gpu.append(int(match.group(1)))
            match = re.search(r"VDD_IN (\d+)mW/(\d+)mW/(\d+)mW", line)
            if match:
                vdd_in.extend(int(value) for value in match.groups())
    return {
        "samples": len(ram_used),
        "ram_used_mb": {"min": min(ram_used) if ram_used else None, "max": max(ram_used) if ram_used else None, "total": max(ram_total) if ram_total else None},
        "tj_c": {"min": min(temperatures) if temperatures else None, "max": max(temperatures) if temperatures else None},
        "gr3d_percent": {"min": min(gpu) if gpu else None, "max": max(gpu) if gpu else None},
        "vdd_in_mw_diagnostic": {"min": min(vdd_in) if vdd_in else None, "max": max(vdd_in) if vdd_in else None},
        "throttling_flags": "not exposed by this tegrastats format",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--telemetry-output", type=Path, required=True)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    args = parser.parse_args()
    if args.duration_sec <= 0:
        raise SystemExit("duration-sec must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.telemetry_output.parent.mkdir(parents=True, exist_ok=True)
    telemetry = subprocess.Popen(
        ["tegrastats", "--interval", "1000", "--logfile", str(args.telemetry_output)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    started = time.monotonic()
    try:
        detector = TensorRTYoloXDetector(
            args.engine,
            checkpoint_path=(args.engine.with_name("yolox_tiny.onnx") if args.engine.with_name("yolox_tiny.onnx").is_file() else None),
            confidence_threshold=args.confidence,
            nms_iou_threshold=args.nms_iou,
        )
        runner = SequentialBaseline(detector, ByteTrackAdapter())
        result = runner.run(
            _cycling_source(args.sequence, "phase5-sustained", args.duration_sec),
            output_dir=args.output.parent / (args.output.stem + "_runtime"),
            config_path=args.config,
            source_name=args.sequence.name,
            source_kind="mot_sequence",
            run_id="phase5-sustained",
            runtime_metadata={"phase": 5, "sustained_duration_requested_sec": args.duration_sec, "external_energy_meter": False},
        )
    finally:
        telemetry.terminate()
        try:
            telemetry.wait(timeout=5)
        except subprocess.TimeoutExpired:
            telemetry.kill(); telemetry.wait()
    summary = {
        "schema_version": 1,
        "phase": 5,
        "run_kind": "sustained_complete_detector_tracker_baseline",
        "duration_requested_sec": args.duration_sec,
        "duration_observed_sec": time.monotonic() - started,
        "sequence": args.sequence.name,
        "summary": result.as_dict(),
        "inventory": collect_inventory(),
        "telemetry_file": args.telemetry_output.name,
        "telemetry": _telemetry_summary(args.telemetry_output),
        "external_energy_meter": False,
        "energy_status": "unavailable_external_meter_not_connected",
        "frames_saved": False,
        "phone_capture": False,
        "limitations": [
            "Onboard tegrastats is a diagnostic series, not whole-device energy measurement.",
            "The current tegrastats format does not expose throttling flags; raw telemetry is retained.",
        ],
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True); handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
