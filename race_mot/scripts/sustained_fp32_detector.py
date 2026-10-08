"""Run repeated local MOT17 FP32 detector smoke batches for thermal evidence."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

from race_mot.detector_smoke import DetectorSmokeConfig, run_detector_smoke


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--duration-sec", type=float, default=1800.0)
    parser.add_argument("--batch-frames", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--telemetry-output", type=Path, required=True)
    args = parser.parse_args()
    if args.duration_sec <= 0 or args.batch_frames < 1:
        raise SystemExit("duration-sec and batch-frames must be positive")

    args.telemetry_output.parent.mkdir(parents=True, exist_ok=True)
    telemetry = subprocess.Popen(
        ["tegrastats", "--interval", "1000", "--logfile", str(args.telemetry_output)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    started = time.monotonic()
    batches = 0
    frames = 0
    batch_fps: list[float] = []
    finite_frames = 0
    person_counts: list[int] = []
    try:
        while time.monotonic() - started < args.duration_sec:
            batch_started = time.monotonic()
            report = run_detector_smoke(
                args.sequence,
                DetectorSmokeConfig(args.engine, frames=args.batch_frames),
            )
            elapsed = max(time.monotonic() - batch_started, 1e-9)
            batches += 1
            processed = int(report["frames_processed"])
            frames += processed
            finite_frames += int(report["finite_output_frames"])
            person_counts.extend(report["person_postprocessing"]["counts_by_frame"])
            batch_fps.append(processed / elapsed)
    finally:
        telemetry.terminate()
        try:
            telemetry.wait(timeout=5)
        except subprocess.TimeoutExpired:
            telemetry.kill()
            telemetry.wait()

    summary = {
        "schema_version": 1,
        "run_kind": "sustained_fp32_detector_thermal_diagnostic",
        "duration_requested_sec": args.duration_sec,
        "duration_observed_sec": time.monotonic() - started,
        "sequence": args.sequence.name,
        "engine": args.engine.name,
        "telemetry_file": args.telemetry_output.name,
        "batch_frames": args.batch_frames,
        "batches": batches,
        "frames_processed": frames,
        "finite_output_frames": finite_frames,
        "finite_output_fraction": finite_frames / frames if frames else None,
        "batch_fps_median": statistics.median(batch_fps) if batch_fps else None,
        "person_count_min": min(person_counts) if person_counts else None,
        "person_count_median": statistics.median(person_counts) if person_counts else None,
        "person_count_max": max(person_counts) if person_counts else None,
        "detector_every_frame": True,
        "tracker_included": False,
        "external_energy_meter": False,
        "frames_saved": False,
        "limitations": [
            "This is a sustained FP32 detector and thermal diagnostic, not the complete detector-plus-tracker baseline.",
            "The same MOT prefix is replayed for each batch; it is not a tracking-quality evaluation.",
            "Whole-pipeline energy is unavailable without an external meter.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
