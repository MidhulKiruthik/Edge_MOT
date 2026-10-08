"""Dependency-free configuration loading and validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class RunConfig:
    schema_version: int
    mode: str
    source_kind: str
    detector_family: str
    input_size: tuple[int, int]
    precision: str
    batch_size: int
    threshold_tau: float | None
    max_consecutive_skips: int | None
    output_root: Path
    source: str | None
    engine_path: Path | None
    tracker_version: str
    expected_width: int | None
    expected_height: int | None
    expected_fps: float | None
    checkpoint_sha256: str | None
    engine_sha256: str | None
    measurement_boundary: str
    annotated_export: bool
    deadline_ms: float | None
    warmup_frames: int
    repetitions: int
    limits_file: Path | None


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = data.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"missing configuration section: {name}")
    return section


def load_config(path: Path) -> RunConfig:
    """Load the JSON representation of the documented configuration contract."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON configuration: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("configuration root must be an object")
    schema_version = data.get("schema_version")
    if schema_version != 1:
        raise ValueError("schema_version must be 1")

    run = _section(data, "run")
    input_config = _section(data, "input")
    detector = _section(data, "detector")
    tracker = _section(data, "tracker")
    policy = _section(data, "policy")

    mode = run.get("mode")
    source_kind = input_config.get("kind")
    source = input_config.get("source")
    detector_family = detector.get("family")
    precision = detector.get("precision")
    batch_size = detector.get("batch_size")
    engine_path = detector.get("engine_path")
    checkpoint_sha256 = detector.get("checkpoint_sha256")
    engine_sha256 = detector.get("engine_sha256")
    input_size = detector.get("input_size")
    if mode not in {"baseline", "adaptive"}:
        raise ValueError("run.mode must be 'baseline' or 'adaptive'")
    if source_kind not in {"rtsp", "video_file", "mot_sequence"}:
        raise ValueError("input.kind is not supported")
    if source is not None and (not isinstance(source, str) or not source.strip()):
        raise ValueError("input.source must be a non-empty string when provided")
    if not isinstance(detector_family, str) or not detector_family:
        raise ValueError("detector.family must be a non-empty string")
    if not isinstance(precision, str) or not precision:
        raise ValueError("detector.precision must be a non-empty string")
    if batch_size != 1:
        raise ValueError("detector.batch_size must be 1 for the MVP")
    if (
        not isinstance(input_size, list)
        or len(input_size) != 2
        or any(not isinstance(value, int) or value <= 0 for value in input_size)
    ):
        raise ValueError(
            "detector.input_size must be [positive_width, positive_height]"
        )
    if tracker.get("family") != "bytetrack":
        raise ValueError("tracker.family must be 'bytetrack' for the MVP")
    tracker_version = tracker.get("implementation_version", "TBD")
    if not isinstance(tracker_version, str) or not tracker_version.strip():
        raise ValueError("tracker.implementation_version must be a non-empty string")
    if engine_path is not None and (not isinstance(engine_path, str) or not engine_path.strip()):
        raise ValueError("detector.engine_path must be a non-empty string when provided")
    for name, value in (("checkpoint_sha256", checkpoint_sha256), ("engine_sha256", engine_sha256)):
        if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value)):
            raise ValueError(f"detector.{name} must be a SHA-256 hex string when provided")

    expected_width = input_config.get("expected_width")
    expected_height = input_config.get("expected_height")
    expected_fps = input_config.get("expected_fps")
    if expected_width is not None and (not isinstance(expected_width, int) or expected_width < 1):
        raise ValueError("input.expected_width must be positive when provided")
    if expected_height is not None and (not isinstance(expected_height, int) or expected_height < 1):
        raise ValueError("input.expected_height must be positive when provided")
    if expected_fps is not None and (not isinstance(expected_fps, (int, float)) or expected_fps <= 0):
        raise ValueError("input.expected_fps must be positive when provided")

    threshold_tau = policy.get("threshold_tau")
    max_skips = policy.get("max_consecutive_skips")
    if mode == "adaptive":
        if (
            not isinstance(threshold_tau, (int, float))
            or not 0.0 <= threshold_tau <= 1.0
        ):
            raise ValueError("adaptive policy.threshold_tau must be in [0, 1]")
        if not isinstance(max_skips, int) or max_skips < 1:
            raise ValueError("adaptive policy.max_consecutive_skips must be positive")
    else:
        threshold_tau = None
        max_skips = None

    measurement = _section(data, "measurement")
    measurement_boundary = measurement.get("boundary")
    if not isinstance(measurement_boundary, str) or not measurement_boundary.strip():
        raise ValueError("measurement.boundary must be a non-empty string")
    deadline_ms = measurement.get("deadline_ms")
    if deadline_ms is not None and (not isinstance(deadline_ms, (int, float)) or deadline_ms <= 0):
        raise ValueError("measurement.deadline_ms must be positive when provided")
    warmup_frames = measurement.get("warmup_frames", 5)
    repetitions = measurement.get("repetitions", 2)
    limits_file = measurement.get("limits_file")
    if not isinstance(warmup_frames, int) or warmup_frames < 0:
        raise ValueError("measurement.warmup_frames must be non-negative")
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("measurement.repetitions must be positive")
    if limits_file is not None and (not isinstance(limits_file, str) or not limits_file.strip()):
        raise ValueError("measurement.limits_file must be a non-empty string when provided")
    privacy = _section(data, "privacy")
    annotated_export = privacy.get("annotated_export", False)
    if not isinstance(annotated_export, bool):
        raise ValueError("privacy.annotated_export must be boolean")

    output_root = run.get("output_root", "runs")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("run.output_root must be a non-empty string")
    return RunConfig(
        schema_version=schema_version,
        mode=mode,
        source_kind=source_kind,
        detector_family=detector_family,
        input_size=(input_size[0], input_size[1]),
        precision=precision,
        batch_size=batch_size,
        threshold_tau=float(threshold_tau) if threshold_tau is not None else None,
        max_consecutive_skips=max_skips,
        output_root=Path(output_root),
        source=source,
        engine_path=Path(engine_path) if engine_path is not None else None,
        tracker_version=tracker_version,
        expected_width=expected_width,
        expected_height=expected_height,
        expected_fps=float(expected_fps) if expected_fps is not None else None,
        checkpoint_sha256=checkpoint_sha256,
        engine_sha256=engine_sha256,
        measurement_boundary=measurement_boundary,
        annotated_export=annotated_export,
        deadline_ms=float(deadline_ms) if deadline_ms is not None else None,
        warmup_frames=warmup_frames,
        repetitions=repetitions,
        limits_file=Path(limits_file) if limits_file is not None else None,
    )
