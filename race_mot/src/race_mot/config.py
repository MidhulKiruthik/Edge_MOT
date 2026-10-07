"""Dependency-free configuration loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class RunConfig:
    mode: str
    source_kind: str
    detector_family: str
    input_size: tuple[int, int]
    precision: str
    batch_size: int
    threshold_tau: float | None
    max_consecutive_skips: int | None
    output_root: Path


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

    run = _section(data, "run")
    input_config = _section(data, "input")
    detector = _section(data, "detector")
    tracker = _section(data, "tracker")
    policy = _section(data, "policy")

    mode = run.get("mode")
    source_kind = input_config.get("kind")
    detector_family = detector.get("family")
    precision = detector.get("precision")
    batch_size = detector.get("batch_size")
    input_size = detector.get("input_size")
    if mode not in {"baseline", "adaptive"}:
        raise ValueError("run.mode must be 'baseline' or 'adaptive'")
    if source_kind not in {"rtsp", "video_file", "mot_sequence"}:
        raise ValueError("input.kind is not supported")
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

    output_root = run.get("output_root", "runs")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("run.output_root must be a non-empty string")
    return RunConfig(
        mode=mode,
        source_kind=source_kind,
        detector_family=detector_family,
        input_size=(input_size[0], input_size[1]),
        precision=precision,
        batch_size=batch_size,
        threshold_tau=float(threshold_tau) if threshold_tau is not None else None,
        max_consecutive_skips=max_skips,
        output_root=Path(output_root),
    )
