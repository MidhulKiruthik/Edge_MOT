"""Bounded, no-save checks for an authorized phone stream.

The checks deliberately open and release the stream for each cycle.  This
exercises client interruption/reconnect behavior without pretending that a
client restart is the same as a physical network outage.  Frames remain in
memory only and are never written by this module.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Any

from race_mot.stream_probe import _open_capture, redact_source


def run_reconnect_check(
    source: str,
    *,
    cycles: int = 3,
    frames_per_cycle: int = 30,
    pause_sec: float = 1.0,
) -> dict[str, Any]:
    """Read bounded frames, release the client, then reconnect repeatedly."""
    if cycles < 2:
        raise ValueError("cycles must be at least 2")
    if frames_per_cycle < 1:
        raise ValueError("frames_per_cycle must be positive")
    if not math.isfinite(pause_sec) or pause_sec < 0:
        raise ValueError("pause_sec must be finite and non-negative")
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - target image dependency
        raise RuntimeError("OpenCV Python (cv2) is required") from exc

    cycle_records: list[dict[str, Any]] = []
    all_read_ms: list[float] = []
    for cycle in range(1, cycles + 1):
        started = time.perf_counter()
        capture, backend = _open_capture(cv2, source)
        opened = bool(capture.isOpened())
        frames = 0
        errors = 0
        width = height = None
        positions: list[float] = []
        reads_ms: list[float] = []
        try:
            if opened:
                # HTTP MJPEG backends may ignore this property; recording the
                # backend keeps the result auditable.
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                while frames < frames_per_cycle:
                    before = time.perf_counter()
                    ok, frame = capture.read()
                    after = time.perf_counter()
                    reads_ms.append((after - before) * 1000.0)
                    if not ok or frame is None:
                        errors += 1
                        break
                    frames += 1
                    height, width = frame.shape[:2]
                    position = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
                    if math.isfinite(position) and position >= 0:
                        positions.append(position)
        finally:
            capture.release()
        observed = max(time.perf_counter() - started, 1e-9)
        all_read_ms.extend(reads_ms)
        cycle_records.append(
            {
                "cycle": cycle,
                "backend": backend,
                "opened": opened,
                "frames_read": frames,
                "read_errors": errors,
                "resolution": {"width": width, "height": height},
                "position_first": positions[0] if positions else None,
                "position_last": positions[-1] if positions else None,
                "observed_sec": observed,
                "read_ms_median": statistics.median(reads_ms) if reads_ms else None,
            }
        )
        if cycle < cycles and pause_sec:
            time.sleep(pause_sec)

    passed = all(
        item["opened"]
        and item["frames_read"] == frames_per_cycle
        and item["read_errors"] == 0
        for item in cycle_records
    )
    return {
        "schema_version": 1,
        "source": redact_source(source),
        "cycles_requested": cycles,
        "frames_per_cycle": frames_per_cycle,
        "pause_sec": pause_sec,
        "cycles": cycle_records,
        "reconnect_passed": passed,
        "read_ms_median": statistics.median(all_read_ms) if all_read_ms else None,
        "raw_video_persistence": False,
        "frames_saved": False,
        "network_fault_injection": False,
        "limitations": [
            "Each cycle intentionally closes and reopens the client; this proves controlled client reconnect, not a physical cable or radio fault.",
            "The stream URL is redacted and no decoded frame is persisted.",
            "Trusted-LAN acceptance is bounded to the observed private route and endpoint; it is not a security audit of the phone application.",
        ],
    }
