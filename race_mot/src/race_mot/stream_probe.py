"""Read-only camera/video input probe; never writes frames to disk."""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def redact_source(source: str) -> str:
    """Remove URL userinfo and query values from a source identifier."""
    if not source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return Path(source).name
    try:
        parsed = urlsplit(source)
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        safe_path = re.sub(r"[^/]+", "<redacted>", parsed.path) if parsed.path else ""
        return urlunsplit((parsed.scheme, host, safe_path, "<redacted>" if parsed.query else "", ""))
    except ValueError:
        return "<invalid-url>"


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _open_capture(cv2: object, source: str) -> tuple[object, str]:
    """Prefer FFmpeg's open/read timeouts for RTSP, then use OpenCV's default backend."""
    is_rtsp = source.lower().startswith(("rtsp://", "rtsps://"))
    timeout_prop = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
    read_timeout_prop = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
    ffmpeg = getattr(cv2, "CAP_FFMPEG", None)
    if is_rtsp and timeout_prop is not None and read_timeout_prop is not None and ffmpeg is not None:
        try:
            capture = cv2.VideoCapture(
                source,
                ffmpeg,
                [timeout_prop, 5000, read_timeout_prop, 5000],
            )
            if capture.isOpened():
                return capture, "FFMPEG"
            capture.release()
        except (TypeError, AttributeError, cv2.error):
            pass

    capture = cv2.VideoCapture(source)
    try:
        backend = capture.getBackendName() if capture.isOpened() else "unknown"
    except (AttributeError, cv2.error):
        backend = "default"
    return capture, str(backend)


def probe_source(source: str, duration_sec: float) -> dict[str, object]:
    """Measure frame-read behavior for a local file or camera stream."""
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV Python (cv2) is unavailable. Use the JetPack-compatible system OpenCV."
        ) from exc

    capture, backend = _open_capture(cv2, source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("Could not open input. Check the URL/path, network, and decoder support.")

    # The backend may ignore buffer size; record the actual backend in the report.
    if source.lower().startswith(("rtsp://", "rtsps://")):
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    started = time.perf_counter()
    deadline = started + duration_sec
    read_times: list[float] = []
    read_intervals: list[float] = []
    last_read: float | None = None
    read_errors = 0
    frames_read = 0

    try:
        while time.perf_counter() < deadline:
            before = time.perf_counter()
            ok, frame = capture.read()
            after = time.perf_counter()
            read_times.append(after - before)
            if not ok or frame is None:
                read_errors += 1
                break
            frames_read += 1
            height, width = frame.shape[:2]
            if last_read is not None:
                read_intervals.append(after - last_read)
            last_read = after
    finally:
        capture.release()

    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "source": redact_source(source),
        "opencv_backend": backend,
        "duration_requested_sec": duration_sec,
        "duration_observed_sec": elapsed,
        "resolution": {"width": width, "height": height},
        "reported_source_fps": reported_fps if reported_fps > 0 else None,
        "frames_read": frames_read,
        "measured_read_fps": frames_read / elapsed,
        "read_errors": read_errors,
        "stopped_on_read_failure": read_errors > 0,
        "read_call_ms": {
            "median": statistics.median(read_times) * 1000 if read_times else None,
            "p95": _percentile(read_times, 0.95) * 1000 if read_times else None,
        },
        "inter_frame_read_interval_ms": {
            "median": statistics.median(read_intervals) * 1000 if read_intervals else None,
            "p95": _percentile(read_intervals, 0.95) * 1000 if read_intervals else None,
        },
        "limitations": [
            "RTSP timeouts are requested only when OpenCV exposes FFmpeg timeout parameters; backend behavior must be verified on the target JetPack image.",
            "The default OpenCV backend may not enforce a hard read timeout.",
            "Read rate is input/decode behavior only; it is not detector, tracker, or end-to-end pipeline FPS.",
            "RTSP timestamps and camera cadence must be verified against the phone app configuration.",
            "No frames are written to disk by this probe.",
        ],
    }


def write_probe_report(source: str, duration_sec: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = probe_source(source, duration_sec)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
