"""Read-only camera/video input probe; never writes frames to disk."""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from configparser import ConfigParser
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
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError("duration_sec must be finite and positive")

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

    try:
        # The backend may ignore buffer size; record the actual backend in the report.
        if source.lower().startswith(("rtsp://", "rtsps://")):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        is_network = source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))
        reported_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        expected_count = int(reported_count) if math.isfinite(reported_count) and reported_count > 0 else None
        observations = {
            name: {"samples": 0, "first": None, "last": None, "repeats": 0, "regressions": 0}
            for name in ("frame_position", "timestamp_ms")
        }
        stop_reason = "duration_limit"
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

        while time.perf_counter() < deadline:
            before = time.perf_counter()
            ok, frame = capture.read()
            after = time.perf_counter()
            read_times.append(after - before)
            if not ok or frame is None:
                if not is_network and expected_count is not None and frames_read >= expected_count:
                    stop_reason = "expected_file_end"
                else:
                    stop_reason = "read_failure"
                    read_errors += 1
                break
            frames_read += 1
            for name, prop in (("frame_position", cv2.CAP_PROP_POS_FRAMES),
                               ("timestamp_ms", cv2.CAP_PROP_POS_MSEC)):
                value = float(capture.get(prop))
                if not math.isfinite(value) or value < 0:
                    continue
                item = observations[name]
                if item["last"] is not None:
                    item["repeats"] += int(value == item["last"])
                    item["regressions"] += int(value < item["last"])
                else:
                    item["first"] = value
                item["last"] = value
                item["samples"] += 1
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
        "reported_frame_count": expected_count,
        "stop_reason": stop_reason,
        "decoder_observations": observations,
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
            "Decoder positions/timestamps are observations, not proof of source ordering or camera cadence; repeated zero values may mean unsupported metadata.",
            "Expected file end relies on reported frame count; unknown-length or early read failure cannot distinguish EOF from corruption.",
            "No reconnect is attempted; input drops cannot be inferred from decoder read counts.",
            "No frames are written to disk by this probe.",
        ],
    }


def probe_mot_sequence(sequence_dir: Path, max_frames: int) -> dict[str, object]:
    """Decode a bounded prefix of a MOT image sequence without retaining images."""
    if max_frames < 1:
        raise ValueError("frames must be at least 1")
    info_path = sequence_dir / "seqinfo.ini"
    if not info_path.is_file():
        raise ValueError("MOT sequence is missing seqinfo.ini")
    parser = ConfigParser()
    parser.read(info_path, encoding="utf-8")
    if not parser.has_section("Sequence"):
        raise ValueError("seqinfo.ini is missing the [Sequence] section")
    sequence = parser["Sequence"]
    try:
        sequence_length = int(sequence["seqLength"])
        width = int(sequence["imWidth"])
        height = int(sequence["imHeight"])
        frame_rate = float(sequence["frameRate"])
    except (KeyError, ValueError) as exc:
        raise ValueError("seqinfo.ini has invalid sequence metadata") from exc
    image_dir = sequence_dir / sequence.get("imDir", "img1")
    extension = sequence.get("imExt", ".jpg")
    if not image_dir.is_dir():
        raise ValueError("MOT sequence is missing its image directory")
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("OpenCV Python (cv2) is unavailable.") from exc

    requested_frames = min(max_frames, sequence_length)
    decode_times: list[float] = []
    decoded_frames = 0
    missing_frames: list[int] = []
    dimension_mismatches: list[int] = []
    started = time.perf_counter()
    for index in range(1, requested_frames + 1):
        path = image_dir / f"{index:06d}{extension}"
        before = time.perf_counter()
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        decode_times.append(time.perf_counter() - before)
        if frame is None:
            missing_frames.append(index)
            break
        decoded_frames += 1
        if frame.shape[:2] != (height, width):
            dimension_mismatches.append(index)
    elapsed = max(time.perf_counter() - started, 1e-9)
    stop_reason = (
        "read_failure" if missing_frames else
        "expected_sequence_end" if requested_frames == sequence_length else
        "frame_limit"
    )
    return {
        "source": sequence_dir.name,
        "source_kind": "mot_sequence",
        "sequence_length": sequence_length,
        "reported_source_fps": frame_rate if frame_rate > 0 else None,
        "expected_resolution": {"width": width, "height": height},
        "frames_requested": requested_frames,
        "frames_decoded": decoded_frames,
        "stop_reason": stop_reason,
        "missing_frames": missing_frames,
        "dimension_mismatches": dimension_mismatches,
        "duration_observed_sec": elapsed,
        "measured_decode_fps": decoded_frames / elapsed,
        "decode_call_ms": {
            "median": statistics.median(decode_times) * 1000 if decode_times else None,
            "p95": _percentile(decode_times, 0.95) * 1000 if decode_times else None,
        },
        "limitations": [
            "Image decode rate is not detector, tracker, or end-to-end pipeline FPS.",
            "This check only decodes a bounded prefix and does not validate every image in the sequence.",
            "No images are written, retained, or transmitted.",
        ],
    }


def write_probe_report(source: str, duration_sec: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = probe_source(source, duration_sec)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
