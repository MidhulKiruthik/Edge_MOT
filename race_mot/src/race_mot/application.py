"""Sequential deterministic detector-every-frame runtime."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from race_mot.domain import Detection, FramePacket, TrackSnapshot
from race_mot.run_manifest import build_manifest, write_manifest


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _detection_record(detection: Detection) -> dict[str, object]:
    return {"xyxy": list(detection.xyxy), "score": detection.score, "class_id": detection.class_id, "class_name": detection.class_name}


def _track_record(track: TrackSnapshot) -> dict[str, object]:
    return {"temporary_track_id": track.temporary_track_id, "xyxy": list(track.xyxy), "score": track.score, "state": track.state.value, "age_frames": track.age_frames}


@dataclass(frozen=True, slots=True)
class BaselineResult:
    run_id: str
    status: str
    output_dir: Path
    frames_processed: int
    detections: int
    tracks: int
    frame_records_sha256: str
    detection_records_sha256: str
    track_records_sha256: str
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": self.status,
            "output_dir": str(self.output_dir),
            "frames_processed": self.frames_processed,
            "detections": self.detections,
            "tracks": self.tracks,
            "frame_records_sha256": self.frame_records_sha256,
            "detection_records_sha256": self.detection_records_sha256,
            "track_records_sha256": self.track_records_sha256,
            "error": self.error,
        }


class SequentialBaseline:
    """Single-worker, bounded detector-every-frame lifecycle.

    ``stop()`` is idempotent.  Calling ``run()`` after a completed run resets the
    detector/tracker lifecycle and creates a new run directory unless the caller
    explicitly reuses the run id (which is rejected by immutable output files).
    """

    def __init__(self, detector: object, tracker: object, *, max_queue_size: int = 1) -> None:
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be positive")
        self.detector = detector
        self.tracker = tracker
        self.max_queue_size = max_queue_size
        self._started = False
        self._stopped = False

    def stop(self) -> None:
        if self._stopped:
            return
        close = getattr(self.detector, "close", None)
        if callable(close):
            close()
        self._stopped = True
        self._started = False

    def run(
        self,
        source: Iterable[FramePacket],
        *,
        output_dir: Path,
        config_path: Path,
        source_name: str,
        source_kind: str,
        run_id: str | None = None,
        max_frames: int | None = None,
        device: Mapping[str, object] | None = None,
        runtime_metadata: Mapping[str, object] | None = None,
    ) -> BaselineResult:
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be positive")
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"output directory is not empty: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = run_id or f"baseline-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self._started, self._stopped = True, False
        reset = getattr(self.tracker, "reset", None)
        if callable(reset):
            reset()
        manifest = build_manifest(
            run_id=run_id,
            config_path=config_path,
            source=source_name,
            source_kind=source_kind,
            device=device,
            model=getattr(self.detector, "metadata", {}),
            tracker=getattr(self.tracker, "metadata", {}),
        )
        phone_capture = bool((runtime_metadata or {}).get("phone_capture", False))
        manifest["runtime"] = {
            "mode": "baseline",
            "action": "DETECT",
            "max_queue_size": self.max_queue_size,
            "phone_capture": phone_capture,
            **dict(runtime_metadata or {}),
        }
        write_manifest(output_dir / "manifest.json", manifest)
        frame_path = output_dir / "frame_log.jsonl"
        deadline_ms = (runtime_metadata or {}).get("deadline_ms")
        frames_processed = detections_total = tracks_total = 0
        frame_hashes: list[object] = []
        detection_hashes: list[object] = []
        track_hashes: list[object] = []
        error: str | None = None
        status = "completed"
        try:
            with frame_path.open("x", encoding="utf-8") as log:
                for packet in source:
                    if max_frames is not None and frames_processed >= max_frames:
                        break
                    if packet.image_bgr is None:
                        raise RuntimeError("source emitted a frame without image data")
                    processing_started_ns = time.monotonic_ns()
                    detections = list(self.detector.detect(packet))
                    tracks = list(self.tracker.update(detections, packet))
                    pipeline_completion_ns = time.monotonic_ns()
                    timing = getattr(self.detector, "last_timing", None)
                    timing_record = timing.as_dict() if timing is not None and hasattr(timing, "as_dict") else None
                    record = {
                        "schema_version": 1,
                        "run_id": run_id,
                        "source_index": packet.source_index,
                        "source_timestamp_ms": packet.source_timestamp_ms,
                        "arrival_monotonic_ns": packet.arrival_monotonic_ns,
                        "pipeline_completion_monotonic_ns": pipeline_completion_ns,
                        "pipeline_latency_ms": max(
                            0.0,
                            (pipeline_completion_ns - packet.arrival_monotonic_ns) / 1_000_000,
                        ),
                        "processing_ms": (pipeline_completion_ns - processing_started_ns) / 1_000_000,
                        "deadline_ms": deadline_ms,
                        "source_kind": packet.source_kind.value,
                        "input_drop_count": packet.input_drop_count,
                        "planned_action": "DETECT",
                        "executed_action": "DETECT",
                        "reason": "BASELINE_EVERY_FRAME",
                        "detections": [_detection_record(item) for item in detections],
                        "tracks": [_track_record(item) for item in tracks],
                        "active_track_count": len(tracks),
                        "detector_timing": timing_record,
                        "tracker_event": getattr(self.tracker, "last_event", "update"),
                        "queue_depth": 0,
                    }
                    log.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                    log.flush()
                    frames_processed += 1
                    detections_total += len(detections)
                    tracks_total += len(tracks)
                    # Timing and local arrival clocks are diagnostics and can
                    # vary between repeats.  The deterministic frame hash is
                    # therefore computed from causal source/action/output
                    # fields while the full JSONL record retains timings.
                    stable_record = {
                        key: record[key]
                        for key in (
                            "source_index", "source_timestamp_ms", "source_kind",
                            "input_drop_count", "planned_action", "executed_action",
                            "reason", "detections", "tracks", "active_track_count",
                        )
                    }
                    frame_hashes.append(stable_record)
                    detection_hashes.append({"source_index": packet.source_index, "detections": record["detections"]})
                    track_hashes.append({"source_index": packet.source_index, "tracks": record["tracks"]})
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            self.stop()
        result = BaselineResult(
            run_id, status, output_dir, frames_processed, detections_total, tracks_total,
            _canonical_hash(frame_hashes), _canonical_hash(detection_hashes), _canonical_hash(track_hashes), error,
        )
        summary = result.as_dict()
        summary["source"] = source_name.split("/")[-1] if source_name else source_name
        summary["phone_capture"] = phone_capture
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
