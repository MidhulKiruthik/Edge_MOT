"""Hardware-independent runtime records for RACE-MOT."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


class SourceKind(str, Enum):
    RTSP = "rtsp"
    VIDEO_FILE = "video_file"
    MOT_SEQUENCE = "mot_sequence"


class TrackState(str, Enum):
    TRACKED = "tracked"
    PREDICTED = "predicted"
    LOST = "lost"
    REMOVED = "removed"


class DetectorAction(str, Enum):
    DETECT = "DETECT"
    SKIP = "SKIP"


class DecisionReason(str, Enum):
    INITIALIZE = "INITIALIZE"
    RISK_THRESHOLD = "RISK_THRESHOLD"
    NO_TRACKS = "NO_TRACKS"
    MAX_SKIP = "MAX_SKIP"
    INVALID_RISK = "INVALID_RISK"
    LOW_RISK = "LOW_RISK"
    SCENE_ACTIVITY_OVERRIDE = "SCENE_ACTIVITY_OVERRIDE"


@dataclass(frozen=True, slots=True)
class FramePacket:
    run_id: str
    source_index: int
    source_timestamp_ms: float | None
    arrival_monotonic_ns: int
    source_kind: SourceKind
    input_drop_count: int = 0
    image_bgr: object | None = None

    def __post_init__(self) -> None:
        if self.source_index < 0:
            raise ValueError("source_index must be non-negative")
        if self.input_drop_count < 0:
            raise ValueError("input_drop_count must be non-negative")


@dataclass(frozen=True, slots=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str = "person"

    def __post_init__(self) -> None:
        if len(self.xyxy) != 4:
            raise ValueError("xyxy must contain four coordinates")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TrackSnapshot:
    temporary_track_id: int
    xyxy: tuple[float, float, float, float]
    score: float | None
    state: TrackState
    age_frames: int


@dataclass(frozen=True, slots=True)
class DetectorScheduleState:
    source_frame_index: int
    timestamp_monotonic_seconds: float
    seconds_since_last_detector_run: float
    source_frames_since_last_detector_run: int
    consecutive_detector_skips: int


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    planned_action: DetectorAction
    executed_action: DetectorAction
    applies_to_frame_index: int
    reason: DecisionReason
    frame_risk: float | None
    threshold: float | None
    scene_activity_score: float | None = None
    scene_guard_triggered: bool = False


def aggregate_frame_risk(track_risks: Mapping[int, float]) -> float | None:
    """Use conservative max aggregation while preserving an empty-track state."""
    if not track_risks:
        return None
    values = tuple(track_risks.values())
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("track risks must be in [0, 1]")
    return max(values)


def active_tracks(tracks: Sequence[TrackSnapshot]) -> tuple[TrackSnapshot, ...]:
    return tuple(
        track
        for track in tracks
        if track.state in (TrackState.TRACKED, TrackState.PREDICTED)
    )
