"""Deterministic ByteTrack-compatible lifecycle adapter.

The project API is kept independent of optional Torch/LAP packages.  On the
Jetson baseline this adapter uses deterministic greedy IoU association with the
same update/skip lifecycle semantics required by the pinned ByteTrack source.
The backend and pinned upstream commit are recorded in every run manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy

from race_mot.domain import Detection, FramePacket, TrackSnapshot, TrackState


@dataclass(slots=True)
class _Track:
    track_id: int
    box: tuple[float, float, float, float]
    score: float
    age_frames: int = 1
    missed: int = 0
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0]); y1 = max(left[1], right[1])
    x2 = min(left[2], right[2]); y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


class ByteTrackAdapter:
    """Stable temporary-ID tracker with explicit detector-run/skip events."""

    def __init__(self, *, match_iou: float = 0.30, max_lost_frames: int = 30) -> None:
        if not 0.0 < match_iou <= 1.0:
            raise ValueError("match_iou must be in (0, 1]")
        if max_lost_frames < 1:
            raise ValueError("max_lost_frames must be positive")
        self.match_iou = match_iou
        self.max_lost_frames = max_lost_frames
        self.implementation_version = "d1bf0191adff59bc8fcfeaa0b33d3d1642552a99"
        self.backend = "deterministic_iou_compat"
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self.last_event: str | None = None
        self.last_source_index: int | None = None

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "family": "bytetrack",
            "implementation_version": self.implementation_version,
            "backend": self.backend,
            "match_iou": self.match_iou,
            "max_lost_frames": self.max_lost_frames,
        }

    def state_dict(self) -> dict[str, object]:
        """Serialize the complete deterministic tracker state for cloning.

        The state contains only temporary IDs and geometry. It is used by the
        offline paired-rollout generator and is never a persistent identity.
        """
        return {
            "schema_version": 1,
            "next_id": self._next_id,
            "last_source_index": self.last_source_index,
            "tracks": {
                str(track_id): {
                    "track_id": track.track_id,
                    "box": list(track.box),
                    "score": track.score,
                    "age_frames": track.age_frames,
                    "missed": track.missed,
                    "velocity": list(track.velocity),
                }
                for track_id, track in sorted(self._tracks.items())
            },
        }

    def clone(self) -> "ByteTrackAdapter":
        """Return an independent tracker with identical serialized state."""
        cloned = ByteTrackAdapter(match_iou=self.match_iou, max_lost_frames=self.max_lost_frames)
        cloned._next_id = self._next_id
        cloned.last_source_index = self.last_source_index
        cloned.last_event = self.last_event
        cloned._tracks = copy.deepcopy(self._tracks)
        return cloned

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self.last_event = "reset"
        self.last_source_index = None

    def initialize(self, detections: list[Detection], frame_packet: FramePacket) -> list[TrackSnapshot]:
        self.reset()
        self.last_event = "initialize"
        return self.update(detections, frame_packet)

    def _snapshot(self, state: TrackState, track: _Track) -> TrackSnapshot:
        return TrackSnapshot(track.track_id, track.box, track.score, state, track.age_frames)

    def update(self, detections: list[Detection], frame_packet: FramePacket) -> list[TrackSnapshot]:
        """Consume a detector result; an empty list is a real no-person result."""
        self.last_event = "update"
        self.last_source_index = frame_packet.source_index
        boxes = [tuple(float(value) for value in detection.xyxy) for detection in detections]
        scores = [float(detection.score) for detection in detections]
        unmatched_tracks = set(self._tracks)
        unmatched_detections = set(range(len(boxes)))
        candidates = sorted(
            (
                _iou(track.box, box),
                track_id,
                detection_index,
            )
            for track_id, track in self._tracks.items()
            for detection_index, box in enumerate(boxes)
            if _iou(track.box, box) >= self.match_iou
        )
        for _, track_id, detection_index in sorted(candidates, reverse=True):
            if track_id not in unmatched_tracks or detection_index not in unmatched_detections:
                continue
            track = self._tracks[track_id]
            old = track.box
            new = boxes[detection_index]
            track.velocity = tuple(new[i] - old[i] for i in range(4))
            track.box = new
            track.score = scores[detection_index]
            track.age_frames += 1
            track.missed = 0
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(detection_index)
        for track_id in unmatched_tracks:
            track = self._tracks[track_id]
            track.missed += 1
            track.age_frames += 1
        for detection_index in sorted(unmatched_detections):
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = _Track(track_id, boxes[detection_index], scores[detection_index])
        self._tracks = {
            track_id: track for track_id, track in self._tracks.items()
            if track.missed <= self.max_lost_frames
        }
        # A detector-run result exposes only matched/new detections.  This keeps
        # update([]) observably different from skip(), which emits predictions.
        return [
            self._snapshot(TrackState.TRACKED, track)
            for track_id, track in sorted(self._tracks.items())
            if track_id not in unmatched_tracks and track.missed == 0
        ]

    def skip(self, frame_packet: FramePacket) -> list[TrackSnapshot]:
        """Advance motion without claiming that detection ran."""
        self.last_event = "skip"
        self.last_source_index = frame_packet.source_index
        predictions: list[TrackSnapshot] = []
        for track_id in sorted(self._tracks):
            track = self._tracks[track_id]
            track.box = tuple(track.box[i] + track.velocity[i] for i in range(4))
            track.age_frames += 1
            track.missed += 1
            if track.missed <= self.max_lost_frames:
                predictions.append(self._snapshot(TrackState.PREDICTED, track))
        self._tracks = {
            track_id: track for track_id, track in self._tracks.items()
            if track.missed <= self.max_lost_frames
        }
        return predictions
