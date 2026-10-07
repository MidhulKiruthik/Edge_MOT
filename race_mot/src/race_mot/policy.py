"""Pure binary DETECT/SKIP scheduling logic for local tests and later runtime use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from race_mot.domain import (
    DecisionReason,
    DetectorAction,
    PolicyDecision,
    TrackSnapshot,
    active_tracks,
    aggregate_frame_risk,
)


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    threshold_tau: float
    max_consecutive_skips: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold_tau <= 1.0:
            raise ValueError("threshold_tau must be in [0, 1]")
        if self.max_consecutive_skips < 1:
            raise ValueError("max_consecutive_skips must be positive")


class BinaryScheduler:
    """Schedule the next source frame without depending on detector hardware."""

    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self._consecutive_skips = 0

    @property
    def consecutive_skips(self) -> int:
        return self._consecutive_skips

    def reset(self) -> None:
        self._consecutive_skips = 0

    def decide_next(
        self,
        *,
        next_frame_index: int,
        tracks: Sequence[TrackSnapshot],
        track_risks: Mapping[int, float] | None,
        predictor_valid: bool = True,
    ) -> PolicyDecision:
        active = active_tracks(tracks)
        active_ids = {track.temporary_track_id for track in active}
        current_risks = {
            track_id: risk
            for track_id, risk in (track_risks or {}).items()
            if track_id in active_ids
        }
        frame_risk = aggregate_frame_risk(current_risks) if predictor_valid else None
        if not active:
            return self._detect(next_frame_index, DecisionReason.NO_TRACKS, frame_risk)
        if not predictor_valid:
            return self._detect(
                next_frame_index, DecisionReason.INVALID_RISK, frame_risk
            )
        if frame_risk is None or frame_risk >= self.config.threshold_tau:
            return self._detect(
                next_frame_index, DecisionReason.RISK_THRESHOLD, frame_risk
            )
        if self._consecutive_skips >= self.config.max_consecutive_skips:
            return self._detect(next_frame_index, DecisionReason.MAX_SKIP, frame_risk)
        return PolicyDecision(
            planned_action=DetectorAction.SKIP,
            executed_action=DetectorAction.SKIP,
            applies_to_frame_index=next_frame_index,
            reason=DecisionReason.LOW_RISK,
            frame_risk=frame_risk,
            threshold=self.config.threshold_tau,
        )

    def apply_execution(
        self,
        decision: PolicyDecision,
        *,
        scene_activity_score: float | None = None,
        guard_threshold: float | None = None,
    ) -> PolicyDecision:
        """Record the action actually taken; a guard may only upgrade SKIP."""
        executed = decision.executed_action
        triggered = (
            decision.planned_action == DetectorAction.SKIP
            and guard_threshold is not None
            and scene_activity_score is not None
            and scene_activity_score >= guard_threshold
        )
        if triggered:
            executed = DetectorAction.DETECT
        self._consecutive_skips = (
            self._consecutive_skips + 1 if executed == DetectorAction.SKIP else 0
        )
        if triggered:
            return PolicyDecision(
                planned_action=decision.planned_action,
                executed_action=executed,
                applies_to_frame_index=decision.applies_to_frame_index,
                reason=DecisionReason.SCENE_ACTIVITY_OVERRIDE,
                frame_risk=decision.frame_risk,
                threshold=decision.threshold,
                scene_activity_score=scene_activity_score,
                scene_guard_triggered=True,
            )
        return PolicyDecision(
            planned_action=decision.planned_action,
            executed_action=executed,
            applies_to_frame_index=decision.applies_to_frame_index,
            reason=decision.reason,
            frame_risk=decision.frame_risk,
            threshold=decision.threshold,
            scene_activity_score=scene_activity_score,
            scene_guard_triggered=False,
        )

    def _detect(
        self, frame_index: int, reason: DecisionReason, risk: float | None
    ) -> PolicyDecision:
        return PolicyDecision(
            planned_action=DetectorAction.DETECT,
            executed_action=DetectorAction.DETECT,
            applies_to_frame_index=frame_index,
            reason=reason,
            frame_risk=risk,
            threshold=self.config.threshold_tau,
        )
