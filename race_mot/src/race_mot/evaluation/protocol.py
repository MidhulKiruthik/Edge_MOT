"""Explicit Phase 2 label protocol; no silent research defaults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RolloutProtocol:
    """Frozen settings required before paired labels can be generated."""

    anchor_iou: float
    visible_minimum: float
    horizon_frames: int
    persistence_frames: int
    detector_version: str
    tracker_version: str
    matching_rule: str = "one_to_one_iou"
    sequence_boundary_censor: bool = True
    tracker_initialization: str = "reset_at_sequence_start_and_anchor_clone"
    branch_semantics: str = (
        "clone_identical_anchor_state; skip branch omits detector at t+1; "
        "detect branch runs detector at t+1; both use detector_every_frame thereafter"
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.anchor_iou <= 1.0:
            raise ValueError("anchor_iou must be in (0, 1]")
        if not 0.0 <= self.visible_minimum <= 1.0:
            raise ValueError("visible_minimum must be in [0, 1]")
        if self.horizon_frames < 1 or self.persistence_frames < 1:
            raise ValueError("horizon_frames and persistence_frames must be positive")
        if self.persistence_frames > self.horizon_frames:
            raise ValueError("persistence_frames cannot exceed horizon_frames")
        if not self.detector_version.strip() or not self.tracker_version.strip():
            raise ValueError("detector_version and tracker_version are required")
        if self.matching_rule != "one_to_one_iou":
            raise ValueError("only the documented one_to_one_iou rule is supported")
        if not self.tracker_initialization.strip() or not self.branch_semantics.strip():
            raise ValueError("tracker initialization and branch semantics are required")

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "RolloutProtocol":
        """Build a protocol from the versioned JSON record."""
        return cls(
            anchor_iou=float(values["anchor_iou"]),
            visible_minimum=float(values["visible_minimum"]),
            horizon_frames=int(values["horizon_frames"]),
            persistence_frames=int(values["persistence_frames"]),
            detector_version=str(values["detector_version"]),
            tracker_version=str(values["tracker_version"]),
            matching_rule=str(values.get("matching_rule", "one_to_one_iou")),
            sequence_boundary_censor=bool(values.get("sequence_boundary_censor", True)),
            tracker_initialization=str(
                values.get("tracker_initialization", "reset_at_sequence_start_and_anchor_clone")
            ),
            branch_semantics=str(values.get("branch_semantics", "")),
        )


def branch_failure(
    *,
    visible: list[bool],
    assigned_identity: list[int | None],
    target_identity: int,
    protocol: RolloutProtocol,
) -> bool:
    """Apply the documented future failure predicate to one branch.

    ``visible`` and ``assigned_identity`` cover frames ``t+1`` through
    ``t+K``. A branch fails on a visible identity mismatch, or on a run of
    ``M`` visible frames where the anchored identity is not retained. Censored
    anchors must be removed before this function is called.
    """
    if len(visible) != protocol.horizon_frames or len(assigned_identity) != protocol.horizon_frames:
        raise ValueError("branch arrays must cover exactly horizon_frames")
    for is_visible, assigned in zip(visible, assigned_identity):
        if is_visible and assigned is not None and assigned != target_identity:
            return True
    for start in range(protocol.horizon_frames - protocol.persistence_frames + 1):
        window_visible = visible[start : start + protocol.persistence_frames]
        window_ids = assigned_identity[start : start + protocol.persistence_frames]
        if all(window_visible) and all(identity != target_identity for identity in window_ids):
            return True
    return False


def avoidable_failure(skip_failed: bool, detect_failed: bool) -> int:
    """Return the frozen target ``F_skip * (1 - F_detect)``."""
    return int(bool(skip_failed) and not bool(detect_failed))
