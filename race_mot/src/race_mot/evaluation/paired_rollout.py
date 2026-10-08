"""Deterministic paired skip/detect rollout contract for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from race_mot.evaluation.protocol import RolloutProtocol, avoidable_failure, branch_failure


@dataclass(frozen=True, slots=True)
class BranchObservation:
    """One branch's matched identity observations after an anchor frame."""

    branch: str
    anchor_state_hash: str
    future_frame_indices: tuple[int, ...]
    visible: tuple[bool, ...]
    assigned_identity: tuple[int | None, ...]
    boundary_censored: bool = False

    def __post_init__(self) -> None:
        if self.branch not in {"skip", "detect"}:
            raise ValueError("branch must be skip or detect")
        if len(self.future_frame_indices) != len(self.visible) or len(self.visible) != len(self.assigned_identity):
            raise ValueError("future frame, visibility, and identity arrays must have equal length")


@dataclass(frozen=True, slots=True)
class PairedRollout:
    """Auditable pair of outcomes generated from one anchor state."""

    anchor_frame_index: int
    target_identity: int
    skip_failed: bool
    detect_failed: bool
    label: int | None
    exclusion: str | None
    anchor_state_hash: str
    future_frame_indices: tuple[int, ...]


def pair_observations(
    *,
    anchor_frame_index: int,
    target_identity: int,
    skip: BranchObservation,
    detect: BranchObservation,
    protocol: RolloutProtocol,
) -> PairedRollout:
    """Validate identical-state branches and produce one eligible label."""
    if skip.branch != "skip" or detect.branch != "detect":
        raise ValueError("observations must be supplied as skip and detect branches")
    if skip.anchor_state_hash != detect.anchor_state_hash:
        raise ValueError("paired branches must share the identical anchor state hash")
    if skip.future_frame_indices != detect.future_frame_indices:
        raise ValueError("paired branches must use identical future frame indices")
    if len(skip.visible) != protocol.horizon_frames:
        raise ValueError("branch observations must cover protocol horizon")

    skip_failed = branch_failure(
        visible=list(skip.visible),
        assigned_identity=list(skip.assigned_identity),
        target_identity=target_identity,
        protocol=protocol,
    )
    detect_failed = branch_failure(
        visible=list(detect.visible),
        assigned_identity=list(detect.assigned_identity),
        target_identity=target_identity,
        protocol=protocol,
    )
    exclusion = None
    if skip.boundary_censored or detect.boundary_censored:
        exclusion = "boundary_censored"
    elif not any(skip.visible) or not any(detect.visible):
        exclusion = "ineligible_no_visible_observation"
    label = None if exclusion else avoidable_failure(skip_failed, detect_failed)
    return PairedRollout(
        anchor_frame_index=anchor_frame_index,
        target_identity=target_identity,
        skip_failed=skip_failed,
        detect_failed=detect_failed,
        label=label,
        exclusion=exclusion,
        anchor_state_hash=skip.anchor_state_hash,
        future_frame_indices=skip.future_frame_indices,
    )


def run_paired_rollout(
    *,
    anchor_frame_index: int,
    target_identity: int,
    anchor_state_hash: str,
    future_frame_indices: Sequence[int],
    protocol: RolloutProtocol,
    branch_runner: Callable[[str, str, tuple[int, ...]], BranchObservation],
) -> PairedRollout:
    """Run both branches from one serialized anchor-state identity.

    ``branch_runner`` owns the tracker/detector implementation. This function
    enforces the data-leakage contract and never permits a branch-specific
    future-frame list or anchor state.
    """
    frames = tuple(future_frame_indices)
    if len(frames) != protocol.horizon_frames:
        raise ValueError("future_frame_indices must cover protocol horizon")
    skip = branch_runner("skip", anchor_state_hash, frames)
    detect = branch_runner("detect", anchor_state_hash, frames)
    return pair_observations(
        anchor_frame_index=anchor_frame_index,
        target_identity=target_identity,
        skip=skip,
        detect=detect,
        protocol=protocol,
    )
