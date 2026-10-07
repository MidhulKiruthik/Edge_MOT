"""Counterfactual avoidable-failure label primitives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RolloutFailure:
    """Failure outcome for one matched identity in one rollout branch."""

    identity_failure: bool
    persistent_loss: bool

    @property
    def failed(self) -> bool:
        return self.identity_failure or self.persistent_loss


def avoidable_failure_label(skip: RolloutFailure, detect: RolloutFailure) -> int:
    """Return 1 only when SKIP fails and DETECT avoids that same failure."""
    return int(skip.failed and not detect.failed)


def validate_rollout_label_inputs(skip: RolloutFailure, detect: RolloutFailure) -> None:
    """Reject impossible caller state before labels enter a training dataset."""
    if not isinstance(skip, RolloutFailure) or not isinstance(detect, RolloutFailure):
        raise TypeError("both rollout outcomes must be RolloutFailure instances")
