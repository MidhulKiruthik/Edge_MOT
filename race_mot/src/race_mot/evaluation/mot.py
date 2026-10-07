"""Small, dependency-free reader for MOTChallenge ground-truth files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GroundTruthBox:
    frame_index: int
    identity: int
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_id: int
    visibility: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)


def read_gt(path: Path, *, person_class_id: int = 1) -> tuple[GroundTruthBox, ...]:
    """Read MOTChallenge ``gt.txt`` and retain visible person annotations."""
    boxes: list[GroundTruthBox] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        fields = [field.strip() for field in raw_line.split(",")]
        if len(fields) < 9:
            raise ValueError(
                f"{path}:{line_number}: expected at least 9 comma-separated fields"
            )
        try:
            frame, identity = int(fields[0]), int(fields[1])
            values = [float(value) for value in fields[2:9]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid numeric field") from exc
        x, y, width, height, confidence, class_id, visibility = values
        if class_id != person_class_id or width <= 0 or height <= 0 or visibility <= 0:
            continue
        boxes.append(
            GroundTruthBox(
                frame,
                identity,
                x,
                y,
                width,
                height,
                confidence,
                int(class_id),
                visibility,
            )
        )
    return tuple(boxes)


def group_by_frame(
    boxes: tuple[GroundTruthBox, ...],
) -> dict[int, tuple[GroundTruthBox, ...]]:
    grouped: dict[int, list[GroundTruthBox]] = {}
    for box in boxes:
        grouped.setdefault(box.frame_index, []).append(box)
    return {frame: tuple(items) for frame, items in sorted(grouped.items())}
