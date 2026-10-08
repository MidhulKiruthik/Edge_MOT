"""Sequential, in-memory reader for a MOTChallenge image sequence."""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterator
import time

from race_mot.domain import FramePacket, SourceKind


@dataclass(frozen=True, slots=True)
class MotSequenceInfo:
    name: str
    image_dir: Path
    image_extension: str
    frame_count: int
    width: int
    height: int
    frame_rate: float


def read_sequence_info(sequence_dir: Path) -> MotSequenceInfo:
    """Read and validate the sequence metadata needed for deterministic replay."""
    sequence_dir = Path(sequence_dir)
    metadata_path = sequence_dir / "seqinfo.ini"
    if not metadata_path.is_file():
        raise ValueError(f"MOT sequence is missing {metadata_path.name}")

    parser = ConfigParser()
    try:
        loaded = parser.read(metadata_path, encoding="utf-8")
        if not loaded or not parser.has_section("Sequence"):
            raise ValueError("seqinfo.ini is missing the [Sequence] section")
        section = parser["Sequence"]
        name = section.get("name", sequence_dir.name)
        relative_image_dir = Path(section.get("imDir", "img1"))
        image_dir = (sequence_dir / relative_image_dir).resolve()
        extension = section.get("imExt", ".jpg")
        frame_count = section.getint("seqLength")
        width = section.getint("imWidth")
        height = section.getint("imHeight")
        frame_rate = section.getfloat("frameRate")
    except (KeyError, ValueError) as exc:
        raise ValueError("seqinfo.ini has invalid sequence metadata") from exc

    if not name or Path(extension).name != extension or not extension.startswith("."):
        raise ValueError("seqinfo.ini has invalid sequence name or image extension")
    if (
        frame_count < 1
        or width < 1
        or height < 1
        or not math.isfinite(frame_rate)
        or frame_rate <= 0
    ):
        raise ValueError("seqinfo.ini dimensions, length, and frame rate must be positive")
    try:
        image_dir.relative_to(sequence_dir.resolve())
    except ValueError as exc:
        raise ValueError("seqinfo.ini image directory must stay inside the sequence") from exc
    if not image_dir.is_dir():
        raise ValueError(f"MOT sequence is missing image directory: {image_dir.name}")

    return MotSequenceInfo(
        name=name,
        image_dir=image_dir,
        image_extension=extension,
        frame_count=frame_count,
        width=width,
        height=height,
        frame_rate=frame_rate,
    )


class MotSequenceSource:
    """Decode every MOT image in source order without writing image data to disk.

    ``source_index`` retains MOTChallenge's one-based frame number. Timestamps
    are nominal sequence times derived from ``frameRate``; arrival times use
    the local monotonic clock after each image has decoded.
    """

    def __init__(self, sequence_dir: Path, *, run_id: str) -> None:
        if not run_id.strip():
            raise ValueError("run_id must be non-empty")
        self.info = read_sequence_info(sequence_dir)
        self.run_id = run_id

    def __iter__(self) -> Iterator[FramePacket]:
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV Python (cv2) is unavailable; use the JetPack-compatible system OpenCV."
            ) from exc

        for frame_number in range(1, self.info.frame_count + 1):
            image_path = self.info.image_dir / (
                f"{frame_number:06d}{self.info.image_extension}"
            )
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(
                    f"could not decode MOT source frame {frame_number}; replay stopped"
                )
            actual_height, actual_width = image.shape[:2]
            if (actual_width, actual_height) != (self.info.width, self.info.height):
                raise RuntimeError(
                    f"MOT source frame {frame_number} has dimensions "
                    f"{actual_width}x{actual_height}; expected "
                    f"{self.info.width}x{self.info.height}"
                )
            yield FramePacket(
                run_id=self.run_id,
                source_index=frame_number,
                source_timestamp_ms=(frame_number - 1) * 1000.0 / self.info.frame_rate,
                arrival_monotonic_ns=time.monotonic_ns(),
                source_kind=SourceKind.MOT_SEQUENCE,
                input_drop_count=0,
                image_bgr=image,
            )
