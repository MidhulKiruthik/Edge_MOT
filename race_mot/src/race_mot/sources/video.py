"""Sequential OpenCV source adapter for local files and private phone streams."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Iterator

from race_mot.domain import FramePacket, SourceKind
from race_mot.stream_probe import _open_capture


class VideoSource:
    """Yield decoded frames while retaining decoder indices and observed drops.

    The adapter never writes frames or the source identifier. Network sources
    use the same bounded OpenCV timeout/open behavior as the probe utility.
    A failed read on a local file is treated as end of file; a failed network
    read is reported as an error so a future reconnect policy cannot hide loss.
    """

    def __init__(
        self,
        source: str | Path,
        *,
        run_id: str,
        max_frames: int | None = None,
    ) -> None:
        if not run_id.strip():
            raise ValueError("run_id must be non-empty")
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be positive when provided")
        self.source = str(source)
        self.run_id = run_id
        self.max_frames = max_frames
        lowered = self.source.lower()
        self.source_kind = (
            SourceKind.RTSP
            if lowered.startswith(("rtsp://", "rtsps://", "http://", "https://"))
            else SourceKind.VIDEO_FILE
        )

    def __iter__(self) -> Iterator[FramePacket]:
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV Python (cv2) is unavailable; use the JetPack-compatible system OpenCV."
            ) from exc

        capture, _backend = _open_capture(cv2, self.source)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("could not open configured video or RTSP source")

        is_network = self.source_kind == SourceKind.RTSP
        previous_index: int | None = None
        sequential_index = 0
        yielded = 0
        try:
            if is_network:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            while self.max_frames is None or yielded < self.max_frames:
                ok, image = capture.read()
                if not ok or image is None:
                    if is_network:
                        raise RuntimeError("source read failed; stream replay stopped")
                    break
                sequential_index += 1
                raw_index = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
                source_index = (
                    int(round(raw_index))
                    if math.isfinite(raw_index) and raw_index >= 1
                    else sequential_index
                )
                input_drop_count = 0
                if previous_index is not None and source_index > previous_index + 1:
                    input_drop_count = source_index - previous_index - 1
                raw_timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                source_timestamp_ms = (
                    raw_timestamp
                    if math.isfinite(raw_timestamp) and raw_timestamp >= 0
                    else None
                )
                previous_index = source_index
                yielded += 1
                yield FramePacket(
                    run_id=self.run_id,
                    source_index=source_index,
                    source_timestamp_ms=source_timestamp_ms,
                    arrival_monotonic_ns=time.monotonic_ns(),
                    source_kind=self.source_kind,
                    input_drop_count=input_drop_count,
                    image_bgr=image,
                )
        finally:
            capture.release()
