"""Input adapters used by the local MOT runtime."""

from race_mot.sources.mot_sequence import MotSequenceSource, MotSequenceInfo
from race_mot.sources.video import VideoSource

__all__ = ["MotSequenceInfo", "MotSequenceSource", "VideoSource"]
