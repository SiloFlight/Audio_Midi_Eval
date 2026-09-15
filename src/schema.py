from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pretty_midi import PrettyMIDI
from dataclasses import dataclass


class TrackType(str, Enum):
    MAESTRO = "MAESTRO"
    MAPS = "MAPS"
    Synthetic = "Synthetic"

@dataclass
class TrackInfo:
    duration: float
    track_type: TrackType
    track_name: str
    track_desc: str
    wav_path: Path
    mid_path: Path

@dataclass
class RawTrack:
    track_info: TrackInfo
    audio: NDArray[np.float32]
    midi: PrettyMIDI
