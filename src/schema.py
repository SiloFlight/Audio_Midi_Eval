from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pretty_midi import PrettyMIDI
from dataclasses import dataclass

AudioWaveform = NDArray[np.float32]

class TrackType(str, Enum):
    MAESTRO = "MAESTRO"
    MAPS = "MAPS"
    Synthetic = "Synthetic"

class InputTypes(str,Enum):
    Waveform = "Waveform"
    CQT = "CQT"
    HCQT = "HCQT"

class CurriculumStage(str, Enum):
    ISOL = "ISOL"
    CHORDS = "Chords"
    FULL = "Full"

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
    audio: AudioWaveform
    midi: PrettyMIDI
