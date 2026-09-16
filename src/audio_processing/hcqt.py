import librosa
import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE,HOP_LENGTH,MIN_NOTE,NOTE_BINS,HARMONICS
from src.schema import AudioWaveform

def compute_cqt(audio : AudioWaveform, scalar : float = 1) -> np.ndarray:
    cqt = librosa.cqt(audio,
                       sr=DEFAULT_SAMPLE_RATE,
                       hop_length=HOP_LENGTH,
                       fmin=librosa.midi_to_hz(MIN_NOTE) * scalar,
                       n_bins=NOTE_BINS,
                       bins_per_octave=12)

    return librosa.amplitude_to_db(np.abs(cqt), ref=np.max)

def compute_hcqt(audio : AudioWaveform) -> np.ndarray:
    cqts = []

    for harmonic in HARMONICS:
        cqts.append(compute_cqt(audio,harmonic))

    # Stack the cqts into a harmonic cqt: the harmonic axis is last (channel-last),
    return np.stack(cqts, axis=-1)