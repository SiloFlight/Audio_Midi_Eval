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

# amplitude_to_db's ref=np.max normalizes the loudest frame to 0dB, and its default
# top_db=80.0 floors anything quieter than that at max-80dB - so -80.0 is what
# silence (0 magnitude) actually maps to in this dB representation. Padding a
# sliced window with plain 0.0 would instead mean "as loud as the loudest moment
# in the track", the opposite of what compute_window's zero-padding means for raw
# audio, so slice_time_window pads with this instead.
SILENCE_DB = -80.0

def slice_time_window(full_array : np.ndarray, index : int, audio_duration : int, time_frames : int) -> np.ndarray:
    """Slice a (NOTE_BINS, T[, ...]) window out of a full-track compute_cqt/compute_hcqt
    result, at the same time position compute_window(raw_audio, index, audio_duration)
    would extract from the raw audio. time_frames must be the same T that
    compute_cqt/compute_hcqt already produces for a window of this audio_duration
    (see compute_input_dims) - it isn't re-derived here to avoid any risk of an
    off-by-one mismatch with that authoritative source.
    """
    boundary_indices = int(audio_duration/2 * DEFAULT_SAMPLE_RATE)
    frame_start = round((index - boundary_indices) / HOP_LENGTH)
    frame_end = frame_start + time_frames
    total_frames = full_array.shape[1]

    left_pad = max(0, -frame_start)
    right_pad = max(0, frame_end - total_frames)

    window = full_array[:, max(frame_start, 0):min(frame_end, total_frames)]

    if left_pad or right_pad:
        pad_width = [(0, 0), (left_pad, right_pad)] + [(0, 0)] * (full_array.ndim - 2)
        window = np.pad(window, pad_width, mode="constant", constant_values=SILENCE_DB)

    return window