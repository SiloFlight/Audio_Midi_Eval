import numpy as np

from src.audio_processing import compute_window, compute_cqt, compute_hcqt
from src.constants import DEFAULT_SAMPLE_RATE, MIN_NOTE, MAX_NOTE, NOTE_BINS
from src.schema import RawTrack

ONSET_ROW = 0
OFFSET_ROW = 1

def create_label(raw_track : RawTrack, index : int, accepted_duration : float) -> np.ndarray:
    center_time = index / DEFAULT_SAMPLE_RATE

    labels = np.zeros((2, NOTE_BINS), dtype=np.float32)

    for instrument in raw_track.midi.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            if not (MIN_NOTE <= note.pitch <= MAX_NOTE):
                continue

            note_bin = note.pitch - MIN_NOTE

            if abs(note.start - center_time) <= accepted_duration:
                labels[ONSET_ROW, note_bin] = 1.0

            if abs(note.end - center_time) <= accepted_duration:
                labels[OFFSET_ROW, note_bin] = 1.0

    return labels

def create_waveform_example(raw_track : RawTrack, index : int, audio_duration : int, accepted_duration : float) -> tuple[np.ndarray, np.ndarray]:
    features = compute_window(raw_track.audio, index, audio_duration)
    labels = create_label(raw_track, index, accepted_duration)

    return features, labels

def create_cqt_example(raw_track : RawTrack, index : int, audio_duration : int, accepted_duration : float) -> tuple[np.ndarray, np.ndarray]:
    window = compute_window(raw_track.audio, index, audio_duration)
    features = compute_cqt(window)
    features = np.expand_dims(features, axis=-1)  # (NOTE_BINS, T, 1) - matches build_cnn_model's input shape

    labels = create_label(raw_track, index, accepted_duration)

    return features, labels

def create_hcqt_example(raw_track : RawTrack, index : int, audio_duration : int, accepted_duration : float) -> tuple[np.ndarray, np.ndarray]:
    window = compute_window(raw_track.audio, index, audio_duration)
    features = compute_hcqt(window)

    labels = create_label(raw_track, index, accepted_duration)

    return features, labels