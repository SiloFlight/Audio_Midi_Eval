import numpy as np

from src.audio_processing import compute_window, compute_cqt, compute_hcqt
from src.constants import DEFAULT_SAMPLE_RATE, MIN_NOTE, MAX_NOTE, NOTE_BINS
from src.schema import RawTrack

ONSET_ROW = 0
OFFSET_ROW = 1

def _sorted_note_events(raw_track : RawTrack) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts, start_bins, ends, end_bins = [], [], [], []

    for instrument in raw_track.midi.instruments:
        if instrument.is_drum:
            continue

        for note in instrument.notes:
            if not (MIN_NOTE <= note.pitch <= MAX_NOTE):
                continue

            note_bin = note.pitch - MIN_NOTE

            starts.append(note.start)
            start_bins.append(note_bin)
            ends.append(note.end)
            end_bins.append(note_bin)

    starts, start_bins = np.asarray(starts), np.asarray(start_bins)
    ends, end_bins = np.asarray(ends), np.asarray(end_bins)

    start_order = np.argsort(starts)
    end_order = np.argsort(ends)

    return starts[start_order], start_bins[start_order], ends[end_order], end_bins[end_order]

def _bins_within(times : np.ndarray, bins : np.ndarray, center_time : float, accepted_duration : float) -> np.ndarray:
    lo = np.searchsorted(times, center_time - accepted_duration, side="left")
    hi = np.searchsorted(times, center_time + accepted_duration, side="right")

    return bins[lo:hi]

def create_labels(raw_track : RawTrack, indices : list[int], accepted_duration : float) -> list[np.ndarray]:
    # Sorting once per track lets each index do an O(log notes) binary search
    # instead of an O(notes) linear scan, turning the per-track cost from
    # O(indices * notes) into roughly O(indices * log(notes)).
    onset_times, onset_bins, offset_times, offset_bins = _sorted_note_events(raw_track)

    labels = []

    for index in indices:
        center_time = index / DEFAULT_SAMPLE_RATE
        label = np.zeros((2, NOTE_BINS), dtype=np.float32)

        label[ONSET_ROW, _bins_within(onset_times, onset_bins, center_time, accepted_duration)] = 1.0
        label[OFFSET_ROW, _bins_within(offset_times, offset_bins, center_time, accepted_duration)] = 1.0

        labels.append(label)

    return labels

def create_waveform_features(raw_track : RawTrack, index : int, audio_duration : int) -> np.ndarray:
    return compute_window(raw_track.audio, index, audio_duration)

def create_cqt_features(raw_track : RawTrack, index : int, audio_duration : int) -> np.ndarray:
    window = compute_window(raw_track.audio, index, audio_duration)
    features = compute_cqt(window)

    return np.expand_dims(features, axis=-1)  # (NOTE_BINS, T, 1) - matches build_cnn_model's input shape

def create_hcqt_features(raw_track : RawTrack, index : int, audio_duration : int) -> np.ndarray:
    window = compute_window(raw_track.audio, index, audio_duration)

    return compute_hcqt(window)
