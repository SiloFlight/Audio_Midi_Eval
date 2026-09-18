import numpy as np
from pretty_midi import PrettyMIDI

from src.schema import RawTrack
from src.constants import DEFAULT_SAMPLE_RATE

def get_index_from_timestamp(t : float) -> int:
    return round(t * DEFAULT_SAMPLE_RATE)

def _get_offsets(midi_obj : PrettyMIDI) -> list[float]:
    # PrettyMIDI exposes get_onsets(), but has no equivalent for note offsets/ends.
    return [note.end for instrument in midi_obj.instruments for note in instrument.notes]

def _get_event_times(midi_obj : PrettyMIDI) -> np.ndarray:
    onsets = midi_obj.get_onsets()
    offsets = _get_offsets(midi_obj)
    return np.concatenate([onsets, offsets])

def get_onset_offset_indices(raw_track : RawTrack, accepted_duration : float, sample_per_event = 5) -> set[int]:
    event_times = _get_event_times(raw_track.midi)

    indices = set()
    for t in event_times:
        sampled_times = np.linspace(t - accepted_duration, t + accepted_duration, sample_per_event)
        indices.update(get_index_from_timestamp(st) for st in sampled_times)

    return indices

def get_negative_indices(raw_track : RawTrack, accepted_duration : float, negative_percentage : float = 0.1) -> set[int]:
    event_times = _get_event_times(raw_track.midi)
    track_length = len(raw_track.audio)

    num_notes = len(event_times) // 2  # one onset + one offset per note

    # Mark every sample within accepted_duration of any onset/offset as excluded,
    # leaving a boolean mask of indices that are genuinely far from every event.
    negative_mask = np.ones(track_length, dtype=bool)
    for t in event_times:
        start = max(get_index_from_timestamp(t - accepted_duration), 0)
        end = min(get_index_from_timestamp(t + accepted_duration), track_length)
        negative_mask[start:end] = False

    negative_pool = np.flatnonzero(negative_mask)

    # Scales with the track's own note count instead of a flat constant, so dense
    # tracks (which also produce far more positives) get proportionally more negatives.
    sample_per_track = round(negative_percentage * num_notes)
    sample_count = min(sample_per_track, len(negative_pool))
    sampled_indices = np.random.choice(negative_pool, size=sample_count, replace=False)

    return set(sampled_indices.tolist())

def get_potential_indices(raw_track : RawTrack, accepted_duration : float) -> list[int]:

    onset_offset_indices = get_onset_offset_indices(raw_track,accepted_duration)

    negative_indices = get_negative_indices(raw_track,accepted_duration)

    return sorted(onset_offset_indices | negative_indices)