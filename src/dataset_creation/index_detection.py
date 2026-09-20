import numpy as np
from pretty_midi import PrettyMIDI

from src.schema import RawTrack
from src.constants import DEFAULT_SAMPLE_RATE

def get_index_from_timestamp(t : float) -> int:
    return round(t * DEFAULT_SAMPLE_RATE)

def _get_offsets(midi_obj : PrettyMIDI) -> list[float]:
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

def get_negative_indices(raw_track : RawTrack, accepted_duration : float, negative_percentage : float = 2.5, rng : np.random.Generator | None = None) -> set[int]:
    rng = rng if rng is not None else np.random.default_rng()

    event_times = _get_event_times(raw_track.midi)
    track_length = len(raw_track.audio)

    num_notes = len(event_times) // 2  # one onset + one offset per note

    negative_mask = np.ones(track_length, dtype=bool)
    for t in event_times:
        start = max(get_index_from_timestamp(t - accepted_duration), 0)
        end = min(get_index_from_timestamp(t + accepted_duration), track_length)
        negative_mask[start:end] = False

    negative_pool = np.flatnonzero(negative_mask)

    sample_per_track = round(negative_percentage * num_notes)
    sample_count = min(sample_per_track, len(negative_pool))
    sampled_indices = rng.choice(negative_pool, size=sample_count, replace=False)

    return set(sampled_indices.tolist())

def get_potential_indices(raw_track : RawTrack, accepted_duration : float, rng : np.random.Generator | None = None, max_examples : int | None = None) -> list[int]:
    rng = rng if rng is not None else np.random.default_rng()

    onset_offset_indices = sorted(get_onset_offset_indices(raw_track,accepted_duration))
    negative_indices = sorted(get_negative_indices(raw_track,accepted_duration,rng=rng))

    total = len(onset_offset_indices) + len(negative_indices)
    if max_examples is not None and total > max_examples:
        # Cap this track's contribution while preserving its own pre-cap
        # positive/negative ratio - splitting the cap proportionally rather than
        # sampling the union keeps a track's negative_percentage tuning intact
        # instead of leaving it to chance.
        pos_target = round(max_examples * len(onset_offset_indices) / total)
        neg_target = max_examples - pos_target

        pos_count = min(pos_target, len(onset_offset_indices))
        neg_count = min(neg_target, len(negative_indices))

        onset_offset_indices = rng.choice(onset_offset_indices, size=pos_count, replace=False).tolist()
        negative_indices = rng.choice(negative_indices, size=neg_count, replace=False).tolist()

    return sorted(set(onset_offset_indices) | set(negative_indices))