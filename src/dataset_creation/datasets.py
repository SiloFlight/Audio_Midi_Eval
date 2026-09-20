import random

import numpy as np
import tensorflow as tf

from src.audio_processing import compute_cqt, compute_hcqt
from src.constants import NOTE_BINS
from src.schema import InputTypes, TrackInfo
from src.data.misc import load_track_info
from src.data.analysis.track_indices import get_maestro_track_index, get_maps_track_index
from src.dataset_creation.index_detection import get_potential_indices
from src.dataset_creation.examples import create_labels, create_waveform_features, create_cqt_features, create_hcqt_features
from src.models import waveform, cqt, hcqt

_FEATURE_FNS = {
    InputTypes.Waveform: create_waveform_features,
    InputTypes.CQT: create_cqt_features,
    InputTypes.HCQT: create_hcqt_features,
}

_INPUT_DIMS_FNS = {
    InputTypes.Waveform: waveform.compute_input_dims,
    InputTypes.CQT: cqt.compute_input_dims,
    InputTypes.HCQT: hcqt.compute_input_dims,
}

# Builds whatever each input type needs once per track, before looping over that
# track's indices - for CQT/HCQT this is the expensive part (the full-track
# transform), computed once instead of once per index. input_dims is this
# input_type's compute_input_dims(audio_duration) result, computed once per
# dataset build (see _build_base_dataset) since it doesn't vary per track.
_TRACK_CONTEXT_FNS = {
    InputTypes.Waveform: lambda raw_track, input_dims: raw_track,
    InputTypes.CQT: lambda raw_track, input_dims: (compute_cqt(raw_track.audio), input_dims[1]),
    InputTypes.HCQT: lambda raw_track, input_dims: (compute_hcqt(raw_track.audio), input_dims[1]),
}

SHUFFLE_BUFFER_SIZE = 10_000

# Caps how many examples a single track can contribute. Without this, a track
# with more potential indices than SHUFFLE_BUFFER_SIZE (confirmed via
# diagnose_track_example_counts.py - some MAESTRO recordings produce 200K+
# potential indices against a 10K buffer) dominates the shuffle window for a
# long stretch, so many consecutive training steps end up correlated instead
# of drawing from a representative mix of tracks.
MAX_EXAMPLES_PER_TRACK = SHUFFLE_BUFFER_SIZE // 10

# compute_cqt/compute_hcqt cost is paid once per track regardless of
# MAX_EXAMPLES_PER_TRACK (the full track has to be transformed before any
# windows can be sliced from it), so an outlier-length track's compute cost
# gets amortized over very few retained examples - benchmarked at up to a
# ~60-290x worse per-example cost for tracks in the tens-of-minutes range.
# Tracks over this are excluded entirely rather than paying that cost.
MAX_TRACK_DURATION = 5 * 60  # seconds

def _as_shape(dims) -> tuple[int, ...]:
    return (dims,) if isinstance(dims, int) else tuple(dims)

def split_tracks() -> tuple[list[TrackInfo], list[TrackInfo], list[TrackInfo]]:
    maestro_tracks = get_maestro_track_index()
    maps_tracks = get_maps_track_index()

    train_tracks = [t for t in maestro_tracks if t.track_desc == "train"] + maps_tracks
    val_tracks = [t for t in maestro_tracks if t.track_desc == "validation"]
    test_tracks = [t for t in maestro_tracks if t.track_desc == "test"]

    return train_tracks, val_tracks, test_tracks

def _example_generator(track_infos : list[TrackInfo], input_type : InputTypes, audio_duration : int, accepted_duration : float, input_dims, rng : np.random.Generator):
    feature_fn = _FEATURE_FNS[input_type]
    context_fn = _TRACK_CONTEXT_FNS[input_type]

    for track_info in track_infos:
        raw_track = load_track_info(track_info)
        context = context_fn(raw_track, input_dims)

        indices = get_potential_indices(raw_track, accepted_duration, rng=rng, max_examples=MAX_EXAMPLES_PER_TRACK)
        labels = create_labels(raw_track, indices, accepted_duration)

        for index, label in zip(indices, labels):
            yield feature_fn(context, index, audio_duration), label

def _build_base_dataset(track_infos : list[TrackInfo], input_type : InputTypes, audio_duration : int, accepted_duration : float, seed : int) -> tf.data.Dataset:
    # Filtered on track_info.duration (already known from the index) rather than
    # inside _example_generator, so excluded tracks are never loaded from disk at all.
    track_infos = [t for t in track_infos if t.duration <= MAX_TRACK_DURATION]
    shuffled_tracks = random.Random(seed).sample(track_infos, len(track_infos))
    rng = np.random.default_rng(seed)

    input_dims = _INPUT_DIMS_FNS[input_type](audio_duration)
    feature_shape = _as_shape(input_dims)

    output_signature = (
        tf.TensorSpec(shape=feature_shape, dtype=tf.float32),
        tf.TensorSpec(shape=(2, NOTE_BINS), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: _example_generator(shuffled_tracks, input_type, audio_duration, accepted_duration, input_dims, rng),
        output_signature=output_signature,
    )

    return dataset.shuffle(SHUFFLE_BUFFER_SIZE, seed=seed)

def create_training_set(input_type : InputTypes, audio_duration : int, accepted_duration : float, batch_size : int, seed : int, n : int = -1) -> tf.data.Dataset:
    train_tracks, _, _ = split_tracks()

    dataset = _build_base_dataset(train_tracks, input_type, audio_duration, accepted_duration, seed)

    return dataset.take(n).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def create_validation_set(input_type : InputTypes, audio_duration : int, accepted_duration : float, batch_size : int, seed : int, n : int) -> tf.data.Dataset:
    _, val_tracks, _ = split_tracks()

    dataset = _build_base_dataset(val_tracks, input_type, audio_duration, accepted_duration, seed)

    # .cache() freezes whichever n examples the first pass produces (including the
    # shuffle order) in memory, so every subsequent epoch re-reads the exact same
    # set instead of re-running the generator/shuffle (which would otherwise draw
    # different examples each time, per tf.data's reshuffle_each_iteration default).
    return dataset.take(n).cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

def create_test_set(input_type : InputTypes, audio_duration : int, accepted_duration : float, batch_size : int, seed : int, n : int = -1) -> tf.data.Dataset:
    _, _, test_tracks = split_tracks()

    dataset = _build_base_dataset(test_tracks, input_type, audio_duration, accepted_duration, seed)

    return dataset.take(n).batch(batch_size).prefetch(tf.data.AUTOTUNE)
