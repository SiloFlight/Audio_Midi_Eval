import random

import numpy as np
import tensorflow as tf

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

SHUFFLE_BUFFER_SIZE = 10_000

def _as_shape(dims) -> tuple[int, ...]:
    return (dims,) if isinstance(dims, int) else tuple(dims)

def split_tracks() -> tuple[list[TrackInfo], list[TrackInfo]]:
    maestro_tracks = get_maestro_track_index()
    maps_tracks = get_maps_track_index()

    test_tracks = [t for t in maestro_tracks if t.track_desc == "test"]
    train_tracks = [t for t in maestro_tracks if t.track_desc != "test"] + maps_tracks

    return train_tracks, test_tracks

def _example_generator(track_infos : list[TrackInfo], input_type : InputTypes, audio_duration : int, accepted_duration : float, rng : np.random.Generator):
    feature_fn = _FEATURE_FNS[input_type]

    for track_info in track_infos:
        raw_track = load_track_info(track_info)

        indices = get_potential_indices(raw_track, accepted_duration, rng=rng)
        labels = create_labels(raw_track, indices, accepted_duration)

        for index, label in zip(indices, labels):
            yield feature_fn(raw_track, index, audio_duration), label

def _build_base_dataset(track_infos : list[TrackInfo], input_type : InputTypes, audio_duration : int, accepted_duration : float, seed : int) -> tf.data.Dataset:
    shuffled_tracks = random.Random(seed).sample(track_infos, len(track_infos))
    rng = np.random.default_rng(seed)

    feature_shape = _as_shape(_INPUT_DIMS_FNS[input_type](audio_duration))

    output_signature = (
        tf.TensorSpec(shape=feature_shape, dtype=tf.float32),
        tf.TensorSpec(shape=(2, NOTE_BINS), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: _example_generator(shuffled_tracks, input_type, audio_duration, accepted_duration, rng),
        output_signature=output_signature,
    )

    return dataset.shuffle(SHUFFLE_BUFFER_SIZE, seed=seed)

def create_training_set(input_type : InputTypes, audio_duration : int, accepted_duration : float, batch_size : int, seed : int, n : int = -1) -> tf.data.Dataset:
    train_tracks, _ = split_tracks()

    dataset = _build_base_dataset(train_tracks, input_type, audio_duration, accepted_duration, seed)

    return dataset.take(n).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def create_test_set(input_type : InputTypes, audio_duration : int, accepted_duration : float, batch_size : int, seed : int, n : int = -1) -> tf.data.Dataset:
    _, test_tracks = split_tracks()

    dataset = _build_base_dataset(test_tracks, input_type, audio_duration, accepted_duration, seed)

    return dataset.take(n).batch(batch_size).prefetch(tf.data.AUTOTUNE)
