import tensorflow as tf

from src.constants import NOTE_BINS
from src.schema import InputTypes, TrackInfo
from src.data.misc import load_track_info
from src.data.analysis.track_indices import get_maestro_track_index, get_maps_track_index
from src.dataset_creation.index_detection import get_potential_indices
from src.dataset_creation.examples import create_waveform_example, create_cqt_example, create_hcqt_example
from src.models import waveform, cqt, hcqt

_EXAMPLE_FNS = {
    InputTypes.Waveform: create_waveform_example,
    InputTypes.CQT: create_cqt_example,
    InputTypes.HCQT: create_hcqt_example,
}

_INPUT_DIMS_FNS = {
    InputTypes.Waveform: waveform.compute_input_dims,
    InputTypes.CQT: cqt.compute_input_dims,
    InputTypes.HCQT: hcqt.compute_input_dims,
}

def _as_shape(dims) -> tuple[int, ...]:
    return (dims,) if isinstance(dims, int) else tuple(dims)

def split_tracks() -> tuple[list[TrackInfo], list[TrackInfo]]:
    # MAESTRO's track_desc holds mirdata's train/validation/test split. MAPS has no
    # split info, so all of it goes to training; only MAESTRO's "test" split is held out.
    maestro_tracks = get_maestro_track_index()
    maps_tracks = get_maps_track_index()

    test_tracks = [t for t in maestro_tracks if t.track_desc == "test"]
    train_tracks = [t for t in maestro_tracks if t.track_desc != "test"] + maps_tracks

    return train_tracks, test_tracks

def _example_generator(track_infos : list[TrackInfo], input_type : InputTypes, audio_duration : int, accepted_duration : float):
    example_fn = _EXAMPLE_FNS[input_type]

    for track_info in track_infos:
        raw_track = load_track_info(track_info)

        for index in get_potential_indices(raw_track, accepted_duration):
            yield example_fn(raw_track, index, audio_duration, accepted_duration)

def _build_dataset(track_infos : list[TrackInfo], input_type : InputTypes, audio_duration : int, accepted_duration : float) -> tf.data.Dataset:
    feature_shape = _as_shape(_INPUT_DIMS_FNS[input_type](audio_duration))

    output_signature = (
        tf.TensorSpec(shape=feature_shape, dtype=tf.float32),
        tf.TensorSpec(shape=(2, NOTE_BINS), dtype=tf.float32),
    )

    return tf.data.Dataset.from_generator(
        lambda: _example_generator(track_infos, input_type, audio_duration, accepted_duration),
        output_signature=output_signature,
    )

def create_training_set(input_type : InputTypes, audio_duration : int, accepted_duration : float) -> tf.data.Dataset:
    train_tracks, _ = split_tracks()

    return _build_dataset(train_tracks, input_type, audio_duration, accepted_duration)

def create_test_set(input_type : InputTypes, audio_duration : int, accepted_duration : float) -> tf.data.Dataset:
    _, test_tracks = split_tracks()

    return _build_dataset(test_tracks, input_type, audio_duration, accepted_duration)
