import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf

from src.audio_processing import compute_cqt, compute_hcqt
from src.constants import (
    MAX_EXAMPLES_PER_TRACK, MAX_TRACK_DURATION, NOTE_BINS, SHUFFLE_BUFFER_SIZE, NEGATIVE_PERCENTAGE,
    CURRICULUM_SPLIT_SEED, CURRICULUM_VAL_TRACKS, CURRICULUM_NEGATIVE_PERCENTAGE, TRACK_LOAD_WORKERS,
    CURRICULUM_MAX_TRACK_DURATION, CURRICULUM_MAX_EXAMPLES_PER_TRACK,
)
from src.schema import CurriculumStage, InputTypes, TrackInfo
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

_TRACK_CONTEXT_FNS = {
    InputTypes.Waveform: lambda raw_track, input_dims: raw_track,
    InputTypes.CQT: lambda raw_track, input_dims: (compute_cqt(raw_track.audio), input_dims[1]),
    InputTypes.HCQT: lambda raw_track, input_dims: (compute_hcqt(raw_track.audio), input_dims[1]),
}

def _as_shape(dims) -> tuple[int, ...]:
    return (dims,) if isinstance(dims, int) else tuple(dims)

def split_tracks() -> tuple[list[TrackInfo], list[TrackInfo], list[TrackInfo]]:
    maestro_tracks = get_maestro_track_index()
    maps_tracks = get_maps_track_index()

    train_tracks = [t for t in maestro_tracks if t.track_desc == "train"] + maps_tracks
    val_tracks = [t for t in maestro_tracks if t.track_desc == "validation"]
    test_tracks = [t for t in maestro_tracks if t.track_desc == "test"]

    return train_tracks, val_tracks, test_tracks

def _load_track_examples(track_info : TrackInfo, track_seed : int, input_type : InputTypes, audio_duration : int, accepted_duration : float, input_dims, negative_percentage : float, max_examples_per_track : int) -> list:
    feature_fn = _FEATURE_FNS[input_type]
    context_fn = _TRACK_CONTEXT_FNS[input_type]

    rng = np.random.default_rng([track_seed, hash(track_info.track_name) & 0xFFFFFFFF])

    raw_track = load_track_info(track_info)
    context = context_fn(raw_track, input_dims)

    indices = get_potential_indices(raw_track, accepted_duration, rng=rng, max_examples=max_examples_per_track, negative_percentage=negative_percentage)
    labels = create_labels(raw_track, indices, accepted_duration)

    return [(feature_fn(context, index, audio_duration), label) for index, label in zip(indices, labels)]

def _example_generator(track_infos : list[TrackInfo], input_type : InputTypes, audio_duration : int, accepted_duration : float, input_dims, seed : int, negative_percentage : float, max_examples_per_track : int):
    # ThreadPoolExecutor.map submits every track's work immediately, even
    # though only TRACK_LOAD_WORKERS run at a time - the rest just queue up.
    # tf.data routinely closes this generator early (GeneratorExit, e.g. once
    # the shuffle buffer has enough), and a plain `with ThreadPoolExecutor()`
    # block's default cleanup (shutdown(wait=True)) would then block until
    # every already-queued track finishes, even though none of it is wanted
    # anymore. cancel_futures drops the unstarted backlog instead of draining
    # it; only the handful of tasks already running have to finish.
    pool = ThreadPoolExecutor(max_workers=TRACK_LOAD_WORKERS)
    try:
        futures = pool.map(
            lambda track_info: _load_track_examples(track_info, seed, input_type, audio_duration, accepted_duration, input_dims, negative_percentage, max_examples_per_track),
            track_infos,
        )
        for examples in futures:
            yield from examples
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

def _build_base_dataset(track_infos : list[TrackInfo], input_type : InputTypes, audio_duration : int, accepted_duration : float, seed : int, negative_percentage : float = NEGATIVE_PERCENTAGE, max_track_duration : float = MAX_TRACK_DURATION, max_examples_per_track : int = MAX_EXAMPLES_PER_TRACK) -> tf.data.Dataset:
    # Filtered on track_info.duration (already known from the index) rather than
    # inside _example_generator, so excluded tracks are never loaded from disk at all.
    track_infos = [t for t in track_infos if t.duration <= max_track_duration]
    shuffled_tracks = random.Random(seed).sample(track_infos, len(track_infos))

    input_dims = _INPUT_DIMS_FNS[input_type](audio_duration)
    feature_shape = _as_shape(input_dims)

    output_signature = (
        tf.TensorSpec(shape=feature_shape, dtype=tf.float32),
        tf.TensorSpec(shape=(2, NOTE_BINS), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: _example_generator(shuffled_tracks, input_type, audio_duration, accepted_duration, input_dims, seed, negative_percentage, max_examples_per_track),
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

def _maps_subset(track_name : str) -> str:
    parts = track_name.split("_")
    # MUS filenames embed the piece name onto the subset code with a hyphen
    # (e.g. "MUS-schub", "MUS-bk") rather than as a separate "_"-delimited
    # field like ISOL/RAND/UCHO - normalize those back to a plain "MUS".
    return "MUS" if parts[1].startswith("MUS") else parts[1]

def split_stage_tracks(stage : CurriculumStage) -> tuple[list[TrackInfo], list[TrackInfo]]:
    """Returns (train_tracks, val_tracks) for one curriculum stage. ISOL/Chords
    are split via a fixed random split (CURRICULUM_SPLIT_SEED, independent of
    the caller's training seed, so the split itself never moves). Full instead
    reuses MAESTRO's own native train/validation split, with MAPS MUS tracks
    folded entirely into train - only ~270 tracks, too few to usefully carve a
    separate held-out slice from on top of MAESTRO's own validation set."""
    if stage == CurriculumStage.FULL:
        maestro_tracks = get_maestro_track_index()
        mus_tracks = [t for t in get_maps_track_index() if _maps_subset(t.track_name) == "MUS"]

        train_tracks = [t for t in maestro_tracks if t.track_desc == "train"] + mus_tracks
        val_tracks = [t for t in maestro_tracks if t.track_desc == "validation"]

        return train_tracks, val_tracks

    if stage == CurriculumStage.ISOL:
        pool = [t for t in get_maps_track_index() if _maps_subset(t.track_name) == "ISOL"]
    elif stage == CurriculumStage.CHORDS:
        pool = [t for t in get_maps_track_index() if _maps_subset(t.track_name) in ("RAND", "UCHO")]
    else:
        raise ValueError(f"Unknown stage: {stage}")

    shuffled = random.Random(CURRICULUM_SPLIT_SEED).sample(pool, len(pool))
    val_count = CURRICULUM_VAL_TRACKS[stage.value]

    return shuffled[val_count:], shuffled[:val_count]

def _curriculum_overrides(stage : CurriculumStage) -> dict:
    return {
        "negative_percentage": CURRICULUM_NEGATIVE_PERCENTAGE.get(stage.value, NEGATIVE_PERCENTAGE),
        "max_track_duration": CURRICULUM_MAX_TRACK_DURATION.get(stage.value, MAX_TRACK_DURATION),
        "max_examples_per_track": CURRICULUM_MAX_EXAMPLES_PER_TRACK.get(stage.value, MAX_EXAMPLES_PER_TRACK),
    }

def create_curriculum_training_set(stage : CurriculumStage, input_type : InputTypes, audio_duration : int, accepted_duration : float, batch_size : int, seed : int, n : int = -1) -> tf.data.Dataset:
    train_tracks, _ = split_stage_tracks(stage)

    dataset = _build_base_dataset(train_tracks, input_type, audio_duration, accepted_duration, seed, **_curriculum_overrides(stage))

    return dataset.take(n).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def create_curriculum_validation_set(stage : CurriculumStage, input_type : InputTypes, audio_duration : int, accepted_duration : float, batch_size : int, seed : int, n : int) -> tf.data.Dataset:
    _, val_tracks = split_stage_tracks(stage)

    dataset = _build_base_dataset(val_tracks, input_type, audio_duration, accepted_duration, seed, **_curriculum_overrides(stage))

    return dataset.take(n).cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
