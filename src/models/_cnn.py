"""Shared CNN backbone for the CQT/HCQT models (independent onset/offset heads).

Adapted from Basic Pitch's (github.com/spotify/basic-pitch) onset head. See
cqt.py/hcqt.py for the compute_input_dims that feeds this.
"""

import tensorflow as tf

tfkl = tf.keras.layers

EVENT_KERNEL_SIZE_1 = (5, 5)
EVENT_KERNEL_SIZE_2 = (3, 3)


def _initializer() -> tf.keras.initializers.VarianceScaling:
    return tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_avg", distribution="uniform", seed=None)


def _kernel_constraint() -> tf.keras.constraints.UnitNorm:
    return tf.keras.constraints.UnitNorm(axis=[0, 1, 2])


def _event_head(x : tf.Tensor, n_filters : int) -> tf.Tensor:
    x = tfkl.Conv2D(
        n_filters,
        EVENT_KERNEL_SIZE_1,
        padding="same",
        kernel_initializer=_initializer(),
        kernel_constraint=_kernel_constraint(),
    )(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.ReLU()(x)
    x = tfkl.Conv2D(
        1,
        EVENT_KERNEL_SIZE_2,
        padding="same",
        kernel_initializer=_initializer(),
        kernel_constraint=_kernel_constraint(),
    )(x)  # (freq, time, 1) logits, no activation

    freq_bins, time_frames = x.shape[1], x.shape[2]
    x = tfkl.Reshape((freq_bins, time_frames))(x)  # (freq, time)
    x = tfkl.Permute((2, 1))(x)                    # (time, freq)
    x = tfkl.GlobalAveragePooling1D()(x)           # (freq,) - average logit across the window's time frames

    return tfkl.Reshape((1, freq_bins))(x)  # (1, freq) - ready to concat into (2, freq)


def build_cnn_model(input_shape : tuple[int, ...], n_filters : int = 32) -> tf.keras.Model:
    """input_shape: (NOTE_BINS, T, channels)."""
    inputs = tf.keras.Input(shape=input_shape)

    onset_logits = _event_head(inputs, n_filters)
    offset_logits = _event_head(inputs, n_filters)

    outputs = tfkl.Concatenate(axis=1)([onset_logits, offset_logits])  # (2, NOTE_BINS)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
