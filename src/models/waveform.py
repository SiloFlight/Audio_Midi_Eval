import numpy as np
import tensorflow as tf

from src.audio_processing import compute_window
from src.constants import NOTE_BINS

tfkl = tf.keras.layers

DENSE_UNITS = [256, 256, 128]
DENSE_DROPOUT_RATE = 0.3

def compute_input_dims(audio_duration : int) -> int:
    sample_arr = np.zeros(1, dtype=np.float32)

    sample_audio = compute_window(sample_arr,0,audio_duration)

    return len(sample_audio)

def create_model(audio_duration : int) -> tf.keras.Model:
    input_dim = compute_input_dims(audio_duration)

    inputs = tf.keras.Input(shape=(input_dim,))

    x = inputs
    for units in DENSE_UNITS:
        x = tfkl.Dense(units, activation="relu")(x)
        x = tfkl.Dropout(DENSE_DROPOUT_RATE)(x)

    x = tfkl.Dense(2 * NOTE_BINS)(x)
    outputs = tfkl.Reshape((2, NOTE_BINS))(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)