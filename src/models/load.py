from pathlib import Path

import tensorflow as tf

from src.constants import Models_Folder
from src.schema import InputTypes

def get_model_path(input_type : InputTypes, audio_duration : int, accepted_duration : float) -> Path:
    filename = f"{input_type.value}_{audio_duration}_{accepted_duration}.keras"

    return Models_Folder / filename

def save_trained_model(model : tf.keras.Model, input_type : InputTypes, audio_duration : int, accepted_duration : float) -> Path:
    path = get_model_path(input_type, audio_duration, accepted_duration)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.save(path)

    return path

def load_trained_model(input_type : InputTypes, audio_duration : int, accepted_duration : float) -> tf.keras.Model:
    path = get_model_path(input_type, audio_duration, accepted_duration)

    return tf.keras.models.load_model(path)
