from pathlib import Path

import tensorflow as tf

from src.constants import Models_Folder
from src.schema import CurriculumStage, InputTypes

def get_model_path(input_type : InputTypes, audio_duration : int, accepted_duration : float, stage : CurriculumStage | None = None) -> Path:
    filename = f"{input_type.value}_{audio_duration}_{accepted_duration}"
    if stage is not None:
        filename += f"_stage_{stage.value}"
    filename += ".keras"

    return Models_Folder / filename

def save_trained_model(model : tf.keras.Model, input_type : InputTypes, audio_duration : int, accepted_duration : float, stage : CurriculumStage | None = None) -> Path:
    path = get_model_path(input_type, audio_duration, accepted_duration, stage=stage)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.save(path)

    return path

def load_trained_model(input_type : InputTypes, audio_duration : int, accepted_duration : float, stage : CurriculumStage | None = None) -> tf.keras.Model:
    path = get_model_path(input_type, audio_duration, accepted_duration, stage=stage)

    return tf.keras.models.load_model(path)
