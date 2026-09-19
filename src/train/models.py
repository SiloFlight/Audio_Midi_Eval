"""Module Script To Handle the Generation of TF Model Objects for Training"""

import tensorflow as tf

from src.schema import InputTypes
from src.models import cqt, hcqt, waveform

_CREATE_MODEL_FNS = {
    InputTypes.Waveform: waveform.create_model,
    InputTypes.CQT: cqt.create_model,
    InputTypes.HCQT: hcqt.create_model,
}

def load_model(input_type : InputTypes, audio_duration : int, seed : int | None = None) -> tf.keras.Model:
    return _CREATE_MODEL_FNS[input_type](audio_duration, seed=seed)
