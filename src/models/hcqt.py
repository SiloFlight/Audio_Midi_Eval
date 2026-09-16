import numpy as np
import tensorflow as tf

from src.audio_processing import compute_window,compute_hcqt
from src.models._cnn import build_cnn_model

def compute_input_dims(audio_duration : int) -> tuple[int,int,int]:
    sample_arr = np.zeros(1, dtype=np.float32)

    sample_audio = compute_window(sample_arr,0,audio_duration)

    sample_hcqt = compute_hcqt(sample_audio)

    return sample_hcqt.shape

def create_model(audio_duration : int) -> tf.keras.Model:
    input_shape = compute_input_dims(audio_duration)

    return build_cnn_model(input_shape)