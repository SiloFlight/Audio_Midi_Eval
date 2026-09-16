import numpy as np
import tensorflow as tf

from src.audio_processing import compute_window,compute_cqt
from src.models._cnn import build_cnn_model

def compute_input_dims(audio_duration : int) -> tuple[int,int]:
    sample_arr = np.zeros(1, dtype=np.float32)

    sample_audio = compute_window(sample_arr,0,audio_duration)

    sample_cqt = compute_cqt(sample_audio)

    return sample_cqt.shape

def create_model(audio_duration : int) -> tf.keras.Model:
    freq_bins, time_frames = compute_input_dims(audio_duration)

    return build_cnn_model((freq_bins, time_frames, 1))