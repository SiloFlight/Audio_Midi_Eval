import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.audio_processing import compute_window

def compute_input_dims(audio_duration : int) -> int:
    sample_arr = np.array([0])
            
    sample_audio = compute_window(sample_arr,0,audio_duration)
    
    return len(sample_audio)