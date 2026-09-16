import numpy as np

from src.constants import DEFAULT_SAMPLE_RATE
from src.schema import AudioWaveform

def compute_window(audio : AudioWaveform, index : int, duration : int) -> AudioWaveform:
    boundary_indices = int(duration/2 * DEFAULT_SAMPLE_RATE)

    start = index - boundary_indices
    end = index + boundary_indices

    left_pad = max(0, -start)
    right_pad = max(0, end - len(audio))

    window = audio[max(start, 0):min(end, len(audio))]

    if left_pad or right_pad:
        window = np.pad(window, (left_pad, right_pad), mode="constant")

    return window