import mirdata,librosa,pretty_midi

from src.constants import DEFAULT_SAMPLE_RATE
from src.schema import TrackType,TrackInfo,RawTrack

def load_track_info(track_info : TrackInfo) -> RawTrack:
    wav_path = track_info.wav_path
    mid_path = track_info.mid_path
    
    raw_audio = librosa.load(wav_path,sr=DEFAULT_SAMPLE_RATE)[0]
    raw_midi = pretty_midi.PrettyMIDI(mid_path)
    
    return RawTrack(
        track_info=track_info,
        audio=raw_audio,
        midi=raw_midi)