import mirdata,librosa,pretty_midi

from src.constants import DEFAULT_SAMPLE_RATE,MAESTRO_Folder
from src.schema import TrackType,TrackInfo,RawTrack

def track_generator():
    dataset = mirdata.initialize("maestro", data_home=MAESTRO_Folder)
    track_ids = dataset.track_ids
    
    for track_id in track_ids:
        track = dataset.track(track_id)
        
        audio_path = track.audio_path
        mid_path = track.midi_path
        
        track_info = TrackInfo(
            duration=track.duration,
            track_type=TrackType.MAESTRO,
            track_name=track.track_id,
            track_desc=track.split,
            wav_path=audio_path,
            mid_path=mid_path
        )
        
        raw_audio = librosa.load(audio_path,sr=DEFAULT_SAMPLE_RATE)[0]
        raw_midi = pretty_midi.PrettyMIDI(mid_path)
        
        
        yield RawTrack(
            track_info=track_info,
            audio=raw_audio,
            midi=raw_midi
        )