from dataclasses import dataclass
from pathlib import Path
from warnings import warn

import librosa
import pretty_midi

from src.constants import DEFAULT_SAMPLE_RATE, MAPS_Folder
from src.schema import RawTrack, TrackInfo, TrackType

MAPS_MAX_DEPTH = 5  # RAND is the deepest playing style, with files 5 levels below MAPS_Folder


@dataclass
class TrackLocation:
    wav_path: Path
    mid_path: Path
    stem: str


def _load_leaf_descriptions(folder: Path) -> list[TrackLocation]:
    locations: list[TrackLocation] = []

    file_stems = {f.with_suffix("") for f in folder.iterdir()}

    for stem in file_stems:
        midi_path = stem.with_name(f"{stem.name}.mid")
        text_path = stem.with_name(f"{stem.name}.txt")
        wav_path = stem.with_name(f"{stem.name}.wav")

        if not midi_path.is_file():
            warn(f"{stem} is missing midi file.")
            continue

        if not text_path.is_file():
            warn(f"{stem} is missing text file.")
            continue

        if not wav_path.is_file():
            warn(f"{stem} is missing wav file.")
            continue

        locations.append(
            TrackLocation(
                wav_path=wav_path,
                mid_path=midi_path,
                stem=stem.name,
            )
        )

    return locations


def load_folder_descriptions(folder: Path, max_depth: int = MAPS_MAX_DEPTH, _depth: int = 0) -> list[TrackLocation]:
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    if _depth > max_depth:
        warn(f"{folder} exceeds max recursion depth of {max_depth}; skipping.")
        return []

    children = list(folder.iterdir())
    subdirs = [c for c in children if c.is_dir()]
    files = [c for c in children if not c.is_dir()]

    if subdirs and files:
        warn(f"{folder} contains a mix of files and subdirectories; recursing into subdirectories and ignoring stray files.")

    if subdirs:
        return [
            desc
            for child in subdirs
            for desc in load_folder_descriptions(child, max_depth=max_depth, _depth=_depth + 1)
        ]

    return _load_leaf_descriptions(folder)


def load_descriptions() -> list[TrackLocation]:
    return load_folder_descriptions(MAPS_Folder)


def load_track(track_location: TrackLocation) -> RawTrack:
    raw_audio, sample_rate = librosa.load(track_location.wav_path, sr=DEFAULT_SAMPLE_RATE, mono=True)
    duration = raw_audio.shape[0] / sample_rate

    track_info = TrackInfo(
        duration=duration,
        track_type=TrackType.MAPS,
        track_name=track_location.stem,
        track_desc="",
        wav_path=track_location.wav_path,
        mid_path=track_location.mid_path,
    )

    raw_midi = pretty_midi.PrettyMIDI(str(track_location.mid_path))

    return RawTrack(
        track_info=track_info,
        audio=raw_audio,
        midi=raw_midi,
    )


def track_generator():
    for track_location in load_descriptions():
        yield load_track(track_location)