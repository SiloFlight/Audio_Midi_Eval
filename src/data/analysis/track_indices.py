import csv
from pathlib import Path
from typing import Iterator

from src.constants import DEFAULT_SAMPLE_RATE, Index_Folder
from src.schema import TrackType,TrackInfo,RawTrack
from src.data.maestro import track_generator as maestro_track_generator
from src.data.maps import track_generator as maps_track_generator

TRACK_INFO_FIELDS = ["duration", "track_type", "track_name", "track_desc", "wav_path", "mid_path"]

def store_track_info(track_gen: Iterator[RawTrack]) -> list[TrackInfo]:
    return [raw_track.track_info for raw_track in track_gen]

def create_track_index(track_infos : list[TrackInfo], path : Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRACK_INFO_FIELDS)
        writer.writeheader()
        for track_info in track_infos:
            writer.writerow({
                "duration": track_info.duration,
                "track_type": track_info.track_type.value,
                "track_name": track_info.track_name,
                "track_desc": track_info.track_desc,
                "wav_path": track_info.wav_path,
                "mid_path": track_info.mid_path,
            })

def load_track_index(path : Path) -> list[TrackInfo]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [
            TrackInfo(
                duration=float(row["duration"]),
                track_type=TrackType(row["track_type"]),
                track_name=row["track_name"],
                track_desc=row["track_desc"],
                wav_path=Path(row["wav_path"]),
                mid_path=Path(row["mid_path"]),
            )
            for row in reader
        ]

def compute_maestro_track_index() -> list[TrackInfo]:
    track_infos = store_track_info(maestro_track_generator())
    create_track_index(track_infos, Index_Folder / "MAESTRO_index.csv")
    return track_infos

def get_maestro_track_index() -> list[TrackInfo]:
    path = Index_Folder / "MAESTRO_index.csv"
    if path.exists():
        return load_track_index(path)
    return compute_maestro_track_index()

def compute_maps_track_index() -> list[TrackInfo]:
    track_infos = store_track_info(maps_track_generator())
    create_track_index(track_infos, Index_Folder / "MAPS_index.csv")
    return track_infos

def get_maps_track_index() -> list[TrackInfo]:
    path = Index_Folder / "MAPS_index.csv"
    if path.exists():
        return load_track_index(path)
    return compute_maps_track_index()