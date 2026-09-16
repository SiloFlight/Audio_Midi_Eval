"""CLI to build the MAESTRO and/or MAPS track indices as CSV files."""

import argparse

from src.constants import Index_Folder
from src.data.analysis.track_indices import (
    compute_maestro_track_index,
    compute_maps_track_index,
)

INDEX_BUILDERS = {
    "MAESTRO": (compute_maestro_track_index, Index_Folder / "MAESTRO_index.csv"),
    "MAPS": (compute_maps_track_index, Index_Folder / "MAPS_index.csv"),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build track indices for MAESTRO and/or MAPS.")
    parser.add_argument(
        "-i", "--index",
        choices=sorted(INDEX_BUILDERS),
        help="Which index to build. Omit to build both.",
    )
    parser.add_argument(
        "-o", "--overwrite",
        action="store_true",
        help="Overwrite an index file that already exists instead of skipping it.",
    )
    return parser.parse_args()

def build_index(name: str, overwrite: bool) -> None:
    compute_fn, path = INDEX_BUILDERS[name]

    if path.exists() and not overwrite:
        print(f"{name} index already exists at {path}, skipping (use -o to overwrite).")
        return

    print(f"Building {name} index...")
    compute_fn()
    print(f"Wrote {name} index to {path}.")

def main() -> None:
    args = parse_args()
    names = [args.index] if args.index else list(INDEX_BUILDERS)

    for name in names:
        build_index(name, args.overwrite)

if __name__ == "__main__":
    main()
