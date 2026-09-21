"""CLI to train and save one curriculum stage (ISOL or Chords) for a given
(input type, audio duration, accepted duration). Chords automatically loads
and continues from ISOL's saved checkpoint - run ISOL first.

Stage 3 (Full) isn't implemented yet - only ISOL and Chords are runnable here.
"""

import argparse

from src.constants import INPUT_DURATIONS, ACCEPTED_DURATIONS
from src.schema import CurriculumStage, InputTypes
from src.train.train import train_curriculum_stage
from src.models.load import save_trained_model, load_trained_model

DEFAULT_N_TRAIN = 20_000
DEFAULT_N_VAL = 10_000
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 500
DEFAULT_SEED = 42

_STAGE_ORDER = [CurriculumStage.ISOL, CurriculumStage.CHORDS]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and save one curriculum stage.")
    parser.add_argument(
        "-s", "--stage",
        required=True,
        choices=[s.value for s in _STAGE_ORDER],
        help="Curriculum stage to train.",
    )
    parser.add_argument(
        "-i", "--input-type",
        required=True,
        choices=[t.value for t in InputTypes],
        help="Input representation to train on.",
    )
    parser.add_argument(
        "-d", "--audio-duration",
        required=True,
        type=int,
        choices=INPUT_DURATIONS,
        help="Window duration in seconds.",
    )
    parser.add_argument(
        "-a", "--accepted-duration",
        required=True,
        type=float,
        choices=ACCEPTED_DURATIONS,
        help="Onset/offset acceptance radius in seconds.",
    )
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN, help=f"Examples per training epoch (default: {DEFAULT_N_TRAIN}).")
    parser.add_argument("--n-val", type=int, default=DEFAULT_N_VAL, help=f"Fixed validation set size, reused every epoch (default: {DEFAULT_N_VAL}).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size (default: {DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Training epochs (default: {DEFAULT_EPOCHS}).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    stage = CurriculumStage(args.stage)
    input_type = InputTypes(args.input_type)

    stage_index = _STAGE_ORDER.index(stage)
    initial_model = None
    if stage_index > 0:
        prior_stage = _STAGE_ORDER[stage_index - 1]
        print(f"Loading {prior_stage.value} checkpoint to continue from...")
        initial_model = load_trained_model(input_type, args.audio_duration, args.accepted_duration, stage=prior_stage)

    model = train_curriculum_stage(
        stage=stage,
        input_type=input_type,
        audio_duration=args.audio_duration,
        accepted_duration=args.accepted_duration,
        n_train=args.n_train,
        n_val=args.n_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        initial_model=initial_model,
    )

    path = save_trained_model(model, input_type, args.audio_duration, args.accepted_duration, stage=stage)
    print(f"Saved {stage.value} checkpoint to {path}")

if __name__ == "__main__":
    main()
