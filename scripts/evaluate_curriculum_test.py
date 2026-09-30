import argparse

from src.constants import INPUT_DURATIONS, ACCEPTED_DURATIONS
from src.schema import CurriculumStage, InputTypes
from src.train.train import evaluate_curriculum_test
from src.models.load import load_trained_model

DEFAULT_N_TEST = 10_000
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 42

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a curriculum checkpoint against MAESTRO's held-out test split.")
    parser.add_argument(
        "-s", "--stage",
        required=True,
        choices=[CurriculumStage.FULL.value],
        help="Checkpoint stage to load. Only Full has a genuine held-out test split today.",
    )
    parser.add_argument(
        "-i", "--input-type",
        required=True,
        choices=[t.value for t in InputTypes],
        help="Input representation the checkpoint was trained on.",
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
    parser.add_argument("--n-test", type=int, default=DEFAULT_N_TEST, help=f"Test set size (default: {DEFAULT_N_TEST}).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size (default: {DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    stage = CurriculumStage(args.stage)
    input_type = InputTypes(args.input_type)

    model = load_trained_model(input_type, args.audio_duration, args.accepted_duration, stage=stage)

    results = evaluate_curriculum_test(
        model,
        input_type=input_type,
        audio_duration=args.audio_duration,
        accepted_duration=args.accepted_duration,
        n_test=args.n_test,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print(f"\n--- {stage.value} checkpoint evaluated against MAESTRO's held-out test split ---")
    for metric_name, value in results.items():
        print(f"  {metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()
