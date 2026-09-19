"""CLI to train, evaluate, and save one model for a given (input type, audio duration, accepted duration)."""

import argparse

from src.constants import INPUT_DURATIONS, ACCEPTED_DURATIONS
from src.schema import InputTypes
from src.train.train import train, evaluate
from src.models.load import save_trained_model

DEFAULT_N_TRAIN = 100_000  # examples per epoch, not a total dataset cap - see train()
DEFAULT_N_TEST = -1          # -1 = the full test set
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_SEED = 42

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, evaluate, and save one model.")
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
    parser.add_argument("--n-test", type=int, default=DEFAULT_N_TEST, help="Test examples to evaluate on; -1 for the full test set (default: -1).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Batch size (default: {DEFAULT_BATCH_SIZE}).")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Training epochs (default: {DEFAULT_EPOCHS}).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    input_type = InputTypes(args.input_type)

    model = train(
        input_type=input_type,
        audio_duration=args.audio_duration,
        accepted_duration=args.accepted_duration,
        n_train=args.n_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )

    path = save_trained_model(model, input_type, args.audio_duration, args.accepted_duration)
    print(f"Saved model to {path}")

    results = evaluate(
        model,
        input_type=input_type,
        audio_duration=args.audio_duration,
        accepted_duration=args.accepted_duration,
        n_test=args.n_test,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print("Evaluation results:")
    for name, value in results.items():
        print(f"  {name}: {value:.4f}")

if __name__ == "__main__":
    main()
