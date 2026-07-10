import argparse
import re


def parse_args() -> argparse.Namespace:
    """
    Parses args passed to main and adds to namespace
    """
    parser = argparse.ArgumentParser(
        description="Train GPT4TS on PMU source identification"
    )

    # Data
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing PMU csv files")

    parser.add_argument("--window-length", type=int, default=500)
    parser.add_argument("--stride", type=int, default=250)

    parser.add_argument("--columns", nargs="+",
                        default=["FREQ"])

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)

    # GPT4TS
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Might use later
    parser.add_argument("--test-only", action="store_true")

    return parser.parse_args()


def create_config(args: argparse.Namespace) -> dict:
    """
    Creates dict from parser args
    """
    config = {
        "train data": args.data_dir,
        "train pattern": re.compile(r"^.*00\.csv$"),
        "validation data": args.data_dir,
        "validation pattern": re.compile(r"^.*01\.csv$"),
        "test data": args.data_dir,
        "test pattern": re.compile(r"^.*02\.csv$"),
        "window length": args.window_length,
        "stride": args.stride,
        "columns": args.columns,
        "batch size": args.batch_size,
        "epochs": args.epochs,
        "learning rate": args.learning_rate,
        "seed": args.seed,
        "gpu": args.gpu,
        "patch_size": args.patch_size,
        "d_model": args.d_model,
        "dropout": args.dropout,
        "test only": args.test_only
    }

    return config