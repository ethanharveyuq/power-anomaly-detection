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
    parser.add_argument("--epochs-per-run", type=int, default=20)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume training after a checkpoint")

    # Smaller experiment set?
    parser.add_argument("--experiment", action="store_true")


    # GPT4TS
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patch-stride", type=int, default=10)

    # Might use later
    parser.add_argument("--test-only", action="store_true")

    return parser.parse_args()


def create_config(args: argparse.Namespace) -> dict:
    """
    Creates dict from parser args
    """

    if args.experiment:
        # Smaller subset
        pmus = ["Bd18850", "Bg088", "Bg105", "Bg108", "Bg142"]
        pmu_pattern = "|".join(map(re.escape, pmus))
        train_pattern = re.compile(rf"^({pmu_pattern}).*00\.csv$")
        validation_pattern = re.compile(rf"^({pmu_pattern}).*01\.csv$")
        test_pattern = re.compile(rf"^({pmu_pattern}).*02\.csv$")
    else:
        train_pattern = re.compile(r"^.*00\.csv$")
        validation_pattern = re.compile(r"^.*01\.csv$")
        test_pattern = re.compile(r"^.*02\.csv$")

    

    config = {
        "train data": args.data_dir,
        "train pattern": train_pattern,
        "validation data": args.data_dir,
        "validation pattern": validation_pattern,
        "test data": args.data_dir,
        "test pattern": test_pattern,
        "window length": args.window_length,
        "stride": args.stride,
        "columns": args.columns,
        "batch size": args.batch_size,
        "epochs per run": args.epochs_per_run,
        "learning rate": args.learning_rate,
        "seed": args.seed,
        "gpu": args.gpu,
        "patch_size": args.patch_size,
        "d_model": args.d_model,
        "dropout": args.dropout,
        "test only": args.test_only,
        "resume": args.resume,
        "patch stride": args.patch_stride,
        "experiment": args.experiment,
        "head learning rate": args.head_learning_rate,
        "backbone learning rate": args.backbone_learning_rate
    }

    return config