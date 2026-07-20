"""
Load a training checkpoint and visualise training/validation progress.
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def load_checkpoint(path: str, device: str = "cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint


def print_run_config(config: dict):
    print("Run configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")


def build_dataframe(training_data: dict) -> pd.DataFrame:
    df = pd.DataFrame(training_data)
    df.index.name = "epoch"
    df.index += 1  # 1-indexed to match printed logs
    return df


def plot_training(df: pd.DataFrame, config: dict, save_path: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Accuracy
    axes[0, 0].plot(df.index, df["training accuracy"], label="Train")
    axes[0, 0].plot(df.index, df["validation accuracy"], label="Validation")
    axes[0, 0].set_title("Accuracy")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()

    # Loss
    axes[0, 1].plot(df.index, df["training loss"], label="Train")
    axes[0, 1].plot(df.index, df["validation loss"], label="Validation")
    axes[0, 1].set_title("Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()

    # F1
    axes[1, 0].plot(df.index, df["validation f1"], color="green")
    axes[1, 0].set_title("Validation F1")
    axes[1, 0].set_xlabel("Epoch")

    # Time per epoch
    axes[1, 1].plot(df.index, df["time elapsed"], color="orange")
    axes[1, 1].set_title("Validation Time (s)")
    axes[1, 1].set_xlabel("Epoch")

    # Title with a few key run params, so the plot is self-describing
    fig.suptitle(
        f"columns={config.get('columns')} | "
        f"window={config.get('window length')} stride={config.get('stride')} | "
        f"head_lr={config.get('head learning rate')} backbone_lr={config.get('backbone learning rate')}"
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--save-fig", default=None, help="Path to save plot instead of showing it")
    parser.add_argument("--save-data", default=None, help="Path to save data instead of showing it")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    training_data = checkpoint["training_data"]

    print_run_config(config)
    print(f"Best F1: {checkpoint['best_f1']:.4f} (at or before epoch {checkpoint['epoch'] + 1})")

    df = build_dataframe(training_data)
    plot_training(df, config, save_path=args.save_fig)
    if args.save_data:
        df.to_csv(args.save_data, index=False)
        print(f"Saved data to {args.save_data}")