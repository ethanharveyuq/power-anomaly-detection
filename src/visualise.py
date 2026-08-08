"""
Load a training checkpoint and visualise training/validation progress.
"""
# --- Standard library ---
import argparse

# --- Third‑party ---
import torch
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import numpy as np


def load_checkpoint(path: str, device: str = "cpu"):
    """
    Loads the last training checkpoint which holds all the data and config etc
    Returns:
        checkpoint.pt after loading
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint

def print_run_config(config: dict) -> None:
    """
    Prints the run configuration so results can be replicated
    """
    print("Run configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

def build_dataframe(training_data: dict) -> pd.DataFrame:
    """
    From the data dictionary, creates the dataframe
    Returns:
        Pandas dataframe of the results with epoch as index
    """
    df = pd.DataFrame(training_data)
    df.index.name = "epoch"
    df.index += 1  # 1-indexed to match printed logs
    return df


def plot_training(df: pd.DataFrame, config: dict, save_path: str = None) -> None:
    """
    Creates a single IEEE‑style plot showing:
    - Training & validation accuracy
    - Training & validation loss
    all vs. epoch index.
    """

    plt.figure(figsize=(10, 6))

    epochs = df.index

    # Plot accuracy
    plt.plot(epochs, df["training accuracy"], label="Train Accuracy",
             linewidth=1.8, color="#1f77b4")
    plt.plot(epochs, df["validation accuracy"], label="Val Accuracy",
             linewidth=1.8, color="#ff7f0e")

    # Plot loss
    plt.plot(epochs, df["training loss"], label="Train Loss",
             linewidth=1.8, linestyle="--", color="#2ca02c")
    plt.plot(epochs, df["validation loss"], label="Val Loss",
             linewidth=1.8, linestyle="--", color="#d62728")

    # Axis labels
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)

    # Title (IEEE papers often avoid overly long titles)
    plt.title("Training and Validation Metrics vs Epoch", fontsize=13)

    # Legend (IEEE prefers outside or unobtrusive)
    plt.legend(loc="upper right", fontsize=10)

    # Grid (subtle)
    plt.grid(alpha=0.3)

    # Tight layout for LaTeX import
    plt.tight_layout()

    # Save as vector PDF for Overleaf
    if save_path:
        plt.savefig(save_path, format="pdf", dpi=600, bbox_inches="tight")
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