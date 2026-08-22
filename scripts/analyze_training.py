#!/usr/bin/env python3
"""
Analyze an iDP3 training run from its logs.json.txt, .hydra config, and train.log.

Produces:
  <run_dir>/analysis/dashboard.png  - loss curves, LR schedule, validation MSE, stability plots
  <run_dir>/analysis/summary.txt    - text report with convergence/stability/validation diagnostics

Usage:
  python scripts/analyze_training.py policies/aug1/g1_dex-3d-idp3-trial1_seed0
  python scripts/analyze_training.py policies/aug1/g1_dex-3d-idp3-trial1_seed0 --smooth-window 500

Run with the `idp3` conda env (has pandas/matplotlib/pyyaml):
  /home/luke/anaconda3/envs/idp3/bin/python scripts/analyze_training.py <run_dir>
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def load_logs(log_path: Path) -> pd.DataFrame:
    """Load JSONL training logs, skipping any trailing incomplete line."""
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame.from_records(records)


def load_hydra_config(run_dir: Path) -> dict:
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def get_param_count(run_dir: Path):
    train_log = run_dir / "train.log"
    if not train_log.exists():
        return None
    pattern = re.compile(r"number of parameters:\s*([\d.eE+\-]+)")
    for line in train_log.read_text().splitlines():
        m = pattern.search(line)
        if m:
            return float(m.group(1))
    return None


def get_run_duration(run_dir: Path, log_path: Path):
    """Estimate wall-clock duration from the wandb offline-run folder name (start)
    and the logs.json.txt mtime (end)."""
    wandb_dir = run_dir / "wandb"
    if not wandb_dir.exists():
        return None
    start_time = None
    for entry in wandb_dir.iterdir():
        m = re.match(r"offline-run-(\d{8}_\d{6})-", entry.name)
        if m:
            start_time = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
            break
    if start_time is None:
        return None
    end_time = datetime.fromtimestamp(log_path.stat().st_mtime)
    return end_time - start_time


def compute_epoch_stats(df: pd.DataFrame, loss_col: str) -> pd.DataFrame:
    return df.groupby("epoch")[loss_col].agg(["mean", "std", "min", "max", "count"]).reset_index()


def get_validation_points(df: pd.DataFrame) -> pd.DataFrame:
    if "test_mean_score" not in df.columns:
        return pd.DataFrame()
    val_df = df[df["test_mean_score"].notna()]
    return val_df[["epoch", "global_step", "train_action_mse_error", "test_mean_score"]]


def make_dashboard(df, epoch_stats, val_df, loss_col, smooth_window, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Raw loss vs step (log scale) + rolling mean
    ax = axes[0, 0]
    ax.plot(df["global_step"], df[loss_col], color="C0", alpha=0.15, linewidth=0.5, label="raw")
    smoothed = df[loss_col].rolling(smooth_window, min_periods=1).mean()
    ax.plot(df["global_step"], smoothed, color="C0", linewidth=1.5, label=f"rolling mean ({smooth_window} steps)")
    ax.set_yscale("log")
    ax.set_xlabel("global step")
    ax.set_ylabel(f"{loss_col} (log scale)")
    ax.set_title("Training loss vs step")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # 2. Per-epoch loss with min-max band
    ax = axes[0, 1]
    ax.fill_between(epoch_stats["epoch"], epoch_stats["min"], epoch_stats["max"],
                     color="C0", alpha=0.2, label="min-max")
    ax.plot(epoch_stats["epoch"], epoch_stats["mean"], color="C0", label="mean")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{loss_col} (log scale)")
    ax.set_title("Per-epoch loss (mean, min-max)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # 3. LR schedule
    ax = axes[0, 2]
    ax.plot(df["global_step"], df["lr"], color="C2")
    ax.set_xlabel("global step")
    ax.set_ylabel("learning rate")
    ax.set_title("LR schedule")
    ax.grid(True, alpha=0.3)

    # 4. Validation action MSE at checkpoints
    ax = axes[1, 0]
    if not val_df.empty:
        ax.plot(val_df["epoch"], val_df["train_action_mse_error"], "o-", color="C3")
        ax.set_yscale("log")
        for _, row in val_df.iterrows():
            ax.annotate(f"{row['train_action_mse_error']:.4f}",
                        (row["epoch"], row["train_action_mse_error"]),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no validation data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("epoch")
    ax.set_ylabel("action MSE (sum over batches)")
    ax.set_title("Validation action MSE")
    ax.grid(True, which="both", alpha=0.3)

    # 5. Per-epoch loss std (stability over training)
    ax = axes[1, 1]
    ax.plot(epoch_stats["epoch"], epoch_stats["std"], color="C1")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss std (log scale)")
    ax.set_title("Per-epoch loss variability")
    ax.grid(True, which="both", alpha=0.3)

    # 6. Loss distribution: first 10% vs last 10% of training
    ax = axes[1, 2]
    n = len(df)
    chunk = max(1, n // 10)
    first = df[loss_col].iloc[:chunk]
    last = df[loss_col].iloc[-chunk:]
    positive = df[loss_col][df[loss_col] > 0]
    bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 50)
    ax.hist(first, bins=bins, alpha=0.5, label="first 10%")
    ax.hist(last, bins=bins, alpha=0.5, label="last 10%")
    ax.set_xscale("log")
    ax.set_xlabel(loss_col)
    ax.set_ylabel("count")
    ax.set_title("Loss distribution: start vs end")
    ax.legend()

    fig.suptitle("iDP3 Training Dashboard", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def build_summary(df, epoch_stats, val_df, loss_col, hydra_cfg, param_count, duration):
    lines = []
    exp_name = hydra_cfg.get("exp_name", "unknown")
    lines.append("=" * 70)
    lines.append(f"iDP3 Training Summary: {exp_name}")
    lines.append("=" * 70)

    training_cfg = hydra_cfg.get("training", {}) or {}
    policy_cfg = hydra_cfg.get("policy", {}) or {}
    pc_cfg = policy_cfg.get("pointcloud_encoder_cfg", {}) or {}
    dataset_cfg = (hydra_cfg.get("task", {}) or {}).get("dataset", {}) or {}
    optimizer_cfg = hydra_cfg.get("optimizer", {}) or {}
    dataloader_cfg = hydra_cfg.get("dataloader", {}) or {}

    lines.append("\n-- Configuration --")
    lines.append(f"  Epochs configured: {training_cfg.get('num_epochs')}")
    lines.append(f"  Batch size: {dataloader_cfg.get('batch_size')}")
    lines.append(f"  LR: {optimizer_cfg.get('lr')}, scheduler: {training_cfg.get('lr_scheduler')}, "
                  f"warmup steps: {training_cfg.get('lr_warmup_steps')}")
    lines.append(f"  Horizon: {hydra_cfg.get('horizon')}, n_obs_steps: {hydra_cfg.get('n_obs_steps')}, "
                  f"n_action_steps: {hydra_cfg.get('n_action_steps')}")
    lines.append(f"  Point cloud: {pc_cfg.get('num_points')} points, "
                  f"use_pc_color: {policy_cfg.get('use_pc_color')}, "
                  f"pointnet_type: {policy_cfg.get('pointnet_type')}")
    if dataset_cfg.get("use_pc_augmentation"):
        lines.append(f"  Point cloud augmentation: jitter_std={dataset_cfg.get('pc_jitter_std')}, "
                      f"dropout_ratio={dataset_cfg.get('pc_dropout_ratio')}")
    lines.append(f"  Dataset val_ratio: {dataset_cfg.get('val_ratio')}")
    if param_count is not None:
        lines.append(f"  Model parameters: {param_count:,.0f}")

    lines.append("\n-- Run extent --")
    lines.append(f"  Total logged steps: {len(df)}")
    lines.append(f"  Epochs covered: {df['epoch'].min()} -> {df['epoch'].max()}")
    if duration is not None:
        total_seconds = duration.total_seconds()
        lines.append(f"  Wall-clock duration: {duration} ({total_seconds / 3600:.2f} h)")
        lines.append(f"  Throughput: {len(df) / total_seconds:.2f} steps/sec")

    lines.append(f"\n-- Loss convergence ({loss_col}) --")
    initial_loss = df[loss_col].iloc[0]
    final_loss = df[loss_col].iloc[-1]
    reduction = (initial_loss - final_loss) / initial_loss * 100
    lines.append(f"  Initial loss: {initial_loss:.6f}")
    lines.append(f"  Final loss:   {final_loss:.6f}")
    lines.append(f"  Reduction: {reduction:.2f}%")

    n_epochs = epoch_stats["epoch"].max() + 1
    chunk_epochs = max(1, int(n_epochs * 0.1))
    last_chunk = epoch_stats[epoch_stats["epoch"] >= n_epochs - chunk_epochs]["mean"]
    prev_chunk = epoch_stats[(epoch_stats["epoch"] >= n_epochs - 2 * chunk_epochs)
                              & (epoch_stats["epoch"] < n_epochs - chunk_epochs)]["mean"]
    if len(prev_chunk) > 0 and len(last_chunk) > 0:
        prev_mean = prev_chunk.mean()
        last_mean = last_chunk.mean()
        rel_change = (prev_mean - last_mean) / prev_mean * 100
        lines.append(f"  Mean loss, last {chunk_epochs} epochs: {last_mean:.6f}")
        lines.append(f"  Mean loss, previous {chunk_epochs} epochs: {prev_mean:.6f}")
        lines.append(f"  Relative improvement in final {chunk_epochs} epochs: {rel_change:.2f}%")
        if rel_change < 2:
            lines.append("  -> Loss appears to have PLATEAUED. More epochs likely won't help much.")
        else:
            lines.append("  -> Loss was STILL DECREASING at the end. Training longer may help further.")

    lines.append("\n-- Stability (within-epoch loss variance) --")
    first_epoch = epoch_stats.iloc[0]
    last_epoch = epoch_stats.iloc[-1]
    lines.append(f"  Epoch {int(first_epoch['epoch'])}: mean={first_epoch['mean']:.6f}, "
                  f"std={first_epoch['std']:.6f}, min={first_epoch['min']:.6f}, max={first_epoch['max']:.6f}")
    lines.append(f"  Epoch {int(last_epoch['epoch'])}: mean={last_epoch['mean']:.6f}, "
                  f"std={last_epoch['std']:.6f}, min={last_epoch['min']:.6f}, max={last_epoch['max']:.6f}")
    if first_epoch["std"] > 0:
        std_drop = (first_epoch["std"] - last_epoch["std"]) / first_epoch["std"] * 100
        lines.append(f"  Loss std shrank by {std_drop:.2f}% from first to last epoch "
                      f"(more consistent per-step predictions).")

    lines.append("\n-- Validation action MSE (computed on the train dataloader) --")
    if not val_df.empty:
        for _, row in val_df.iterrows():
            lines.append(f"  epoch {int(row['epoch']):4d}: action_mse_sum={row['train_action_mse_error']:.6f}")
        if len(val_df) > 1 and val_df["train_action_mse_error"].iloc[0] != 0:
            improvement = ((val_df["train_action_mse_error"].iloc[0] - val_df["train_action_mse_error"].iloc[-1])
                           / val_df["train_action_mse_error"].iloc[0] * 100)
            lines.append(f"  Improvement from first to last checkpoint: {improvement:.2f}%")
        lines.append("  NOTE: this metric re-runs predict_action() on the TRAINING dataloader")
        lines.append("  (see idp3_workspace.py), not a held-out validation split, even though")
        lines.append(f"  val_ratio={dataset_cfg.get('val_ratio')} is set in the dataset config. It mainly")
        lines.append("  reflects how well the policy fits the training distribution, not")
        lines.append("  generalization to unseen episodes.")
    else:
        lines.append("  No validation checkpoints found in logs.")

    lines.append("\n-- LR schedule --")
    lines.append(f"  Configured LR: {optimizer_cfg.get('lr')}")
    lines.append(f"  Max LR reached: {df['lr'].max():.2e}")
    lines.append(f"  Final LR: {df['lr'].iloc[-1]:.2e}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze an iDP3 training run's logs.")
    parser.add_argument("--run_dir", type=Path,
                        help="Path to the training output directory (containing logs.json.txt, .hydra/, train.log)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write analysis outputs (default: <run_dir>/analysis)")
    parser.add_argument("--smooth-window", type=int, default=None,
                        help="Rolling-mean window in steps for the loss curve (default: one epoch's worth of steps)")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    log_path = run_dir / "logs.json.txt"
    if not log_path.exists():
        raise FileNotFoundError(f"No logs.json.txt found in {run_dir}")

    output_dir = args.output_dir or (run_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {log_path} ...")
    df = load_logs(log_path)
    print(f"Loaded {len(df)} log entries spanning epochs {df['epoch'].min()}-{df['epoch'].max()}")

    loss_col = "bc_loss" if "bc_loss" in df.columns else "train_loss"

    hydra_cfg = load_hydra_config(run_dir)
    param_count = get_param_count(run_dir)
    duration = get_run_duration(run_dir, log_path)

    epoch_stats = compute_epoch_stats(df, loss_col)
    val_df = get_validation_points(df)

    smooth_window = args.smooth_window
    if smooth_window is None:
        smooth_window = max(1, int(epoch_stats["count"].iloc[0]))

    dashboard_path = output_dir / "dashboard.png"
    make_dashboard(df, epoch_stats, val_df, loss_col, smooth_window, dashboard_path)
    print(f"Saved dashboard to {dashboard_path}")

    summary = build_summary(df, epoch_stats, val_df, loss_col, hydra_cfg, param_count, duration)
    print("\n" + summary)
    summary_path = output_dir / "summary.txt"
    summary_path.write_text(summary + "\n")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
