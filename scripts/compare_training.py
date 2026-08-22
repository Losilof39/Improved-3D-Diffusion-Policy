#!/usr/bin/env python3
"""
Compare N iDP3 training runs side by side (e.g. different tasks or ablations).

Produces two figures in --output-dir:
  comparison_train_loss.png - per-epoch mean training loss, one line per run
  comparison_val_loss.png   - validation action MSE at checkpoints, one line per run

Usage:
  python scripts/compare_training.py \
      --runs data/outputs/g1_dex-3d-idp3-trial_60ep_bucket_img_joint_aug_seed0 \
             data/outputs/g1_dex-3d-idp3-trial_60ep_pick_and_place_seed0 \
      --labels Bucket "Pick & Place"

Run with the `idp3` conda env (has pandas/matplotlib/pyyaml):
  /home/luke/anaconda3/envs/idp3/bin/python scripts/compare_training.py ...
"""

import argparse
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_training import load_logs, load_hydra_config, compute_epoch_stats, get_validation_points

# dataviz reference palette, categorical slots 1-4 (blue/green/magenta/yellow) -
# the documented set that passes all-pairs CVD + normal-vision floors in both
# light and dark modes. Assign by position, never re-sorted by value.
COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def common_prefix_words(names: List[str]) -> str:
    """Longest common '_'-separated prefix shared by all names."""
    split = [n.split("_") for n in names]
    prefix = []
    for parts in zip(*split):
        if len(set(parts)) == 1:
            prefix.append(parts[0])
        else:
            break
    return "_".join(prefix) + "_" if prefix else ""


def default_labels(run_dirs: List[Path]) -> List[str]:
    names = [d.name for d in run_dirs]
    prefix = common_prefix_words(names)
    return [(n[len(prefix):] if prefix else n).replace("_", " ") for n in names]


def load_run(run_dir: Path, label: str):
    log_path = run_dir / "logs.json.txt"
    if not log_path.exists():
        raise FileNotFoundError(f"No logs.json.txt found in {run_dir}")
    df = load_logs(log_path)
    loss_col = "bc_loss" if "bc_loss" in df.columns else "train_loss"
    epoch_stats = compute_epoch_stats(df, loss_col)
    val_df = get_validation_points(df)
    return {
        "label": label,
        "epoch_stats": epoch_stats,
        "val_df": val_df,
        "loss_col": loss_col,
    }


def style_axes(ax):
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3, color=GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def make_train_loss_figure(runs, colors, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for run, color in zip(runs, colors):
        es = run["epoch_stats"]
        ax.plot(es["epoch"], es["mean"], color=color, linewidth=2, label=run["label"])
    style_axes(ax)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (log scale)")
    ax.set_title("Training loss: per-epoch mean, by run", color=INK)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


def make_val_loss_figure(runs, colors, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Annotating every checkpoint reads fine for 2 lines but collides once more
    # runs share the same checkpoints, so beyond 2 runs only label the final
    # point per run (markers still mark every checkpoint).
    annotate_all = len(runs) <= 2
    any_data = False
    for run, color in zip(runs, colors):
        val_df = run["val_df"]
        if val_df.empty:
            continue
        any_data = True
        ax.plot(val_df["epoch"], val_df["train_action_mse_error"], "o-",
                 color=color, linewidth=2, markersize=8, label=run["label"])
        rows_to_label = val_df.iterrows() if annotate_all else [(None, val_df.iloc[-1])]
        for _, row in rows_to_label:
            ax.annotate(f"{row['train_action_mse_error']:.4f}",
                        (row["epoch"], row["train_action_mse_error"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=8, color=MUTED)
    if not any_data:
        ax.text(0.5, 0.5, "no validation data", ha="center", va="center",
                 transform=ax.transAxes, color=MUTED)
    style_axes(ax)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation action MSE (log scale)")
    ax.set_title("Validation loss at checkpoints, by run", color=INK)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare N iDP3 training runs.")
    parser.add_argument("--runs", type=Path, nargs="+", required=True,
                        help="Run directories to compare (2 or more)")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Legend labels, one per --runs entry (default: derived from directory names)")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/comparison"),
                        help="Where to write the comparison figures")
    args = parser.parse_args()

    if len(args.runs) < 2:
        parser.error("--runs requires at least 2 directories")
    if len(args.runs) > len(COLORS):
        parser.error(f"--runs supports at most {len(COLORS)} directories (dataviz palette limit)")
    if args.labels is not None and len(args.labels) != len(args.runs):
        parser.error(f"--labels has {len(args.labels)} entries but --runs has {len(args.runs)}")

    run_dirs = [r.resolve() for r in args.runs]
    labels = args.labels or default_labels(run_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for run_dir, label in zip(run_dirs, labels):
        print(f"Loading {run_dir} ...")
        runs.append(load_run(run_dir, label))

    colors = COLORS[:len(runs)]

    train_loss_path = args.output_dir / "comparison_train_loss.png"
    make_train_loss_figure(runs, colors, train_loss_path)
    print(f"Saved {train_loss_path}")

    val_loss_path = args.output_dir / "comparison_val_loss.png"
    make_val_loss_figure(runs, colors, val_loss_path)
    print(f"Saved {val_loss_path}")


if __name__ == "__main__":
    main()
