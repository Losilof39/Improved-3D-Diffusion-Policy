#!/usr/bin/env python3
"""
Plot reference (commanded) vs actual (measured) joint pose for a deployment rollout.

Reads episode_log.npz (written by deploy_policy.sh): `agent_pos` (T, 26) is the
measured robot state at each control step, `actions` (T, H, 26) is the policy's
predicted action chunk at each step. The "reference" trajectory is the first
predicted action of each chunk (actions[:, 0, :]) -- the command actually sent
to the controller at that step -- plotted against the `agent_pos` actually
achieved, per joint group.

Produces, in <output-dir>:
  pose_tracking_<group>.png  - one figure per joint group, one subplot per DOF
  summary.txt                - per-DOF tracking RMSE

Usage:
  python scripts/plot_pose_tracking.py --episode-log policies/logs/vanilla/medium/20260622_074313/episode_log.npz

Run with the idp3 conda env:
  conda run -n idp3 python scripts/plot_pose_tracking.py --episode-log <path-to-episode_log.npz>
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_ood import JOINT_GROUPS, load_eval_data


def plot_group(group_name, sl, reference, actual, out_path):
    n_dof = sl.stop - sl.start
    n_cols = min(n_dof, 4)
    n_rows = -(-n_dof // n_cols)  # ceil div
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    t = np.arange(len(actual))

    for i in range(n_dof):
        ax = axes[i // n_cols][i % n_cols]
        dof = sl.start + i
        ax.plot(t, reference[:, i], "--", color="C1", label="reference", linewidth=1.5)
        ax.plot(t, actual[:, i], "-", color="C0", label="actual", linewidth=1.5)
        ax.set_title(f"dof {dof}")
        ax.set_xlabel("timestep")
        if i == 0:
            ax.legend(loc="best", fontsize=8)

    for i in range(n_dof, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].axis("off")

    fig.suptitle(f"{group_name}: reference vs actual")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def rmse_report(reference, actual):
    lines = ["-- per-joint-group tracking RMSE (reference vs actual) --"]
    for group, sl in JOINT_GROUPS.items():
        per_dof_rmse = np.sqrt(((reference[:, sl] - actual[:, sl]) ** 2).mean(axis=0))
        lines.append(f"  {group:10s} dims {sl.start:2d}:{sl.stop:2d}: "
                      f"mean={per_dof_rmse.mean():.4f}  per-dof=[{', '.join(f'{v:.4f}' for v in per_dof_rmse)}]")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Plot reference vs actual joint pose for a deployment rollout.")
    parser.add_argument("--episode-log", type=Path, required=True,
                         help="Path to episode_log.npz written during deployment")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Where to write plots/summary (default: <episode-log dir>/pose_tracking)")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.episode_log.parent / "pose_tracking")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading eval rollout from {args.episode_log} ...")
    actual, reference = load_eval_data(args.episode_log)
    print(f"  agent_pos (actual): {actual.shape}, executed action (reference): {reference.shape}")

    for group_name, sl in JOINT_GROUPS.items():
        plot_path = output_dir / f"pose_tracking_{group_name}.png"
        plot_group(group_name, sl, reference, actual, plot_path)
        print(f"Saved {plot_path}")

    summary = rmse_report(reference, actual)
    print("\n" + summary)
    (output_dir / "summary.txt").write_text(summary + "\n")
    print(f"\nSaved summary to {output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
