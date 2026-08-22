#!/usr/bin/env python3
"""
PCA / Mahalanobis out-of-distribution (OOD) check for an iDP3 deployment rollout.

Compares the joint-state (agent_pos) and action distributions observed during a
real deployment rollout (episode_log.npz, written by deploy_policy.sh) against the
training distribution stored in the zarr dataset the checkpoint was trained on.

Produces:
  <output-dir>/pca_agent_pos.png - training state distribution (hexbin) + eval trajectory in PC space
  <output-dir>/pca_action.png    - same for the policy's executed actions
  <output-dir>/summary.txt       - Mahalanobis-distance OOD scores + per-joint-group range checks

Usage:
  python scripts/analyze_ood.py --episode-log policies/eval_logs/20260611_113404/episode_log.npz

Run with the idp3 conda env:
  conda run -n idp3 python scripts/analyze_ood.py --episode-log <path-to-episode_log.npz>
"""

import argparse
from pathlib import Path

import numpy as np
import zarr

from ood_common import fit_pca, mahalanobis_report, plot_pca

# Unitree G1 + Inspire/BrainCo hands: 26 DOF, see config/task/g1_dex-3d.yaml
JOINT_GROUPS = {
    "left_arm": slice(0, 7),
    "right_arm": slice(7, 14),
    "left_hand": slice(14, 20),
    "right_hand": slice(20, 26),
}


def load_training_data(zarr_path):
    z = zarr.open(str(zarr_path), 'r')
    return np.asarray(z['data/state'][:]), np.asarray(z['data/action'][:])


def load_eval_data(episode_log_path):
    d = np.load(str(episode_log_path))
    agent_pos = d['agent_pos']           # (T, 26)
    actions = d['actions']                # (T, H, 26)
    return agent_pos, actions[:, 0, :]    # first predicted = executed action


def range_check_report(train_data, eval_data, name):
    lines = [f"-- {name}: per-joint-group range check (eval vs training [min, max]) --"]
    train_min = train_data.min(axis=0)
    train_max = train_data.max(axis=0)
    for group, sl in JOINT_GROUPS.items():
        below = (train_min[sl] - eval_data[:, sl]).clip(min=0)
        above = (eval_data[:, sl] - train_max[sl]).clip(min=0)
        n_violations = ((below > 0) | (above > 0)).sum()
        n_total = eval_data[:, sl].size
        lines.append(
            f"  {group:10s} dims {sl.start:2d}:{sl.stop:2d}: "
            f"{n_violations:4d}/{n_total:4d} samples out of train range "
            f"(max below={below.max():.4f}, max above={above.max():.4f})"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="PCA / Mahalanobis OOD check for an iDP3 deployment rollout.")
    parser.add_argument("--zarr-path", type=Path,
                         default=Path("Improved-3D-Diffusion-Policy/data/g1_empty_bucket_v3_abs"),
                         help="Training zarr dataset (must contain data/state and data/action)")
    parser.add_argument("--episode-log", type=Path, required=True,
                         help="Path to episode_log.npz written during deployment")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Where to write plots/summary (default: <episode-log dir>/ood_analysis)")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.episode_log.parent / "ood_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading training data from {args.zarr_path} ...")
    train_state, train_action = load_training_data(args.zarr_path)
    print(f"  state: {train_state.shape}, action: {train_action.shape}")

    print(f"Loading eval rollout from {args.episode_log} ...")
    eval_state, eval_action = load_eval_data(args.episode_log)
    print(f"  agent_pos: {eval_state.shape}, executed action: {eval_action.shape}")

    summary_lines = [
        "=" * 70, "iDP3 Deployment OOD Check", "=" * 70,
        f"Training data: {args.zarr_path} ({len(train_state)} samples)",
        f"Eval rollout:  {args.episode_log} ({len(eval_state)} samples)", "",
    ]

    for name, train_data, eval_data, fname in [
        ("agent_pos", train_state, eval_state, "pca_agent_pos.png"),
        ("action", train_action, eval_action, "pca_action.png"),
    ]:
        scaler, pca = fit_pca(train_data)
        train_scaled = pca.transform(scaler.transform(train_data))
        eval_scaled = pca.transform(scaler.transform(eval_data))
        var_explained = pca.explained_variance_ratio_.sum() * 100

        plot_path = output_dir / fname
        plot_pca(train_scaled, eval_scaled,
                 f"{name}: PCA (train density vs eval rollout, {var_explained:.1f}% var explained)",
                 plot_path, eval_cmap_label='eval timestep', connect_line=True)
        print(f"Saved {plot_path}")

        summary_lines.append(mahalanobis_report(train_data, eval_data, name))
        summary_lines.append("")
        summary_lines.append(range_check_report(train_data, eval_data, name))
        summary_lines.append("")

    summary = "\n".join(summary_lines)
    print("\n" + summary)
    (output_dir / "summary.txt").write_text(summary + "\n")
    print(f"\nSaved summary to {output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
