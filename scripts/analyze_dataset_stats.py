#!/usr/bin/env python3
"""
Extract thesis-ready statistics and figures from iDP3 G1 Zarr datasets.

For each --zarr-path, reports exact episode/timestep/disk-size numbers plus
sampled per-dimension action/state/point_cloud statistics (a fixed-size random
subset of episodes, not the full dataset - point clouds are multi-GB).

Produces, in --output-dir:
  report.md   - prose + tables, meant to be pasted into a thesis
  stats.json  - the same numbers in structured form
  figures/episode_length_distribution.png
  figures/action_range_by_joint_group.png
  figures/dataset_overview.png

Usage:
  python scripts/analyze_dataset_stats.py

Run with the `idp3` conda env (has zarr/numpy/matplotlib):
  /home/luke/anaconda3/envs/idp3/bin/python scripts/analyze_dataset_stats.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_INNER = Path(__file__).resolve().parent.parent / "Improved-3D-Diffusion-Policy"
sys.path.insert(0, str(REPO_INNER))

from diffusion_policy_3d.common.replay_buffer import ReplayBuffer

DEFAULT_ZARR_PATHS = [
    REPO_INNER / "data" / "g1_empty_bucket_v3_abs",
    REPO_INNER / "data" / "g1_empty_bucket_60ep_cut",
    REPO_INNER / "data" / "g1_pick_and_place_60ep",
]

DISPLAY_NAMES = {
    "g1_empty_bucket_v3_abs": "Bucket Task (100 episodes)",
    "g1_empty_bucket_60ep_cut": "Bucket Task (60 episodes)",
    "g1_pick_and_place_60ep": "Pick&Place (60 episodes)",
}

# dataviz reference palette, categorical slots 1-3 (blue/green/magenta) - the
# documented set that passes all-pairs CVD + normal-vision floors in both
# light and dark modes. Assign by position, never re-sorted by value.
COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

STATE_OUTLIER_THRESHOLD = 10.0

KNOWN_RELATIONSHIPS = {
    "g1_empty_bucket_v3_abs": (
        "g1_empty_bucket_60ep_cut is a subset of this dataset: episodes 0-19 "
        "are byte-identical between the two. This is a re-cut/extended "
        "collection (100 vs. 60 episodes), not an independent set of "
        "demonstrations - do not sum episode counts across the two when "
        "reporting total unique demonstrations."
    ),
}


def joint_groups_for_dim(d):
    if d == 26:
        return [
            ("left_arm", slice(0, 7)),
            ("right_arm", slice(7, 14)),
            ("left_hand", slice(14, 20)),
            ("right_hand", slice(20, 26)),
        ]
    if d == 13:
        return [
            ("left_arm", slice(0, 7)),
            ("left_hand", slice(7, 13)),
        ]
    return [("all_dims", slice(0, d))]


def dir_size_bytes(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total


def sample_episode_data(rb, keys, n_sample, rng):
    n_episodes = rb.n_episodes
    idxs = rng.choice(n_episodes, size=min(n_sample, n_episodes), replace=False)
    idxs.sort()
    out = {k: [] for k in keys}
    for idx in idxs:
        sl = rb.get_episode_slice(int(idx))
        for k in keys:
            out[k].append(rb[k][sl])
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}, idxs.tolist()


def per_dim_stats(arr):
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
    }


def group_stats(arr, groups):
    out = {}
    for name, sl in groups:
        sub = arr[:, sl]
        out[name] = {
            "min": float(sub.min()),
            "max": float(sub.max()),
            "mean": float(sub.mean()),
            "std": float(sub.std()),
        }
    return out


def outlier_report(arr, threshold):
    n = arr.shape[0]
    report = []
    for d in range(arr.shape[1]):
        col = arr[:, d]
        count = int(np.sum(np.abs(col) > threshold))
        if count > 0:
            report.append({
                "dim": d,
                "count": count,
                "pct": 100.0 * count / n,
                "max_abs": float(np.max(np.abs(col))),
            })
    return report


def analyze_dataset(path, n_sample, seed):
    path = Path(path)
    name = path.name
    rb = ReplayBuffer.create_from_path(str(path), mode="r")

    n_episodes = rb.n_episodes
    episode_lengths = np.asarray(rb.episode_lengths)
    n_steps = int(rb.n_steps)
    disk_bytes = dir_size_bytes(path)

    shapes = {k: {"shape": list(rb[k].shape), "dtype": str(rb[k].dtype)} for k in rb.keys()}

    action_dim = rb["action"].shape[1]
    groups = joint_groups_for_dim(action_dim)

    rng = np.random.default_rng(seed)
    sample_keys = [k for k in ("action", "state", "point_cloud") if k in rb]
    sampled, sample_ep_idxs = sample_episode_data(rb, sample_keys, n_sample, rng)

    action_sample = sampled["action"]
    state_sample = sampled["state"]
    pc_sample = sampled.get("point_cloud")

    result = {
        "name": name,
        "display_name": DISPLAY_NAMES.get(name, name),
        "path": str(path),
        "n_episodes": int(n_episodes),
        "n_steps": n_steps,
        "episode_length": {
            "min": int(episode_lengths.min()),
            "max": int(episode_lengths.max()),
            "mean": float(episode_lengths.mean()),
            "median": float(np.median(episode_lengths)),
            "std": float(episode_lengths.std()),
        },
        "episode_lengths_raw": episode_lengths.tolist(),
        "disk_bytes": int(disk_bytes),
        "disk_gb": disk_bytes / 1e9,
        "shapes": shapes,
        "action_dim": int(action_dim),
        "joint_groups": [g[0] for g in groups],
        "sample": {
            "n_episodes_sampled": len(sample_ep_idxs),
            "episode_idxs": sample_ep_idxs,
            "n_frames_sampled": int(action_sample.shape[0]),
        },
        "action_per_dim": per_dim_stats(action_sample),
        "state_per_dim": per_dim_stats(state_sample),
        "action_per_group": group_stats(action_sample, groups),
        "state_per_group": group_stats(state_sample, groups),
        "state_outliers": outlier_report(state_sample, STATE_OUTLIER_THRESHOLD),
    }

    if pc_sample is not None:
        channel_names = ["x", "y", "z", "r", "g", "b"][:pc_sample.shape[-1]]
        flat = pc_sample.reshape(-1, pc_sample.shape[-1])
        result["point_cloud"] = {
            "points_per_frame": int(pc_sample.shape[1]),
            "channels": channel_names,
            "per_channel": {
                ch: {
                    "min": float(flat[:, i].min()),
                    "max": float(flat[:, i].max()),
                    "mean": float(flat[:, i].mean()),
                }
                for i, ch in enumerate(channel_names)
            },
        }

    if name in KNOWN_RELATIONSHIPS:
        result["known_relationship"] = KNOWN_RELATIONSHIPS[name]

    return result


def make_episode_length_figure(results, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = [np.asarray(r["episode_lengths_raw"]) for r in results]
    labels = [r["display_name"] for r in results]
    colors = COLORS[:len(results)]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    for element in ("whiskers", "caps"):
        for line, color in zip(bp[element], np.repeat(colors, 2)):
            line.set_color(color)
    for line, color in zip(bp["medians"], colors):
        line.set_color(color)
    ax.set_ylabel("episode length (timesteps)")
    ax.set_title("Episode length distribution by dataset", color=INK)
    ax.grid(True, axis="y", alpha=0.3, color=GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


def make_action_range_figure(results, out_path):
    all_groups = []
    for r in results:
        for g in r["joint_groups"]:
            if g not in all_groups:
                all_groups.append(g)

    fig, ax = plt.subplots(figsize=(10, 6))
    n_datasets = len(results)
    n_groups = len(all_groups)
    bar_width = 0.8 / n_datasets
    x = np.arange(n_groups)

    for i, (r, color) in enumerate(zip(results, COLORS[:n_datasets])):
        mins = [r["action_per_group"].get(g, {}).get("min", np.nan) for g in all_groups]
        maxs = [r["action_per_group"].get(g, {}).get("max", np.nan) for g in all_groups]
        means = [r["action_per_group"].get(g, {}).get("mean", np.nan) for g in all_groups]
        offsets = x + (i - (n_datasets - 1) / 2) * bar_width
        heights = np.array(maxs) - np.array(mins)
        ax.bar(offsets, heights, bottom=mins, width=bar_width * 0.9,
               color=color, alpha=0.35, edgecolor=color, label=r["display_name"])
        ax.scatter(offsets, means, color=color, s=18, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(all_groups)
    ax.set_ylabel("action value range (min-max, dot = mean)")
    ax.set_title("Action range by joint group, by dataset (sampled episodes)", color=INK)
    ax.grid(True, axis="y", alpha=0.3, color=GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


def make_overview_figure(results, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    labels = [r["display_name"] for r in results]
    colors = COLORS[:len(results)]

    metrics = [
        ("n_episodes", "episodes", axes[0]),
        ("n_steps", "total timesteps", axes[1]),
        ("disk_gb", "disk size (GB)", axes[2]),
    ]
    for key, title, ax in metrics:
        values = [r[key] for r in results]
        bars = ax.bar(labels, values, color=colors, alpha=0.85)
        for bar, v in zip(bars, values):
            label = f"{v:.1f}" if isinstance(v, float) else str(v)
            ax.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=8, color=MUTED)
        ax.set_title(title, color=INK)
        ax.grid(True, axis="y", alpha=0.3, color=GRID)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelrotation=15)
        ax.yaxis.label.set_color(INK)

    fig.suptitle("Dataset overview", color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


def format_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def build_report(results):
    lines = ["# G1 iDP3 dataset statistics", ""]

    lines.append("## Overview")
    lines.append("")
    overview_rows = []
    for r in results:
        el = r["episode_length"]
        overview_rows.append([
            r["display_name"],
            r["n_episodes"],
            r["n_steps"],
            f'{el["min"]} / {el["mean"]:.1f} / {el["max"]}',
            r["action_dim"],
            f'{r["disk_gb"]:.1f}',
        ])
    lines.append(format_table(
        ["dataset", "episodes", "timesteps", "ep. length min/mean/max", "action dim", "disk (GB)"],
        overview_rows,
    ))
    lines.append("")

    for r in results:
        lines.append(f"## {r['display_name']}")
        lines.append("")
        lines.append(f"- Path: `{r['path']}`")
        lines.append(f"- Episodes: {r['n_episodes']}, total timesteps: {r['n_steps']}")
        el = r["episode_length"]
        lines.append(
            f"- Episode length (timesteps): min={el['min']}, mean={el['mean']:.1f}, "
            f"median={el['median']:.1f}, max={el['max']}, std={el['std']:.1f}"
        )
        lines.append(f"- On-disk size: {r['disk_gb']:.2f} GB")
        lines.append(f"- Action/state dimensionality: {r['action_dim']} ({', '.join(r['joint_groups'])})")
        for key in ("action", "state", "point_cloud", "img"):
            if key in r["shapes"]:
                s = r["shapes"][key]
                lines.append(f"  - `{key}`: shape {tuple(s['shape'])}, dtype {s['dtype']}")
        if "known_relationship" in r:
            lines.append("")
            lines.append(f"> **Note:** {r['known_relationship']}")
        lines.append("")

        lines.append(
            f"Per-group statistics computed from a random sample of "
            f"{r['sample']['n_episodes_sampled']} of {r['n_episodes']} episodes "
            f"({r['sample']['n_frames_sampled']} timesteps), fixed seed."
        )
        lines.append("")

        action_rows = [
            [g, f'{s["min"]:.3f}', f'{s["mean"]:.3f}', f'{s["max"]:.3f}', f'{s["std"]:.3f}']
            for g, s in r["action_per_group"].items()
        ]
        lines.append("**Action, by joint group** (min / mean / max / std):")
        lines.append(format_table(["group", "min", "mean", "max", "std"], action_rows))
        lines.append("")

        state_rows = [
            [g, f'{s["min"]:.3f}', f'{s["mean"]:.3f}', f'{s["max"]:.3f}', f'{s["std"]:.3f}']
            for g, s in r["state_per_group"].items()
        ]
        lines.append("**State (agent_pos), by joint group** (min / mean / max / std):")
        lines.append(format_table(["group", "min", "mean", "max", "std"], state_rows))
        lines.append("")

        if r["state_outliers"]:
            lines.append(
                f"> **Data-quality note:** raw `state` contains values exceeding "
                f"|{STATE_OUTLIER_THRESHOLD:.0f}| in {len(r['state_outliers'])} dimension(s) "
                f"of the sample - likely a sensor/unit artifact in specific hand-joint "
                f"channels, not physically plausible joint values. `action` is unaffected "
                f"because it is min-max normalized at train time while `state`/`agent_pos` "
                f"uses identity normalization (see `GR1DexDataset3D.get_normalizer()`)."
            )
            outlier_rows = [
                [o["dim"], o["count"], f'{o["pct"]:.2f}%', f'{o["max_abs"]:.1f}']
                for o in r["state_outliers"]
            ]
            lines.append(format_table(["dim", "count", "% of sample", "max |value|"], outlier_rows))
            lines.append("")

        if "point_cloud" in r:
            pc = r["point_cloud"]
            lines.append(
                f"**Point cloud**: {pc['points_per_frame']} points/frame, "
                f"channels {pc['channels']} (xyz in metres, depth-camera frame; rgb normalized [0,1])."
            )
            pc_rows = [
                [ch, f'{s["min"]:.3f}', f'{s["mean"]:.3f}', f'{s["max"]:.3f}']
                for ch, s in pc["per_channel"].items()
            ]
            lines.append(format_table(["channel", "min", "mean", "max"], pc_rows))
            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Extract statistics/figures from iDP3 G1 zarr datasets.")
    parser.add_argument("--zarr-paths", type=Path, nargs="+", default=DEFAULT_ZARR_PATHS,
                        help="Zarr dataset directories to analyze")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_dataset_stats"),
                        help="Where to write report.md, stats.json, and figures/")
    parser.add_argument("--sample-episodes", type=int, default=20,
                        help="Number of episodes to sample per dataset for per-dimension stats")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for episode sampling")
    args = parser.parse_args()

    if len(args.zarr_paths) > len(COLORS):
        parser.error(f"supports at most {len(COLORS)} datasets (dataviz palette limit)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for path in args.zarr_paths:
        print(f"Analyzing {path} ...")
        results.append(analyze_dataset(path, args.sample_episodes, args.seed))

    stats_path = args.output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {stats_path}")

    report_path = args.output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(build_report(results))
    print(f"Saved {report_path}")

    make_episode_length_figure(results, figures_dir / "episode_length_distribution.png")
    print(f"Saved {figures_dir / 'episode_length_distribution.png'}")

    make_action_range_figure(results, figures_dir / "action_range_by_joint_group.png")
    print(f"Saved {figures_dir / 'action_range_by_joint_group.png'}")

    make_overview_figure(results, figures_dir / "dataset_overview.png")
    print(f"Saved {figures_dir / 'dataset_overview.png'}")


if __name__ == "__main__":
    main()
