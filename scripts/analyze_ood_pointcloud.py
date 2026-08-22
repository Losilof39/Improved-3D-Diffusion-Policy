#!/usr/bin/env python3
"""
Point-cloud-encoder OOD check for an iDP3 checkpoint.

Reconstructs point clouds from the depth+colour PNGs saved during real deployment
rollouts (policies/eval_logs/**/NNNNNN_{color,depth}.png), runs them through the
trained PointNet encoder, and compares the resulting embeddings against embeddings
of the training point clouds via PCA and Mahalanobis distance.

Produces:
  <output-dir>/pca_pointcloud_embedding.png
  <output-dir>/summary.txt

Usage:
  python scripts/analyze_ood_pointcloud.py

Run with the idp3 conda env:
  conda run -n idp3 python scripts/analyze_ood_pointcloud.py
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import zarr
from PIL import Image

from convert_unitree_to_zarr import depth_to_colored_pointcloud
from ood_common import fit_pca, mahalanobis_report, mahalanobis_scores, plot_pca

from diffusion_policy_3d.model.vision_3d.multi_stage_pointnet import MultiStagePointNetEncoder

# Camera intrinsics used to build data/g1_empty_bucket_v2 (see convert_g1_dataset.sh)
DEFAULT_INTRINSICS = dict(fx=607.7126, fy=607.5232, cx=319.4251, cy=243.5345)
DEFAULT_Z_NEAR = 0.1
DEFAULT_Z_FAR = 1.0
DEFAULT_DEPTH_SCALE = 1000.0


def build_extractor(ckpt_path, device):
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    cfg = ckpt['cfg']
    pc_cfg = cfg.policy.pointcloud_encoder_cfg
    use_pc_color = cfg.policy.use_pc_color
    in_channels = pc_cfg.in_channels

    extractor = MultiStagePointNetEncoder(in_channels=in_channels, out_channels=pc_cfg.out_channels)
    sd_key = 'ema_model' if 'ema_model' in ckpt['state_dicts'] else 'model'
    sd = ckpt['state_dicts'][sd_key]
    prefix = 'obs_encoder.extractor.'
    extractor_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    extractor.load_state_dict(extractor_sd)
    extractor.eval().to(device)
    return extractor, use_pc_color, in_channels


def embed_point_clouds(extractor, point_clouds, device, batch_size=256):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(point_clouds), batch_size):
            batch = torch.from_numpy(point_clouds[i:i + batch_size]).float().to(device)
            embeddings.append(extractor(batch).cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def load_training_point_clouds(zarr_path, n_samples, channels, seed=42):
    z = zarr.open(str(zarr_path), 'r')
    arr = z['data/point_cloud']
    n_total = arr.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_total, size=min(n_samples, n_total), replace=False))
    clouds = arr.oindex[idx][..., :channels]
    return np.asarray(clouds, dtype=np.float32), n_total


def find_eval_runs(eval_logs_dir):
    """Return list of (run_dir, sorted_frame_ids) for dirs with NNNNNN_color.png + episode_log.npz."""
    run_dirs = sorted({p.parent for p in eval_logs_dir.rglob("*_color.png")})
    runs = []
    for run_dir in run_dirs:
        if not (run_dir / "episode_log.npz").exists():
            continue
        frames = sorted(p.stem.rsplit('_', 1)[0] for p in run_dir.glob("*_color.png"))
        runs.append((run_dir, frames))
    return runs


def reconstruct_point_cloud(run_dir, frame_id, intrinsics, z_near, z_far, depth_scale, num_points, channels):
    color = np.array(Image.open(run_dir / f"{frame_id}_color.png").convert("RGB"))
    depth_raw = np.array(Image.open(run_dir / f"{frame_id}_depth.png"))
    depth_m = depth_raw.astype(np.float32) / depth_scale
    cloud = depth_to_colored_pointcloud(depth_m, color, **intrinsics, z_near=z_near, z_far=z_far, num_points=num_points)
    return cloud[:, :channels]


def main():
    parser = argparse.ArgumentParser(description="Point-cloud-encoder OOD check using deployment depth/color images.")
    parser.add_argument("--ckpt", type=Path, default=Path("latest.ckpt"))
    parser.add_argument("--zarr-path", type=Path, default=Path("Improved-3D-Diffusion-Policy/data/g1_empty_bucket_v3_abs"))
    parser.add_argument("--eval-logs-dir", type=Path, default=Path("policies/logs/vanilla/hard/20260622_075054"))
    parser.add_argument("--n-train-samples", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("policies/eval_logs/pointcloud_ood_analysis"))
    parser.add_argument("--fx", type=float, default=DEFAULT_INTRINSICS['fx'])
    parser.add_argument("--fy", type=float, default=DEFAULT_INTRINSICS['fy'])
    parser.add_argument("--cx", type=float, default=DEFAULT_INTRINSICS['cx'])
    parser.add_argument("--cy", type=float, default=DEFAULT_INTRINSICS['cy'])
    parser.add_argument("--z-near", type=float, default=DEFAULT_Z_NEAR)
    parser.add_argument("--z-far", type=float, default=DEFAULT_Z_FAR)
    parser.add_argument("--depth-scale", type=float, default=DEFAULT_DEPTH_SCALE)
    parser.add_argument("--num-points", type=int, default=4096)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading encoder from {args.ckpt} (device={device}) ...")
    extractor, use_pc_color, in_channels = build_extractor(args.ckpt, device)
    print(f"  use_pc_color={use_pc_color}, in_channels={in_channels}")

    print(f"Sampling {args.n_train_samples} training point clouds from {args.zarr_path} ...")
    train_clouds, n_train_total = load_training_point_clouds(args.zarr_path, args.n_train_samples, in_channels)
    print(f"  train_clouds: {train_clouds.shape} (of {n_train_total} total)")

    runs = find_eval_runs(args.eval_logs_dir)
    print(f"Found {len(runs)} eval rollouts under {args.eval_logs_dir}")

    intrinsics = dict(fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy)
    eval_clouds = []
    eval_run_idx = []
    for run_i, (run_dir, frames) in enumerate(runs):
        print(f"  [{run_i}] {run_dir} ({len(frames)} frames)")
        for frame_id in frames:
            cloud = reconstruct_point_cloud(run_dir, frame_id, intrinsics, args.z_near, args.z_far,
                                             args.depth_scale, args.num_points, in_channels)
            eval_clouds.append(cloud)
            eval_run_idx.append(run_i)
    eval_clouds = np.stack(eval_clouds).astype(np.float32)
    eval_run_idx = np.array(eval_run_idx)
    print(f"  eval_clouds: {eval_clouds.shape}")

    print("Embedding point clouds with the trained PointNet encoder ...")
    train_emb = embed_point_clouds(extractor, train_clouds, device)
    eval_emb = embed_point_clouds(extractor, eval_clouds, device)
    print(f"  train_emb: {train_emb.shape}, eval_emb: {eval_emb.shape}")

    scaler, pca = fit_pca(train_emb)
    train_scaled = pca.transform(scaler.transform(train_emb))
    eval_scaled = pca.transform(scaler.transform(eval_emb))
    var_explained = pca.explained_variance_ratio_.sum() * 100

    plot_path = args.output_dir / "pca_pointcloud_embedding.png"
    plot_pca(train_scaled, eval_scaled,
             f"Point cloud encoder embeddings: train density vs eval rollouts ({var_explained:.1f}% var explained)",
             plot_path, eval_color=eval_run_idx, eval_cmap_label='eval run index')
    print(f"Saved {plot_path}")

    train_dist, eval_dist = mahalanobis_scores(train_emb, eval_emb)
    p95 = np.percentile(train_dist, 95)

    summary_lines = [
        "=" * 70, "iDP3 Point-Cloud Encoder OOD Check", "=" * 70,
        f"Checkpoint: {args.ckpt}",
        f"Training point clouds: {args.zarr_path} ({len(train_clouds)} sampled of {n_train_total})",
        f"Eval rollouts: {len(runs)} runs, {len(eval_clouds)} frames total",
        "",
        mahalanobis_report(train_emb, eval_emb, "point cloud embedding"),
        "",
        "-- Per-run breakdown (frames beyond training p95) --",
    ]
    for run_i, (run_dir, frames) in enumerate(runs):
        mask = eval_run_idx == run_i
        frac = (eval_dist[mask] > p95).mean() * 100
        summary_lines.append(f"  [{run_i}] {run_dir}: {mask.sum():3d} frames, {frac:5.1f}% beyond training p95")

    summary = "\n".join(summary_lines)
    print("\n" + summary)
    (args.output_dir / "summary.txt").write_text(summary + "\n")
    print(f"\nSaved summary to {args.output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
