"""
Convert xr_teleoperate episode directories to an iDP3-compatible zarr dataset.

Episode directory structure (xr_teleoperate output):
  <input_dir>/
    episode_0000/
      data.json
      colors/000000_color_0.jpg  ...
      depths/000000_depth_0.png  ...
    episode_0001/
      ...

Output zarr structure (ReplayBuffer format):
  <output_zarr>/
    data/
      state        float32 (T, state_dim)   -- concatenated arm + hand qpos
      action       float32 (T, state_dim)   -- same layout as state
      point_cloud  float32 (T, N, 6)        -- XYZ + normalised RGB
      img          uint8   (T, H, W, 3)     -- RGB image (native resolution)
    meta/
      episode_ends int64   (num_episodes,)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Improved-3D-Diffusion-Policy"))

from diffusion_policy_3d.common.replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Depth → coloured point cloud
# ---------------------------------------------------------------------------

def depth_to_colored_pointcloud(
    depth_m: np.ndarray,        # (H, W) float32, metres
    rgb: np.ndarray,             # (H, W, 3) uint8
    fx: float, fy: float, cx: float, cy: float,
    z_near: float, z_far: float,
    num_points: int,
) -> np.ndarray:
    """Return (num_points, 6) float32 array: [x, y, z, r, g, b]."""
    H, W = depth_m.shape

    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_m.reshape(-1)
    x = (us.reshape(-1) - cx) * z / fx
    y = (vs.reshape(-1) - cy) * z / fy

    mask = (z >= z_near) & (z <= z_far)
    xyz = np.stack([x, y, z], axis=1)[mask]          # (N_valid, 3)
    color = (rgb.reshape(-1, 3).astype(np.float32) / 255.0)[mask]  # (N_valid, 3)

    cloud = np.concatenate([xyz, color], axis=1)     # (N_valid, 6)

    n = len(cloud)
    if n == 0:
        return np.zeros((num_points, 6), dtype=np.float32)

    if n >= num_points:
        idx = np.random.choice(n, num_points, replace=False)
    else:
        idx = np.concatenate([
            np.arange(n),
            np.random.choice(n, num_points - n, replace=True),
        ])
    return cloud[idx].astype(np.float32)


# ---------------------------------------------------------------------------
# JSON state / action helpers
# ---------------------------------------------------------------------------

def _get_nested(d: dict, dotted_key: str) -> list:
    """Walk 'left_arm.qpos' style keys."""
    keys = dotted_key.split(".")
    v = d
    for k in keys:
        v = v[k]
    return v


STATE_KEYS = ["left_arm.qpos", "right_arm.qpos", "left_ee.qpos", "right_ee.qpos"]


def extract_vector(frame_dict: dict, section: str) -> np.ndarray:
    """Concatenate all joint-position fields from states or actions."""
    parts = []
    for k in STATE_KEYS:
        top, field = k.split(".", 1)
        val = frame_dict[section][top][field]
        if val:
            parts.append(np.array(val, dtype=np.float32))
    return np.concatenate(parts) if parts else np.array([], dtype=np.float32)


# ---------------------------------------------------------------------------
# Episode processing
# ---------------------------------------------------------------------------

def process_episode(
    episode_dir: Path,
    fx: float, fy: float, cx: float, cy: float,
    color_key: str, depth_key: str,
    depth_scale: float,
    z_near: float, z_far: float,
    num_points: int,
):
    data_json = episode_dir / "data.json"
    if not data_json.exists():
        return None

    with open(data_json) as f:
        data = json.load(f)

    frames = data.get("data", [])
    if not frames:
        return None

    ep_states, ep_actions, ep_clouds, ep_imgs = [], [], [], []

    for frame in frames:
        # --- proprioception ---
        state  = extract_vector(frame, "states")
        action = extract_vector(frame, "actions")

        # --- point cloud from depth + colour ---
        depth_rel = frame.get("depths", {}).get(depth_key)
        color_rel = frame.get("colors", {}).get(color_key)

        if depth_rel is None or color_rel is None:
            print(f"  [warn] frame {frame['idx']} missing {depth_key}/{color_key}, skipping episode")
            return None

        depth_path = episode_dir / depth_rel
        color_path = episode_dir / color_rel

        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        if depth_raw is None:
            print(f"  [warn] could not read {depth_path}, skipping episode")
            return None
        depth_m = depth_raw.astype(np.float32) / depth_scale

        bgr = cv2.imread(str(color_path))
        if bgr is None:
            print(f"  [warn] could not read {color_path}, skipping episode")
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # align resolutions
        if depth_m.shape[:2] != rgb.shape[:2]:
            h, w = rgb.shape[:2]
            depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)

        cloud = depth_to_colored_pointcloud(
            depth_m, rgb, fx, fy, cx, cy, z_near, z_far, num_points
        )

        ep_states.append(state)
        ep_actions.append(action)
        ep_clouds.append(cloud)
        ep_imgs.append(rgb)

    return (
        np.stack(ep_states),   # (T, state_dim)
        np.stack(ep_actions),  # (T, state_dim)
        np.stack(ep_clouds),   # (T, num_points, 6)
        np.stack(ep_imgs),     # (T, H, W, 3)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert xr_teleoperate episodes to iDP3 zarr")
    parser.add_argument("--input_dir",   required=True,  help="Folder containing episode_XXXX/ subdirs")
    parser.add_argument("--output_zarr", required=True,  help="Output zarr path")
    parser.add_argument("--fx",   type=float, required=True, help="Camera focal length x (pixels)")
    parser.add_argument("--fy",   type=float, required=True, help="Camera focal length y (pixels)")
    parser.add_argument("--cx",   type=float, required=True, help="Camera principal point x (pixels)")
    parser.add_argument("--cy",   type=float, required=True, help="Camera principal point y (pixels)")
    parser.add_argument("--color_key",   default="color_0", help="Color image key in data.json (default: color_0)")
    parser.add_argument("--depth_key",   default="depth_0", help="Depth image key in data.json (default: depth_0)")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="Divide raw depth by this to get metres (default: 1000.0)")
    parser.add_argument("--z_near",      type=float, default=0.1,    help="Minimum depth in metres (default: 0.1)")
    parser.add_argument("--z_far",       type=float, default=1.5,    help="Maximum depth in metres (default: 1.5)")
    parser.add_argument("--num_points",  type=int,   default=4096,   help="Points per cloud (default: 4096)")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    input_dir = Path(args.input_dir)
    episode_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("episode"))

    if not episode_dirs:
        sys.exit(f"No episode_XXXX/ directories found in {input_dir}")

    print(f"Found {len(episode_dirs)} episode directories")

    rb = ReplayBuffer.create_empty_numpy()
    n_ok, n_skip = 0, 0

    for ep_dir in episode_dirs:
        print(f"  Processing {ep_dir.name} ...", end=" ", flush=True)
        result = process_episode(
            ep_dir,
            args.fx, args.fy, args.cx, args.cy,
            args.color_key, args.depth_key,
            args.depth_scale, args.z_near, args.z_far,
            args.num_points,
        )
        if result is None:
            print("SKIPPED")
            n_skip += 1
            continue

        ep_states, ep_actions, ep_clouds, ep_imgs = result
        rb.add_episode({
            "state":       ep_states,
            "action":      ep_actions,
            "point_cloud": ep_clouds,
            "img":         ep_imgs,
        })
        print(f"OK  ({len(ep_states)} frames, state_dim={ep_states.shape[1]}, img={ep_imgs.shape[1:]})")
        n_ok += 1

    if n_ok == 0:
        sys.exit("No episodes were converted successfully.")

    output_path = Path(args.output_zarr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rb.save_to_path(str(output_path))

    print(f"\nSaved {n_ok} episodes ({n_skip} skipped) → {output_path}")
    print(f"  state:        {rb['state'].shape}")
    print(f"  action:       {rb['action'].shape}")
    print(f"  point_cloud:  {rb['point_cloud'].shape}")
    print(f"  img:          {rb['img'].shape}")
    print(f"  episode_ends: {rb.episode_ends[:10]}{'...' if rb.n_episodes > 10 else ''}")


if __name__ == "__main__":
    main()
