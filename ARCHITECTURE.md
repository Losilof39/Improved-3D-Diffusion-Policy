# iDP3 Model Architecture — G1 Bucket-Emptying Task

This documents the model trained as `g1_dex-3d-idp3-empty_bucket_v2_seed0`, and the
augmentation/RGB changes added afterwards for the next training run.

## Overview

- **Robot**: Unitree G1 humanoid + Inspire/BrainCo dexterous hands.
- **Task** (`g1_task`, `diffusion_policy_3d/config/task/g1_dex-3d.yaml`): pick up a bucket
  with one hand while the other hand helps empty it into a box.
- **Algorithm**: iDP3 (point-cloud-conditioned diffusion policy), trained via
  `diffusion_policy_3d/workspace/idp3_workspace.py`.
- **Action space**: 26 raw joint angles — `[0:7] left_arm`, `[7:14] right_arm`,
  `[14:20] left_hand`, `[20:26] right_hand`. No end-effector/Cartesian targets.

## Observation / Action Spaces

From `shape_meta` (`diffusion_policy_3d/config/task/g1_dex-3d.yaml`):

| Key | Shape | Type | Notes |
|---|---|---|---|
| `point_cloud` | `[4096, 6]` | point_cloud | XYZ + RGB (RGB normalized to `[0,1]`) |
| `agent_pos` | `[26]` | low_dim | proprioception (joint angles) |
| `action` | `[26]` | — | joint-angle targets |

**Temporal config** (`diffusion_policy_3d/config/idp3.yaml`):
- `n_obs_steps = 2`, `horizon = 16`, `n_action_steps = 15` (predict 16, execute 15,
  re-plan with 2-step observation history).

## Point Cloud Encoder

`iDP3Encoder` (`diffusion_policy_3d/model/vision_3d/pointnet_extractor.py`) combines:

1. **Multi-Stage PointNet** (`diffusion_policy_3d/model/vision_3d/multi_stage_pointnet.py`)
   - `in_channels`: 3 (XYZ-only, original run) → **6 (XYZ+RGB) for the new run**, see
     [Point Cloud Color (RGB)](#point-cloud-color-rgb) below.
   - `h_dim = 128`, `out_channels = 128`, `num_layers = 4`.
   - 4-stage local→global feature extraction with per-stage max-pooling, concatenated
     (4×128=512) and projected to `out_channels=128`, then global max-pool → `[B,128]`.
   - `use_layernorm: true`, `final_norm: layernorm`, `normal_channel: false`.
2. **Proprioception MLP**: `agent_pos` (26-dim) → `64 → 64` MLP → `[B,64]`.
3. **Final encoder output**: concat → `[B,192]` (192 = 128 point-cloud feats + 64 state feats).

**Point cloud preprocessing**: `point_downsample: true`, points are uniform-randomly
re-sampled/shuffled to `num_points=4096` on every `__getitem__`
(`diffusion_policy_3d/model/vision_3d/point_process.py:uniform_sampling_numpy`). Since the
zarr already stores exactly 4096 points/frame, this is effectively a random re-shuffle
(harmless — PointNet is permutation-invariant via max-pooling).

## Diffusion Model

`ConditionalUnet1D` (`diffusion_policy_3d/model/diffusion/conditional_unet1d.py`), driven
by `DiffusionPointcloudPolicy` (`diffusion_policy_3d/policy/diffusion_pointcloud_policy.py`):

- `down_dims = [256, 512, 1024]`, `kernel_size = 5`, `n_groups = 8` (GroupNorm),
  `diffusion_step_embed_dim = 128`.
- FiLM conditioning: `obs_as_global_cond: true`, `use_down_condition/use_mid_condition/use_up_condition: true`.
- Conditioning vector = diffusion-step embedding (128) + observation encoding (192) = 320-dim.

**Noise scheduler**: DDIM (`diffusers.schedulers.scheduling_ddim.DDIMScheduler`)
- `num_train_timesteps = 50`, `num_inference_steps = 10` (5x faster sampling at deploy time).
- `beta_schedule = squaredcos_cap_v2`, `beta_start = 1e-4`, `beta_end = 0.02`.
- `prediction_type = sample` (model predicts the clean trajectory directly).
- `clip_sample = true`, `set_alpha_to_one = true`.

## EMA

`diffusion_policy_3d/model/diffusion/ema_model.py`, enabled (`use_ema: true`):
- `update_after_step = 0`, `inv_gamma = 1.0`, `power = 0.75`, `min_value = 0.0`, `max_value = 0.9999`.
- Decay ramps up from 0 toward 0.9999 as training progresses (≈0.999 by ~10k steps).

## Optimizer & LR Schedule

- **Optimizer**: AdamW, `lr = 1e-4`, `betas = [0.95, 0.999]`, `eps = 1e-8`, `weight_decay = 1e-6`.
- **LR schedule**: cosine, `lr_warmup_steps = 500`.

## Training Hyperparameters

- `num_epochs = 301`, `batch_size = 24` (train & val), `gradient_accumulate_every = 1`.
- `val_every = 100`, `checkpoint_every = 100`.
- Checkpointing: `save_last_ckpt: true`, `topk.k = 0` → only `latest.ckpt` is kept (no
  best-epoch snapshot). `topk.monitor_key = test_mean_score` (mode max), but with `k=0`
  this never actually saves a top-k checkpoint.

## Data Augmentation & Preprocessing

### What existed in the `empty_bucket_v2_seed0` run (audit result)

- **Existed**: per-`__getitem__` point resampling/shuffling to 4096 points (incidental,
  not real augmentation since the zarr already has exactly 4096 pts/frame); action
  min-max normalization to `[-1,1]` (`gr1_dex_dataset_3d.py:get_normalizer`,
  `mode='limits'`); `point_cloud`/`agent_pos` use **identity** normalization (no scaling);
  RGB channels discarded (`use_pc_color: false`).
- **Absent**: any Gaussian noise/jitter, random rotation/translation/scaling, point
  dropout, image augmentation (n/a — no RGB images fed to the policy), temporal/sequence
  augmentation.

So the `empty_bucket_v2_seed0` model was trained on essentially raw, unaugmented point
clouds — it has only ever seen the exact geometry of your demonstrations.

### New: point cloud jitter + dropout (`diffusion_policy_3d/dataset/gr1_dex_dataset_3d.py`)

Added to `GR1DexDataset3D`, applied in `_sample_to_data` after the 4096-point resampling,
controlled by new `task.dataset` config keys (set in `g1_dex-3d.yaml`):

```yaml
use_pc_augmentation: true
pc_jitter_std: 0.005   # meters, gaussian jitter on point cloud XYZ
pc_dropout_ratio: 0.05 # fraction of points replaced with duplicates each frame
```

- **Jitter**: adds `N(0, pc_jitter_std)` noise to XYZ only (channels `:3`); RGB channels
  are untouched. Simulates depth-sensor noise — doesn't change the scene's overall
  geometry, so the (observation, action) pairing stays valid.
- **Dropout**: for each frame, `round(4096 * pc_dropout_ratio)` random points are
  overwritten (all 6 channels) with copies of other random points in the same frame —
  simulates missing/occluded depth returns while preserving the point count.
- Applied independently per frame in the `horizon`-length sequence.
- `get_validation_dataset()` disables augmentation on its copy (relevant only if
  `val_ratio > 0` is used in the future — currently `val_ratio: 0.0`).
- Tune/disable via Hydra overrides, e.g.:
  `task.dataset.pc_jitter_std=0.01`, `task.dataset.use_pc_augmentation=false`.

**Known minor side-effect**: the in-loop "validation" pass in `idp3_workspace.py`
(lines 238-258) re-iterates `train_dataloader`, so `train_action_mse_error`/
`test_mean_score` will now also see jittered/dropout point clouds, slightly inflating
that number. This metric was already training-set MSE (not a held-out signal, since
`val_ratio=0.0`), so this is cosmetic.

## Point Cloud Color (RGB)

The conversion script (`scripts/convert_unitree_to_zarr.py`) has always written
6-channel point clouds — `point_cloud` shape `(N_frames, 4096, 6)`, channels 3-5 = RGB
normalized to `[0,1]` (`color = rgb / 255.0`). The `empty_bucket_v2_seed0` model never
used this — `policy.use_pc_color: false` stripped the point cloud to XYZ only.

**Now enabled by default for G1** (`scripts/train_policy.sh` passes):
```
policy.use_pc_color=true
policy.pointcloud_encoder_cfg.in_channels=6
```

Code changes that made this work:
- `MultiStagePointNetEncoder` (`model/vision_3d/multi_stage_pointnet.py`) used to
  hardcode `Conv1d(3, h_dim, ...)` and ignore the `in_channels` config entirely. It now
  takes `in_channels` (default `3`, backward compatible) and builds
  `Conv1d(in_channels, h_dim, ...)` accordingly.
- `iDP3Encoder` (`model/vision_3d/pointnet_extractor.py`) now passes
  `in_channels=pointcloud_encoder_cfg.in_channels` through to the PointNet encoder.
- `DiffusionPointcloudPolicy` (`policy/diffusion_pointcloud_policy.py`) previously did
  `point_cloud[..., 3:] /= 255.0` when `use_pc_color=True`. Since this dataset's RGB is
  *already* `[0,1]`, that extra division has been removed (it would have shrunk color
  values to `~[0, 0.004]`, effectively zeroing the signal).

**Important**: changing `in_channels` from 3→6 changes the shape of the encoder's first
conv layer, so **the existing `latest.ckpt` cannot be resumed/fine-tuned** — this must be
trained from scratch with a new run directory, e.g.:
```
bash scripts/train_policy.sh idp3 g1_dex-3d empty_bucket_v2_rgb_aug
```
(a new `addition_info` value gives a fresh `exp_name`/`run_dir`, avoiding a shape
mismatch when `idp3_workspace.py` tries to resume `g1_dex-3d-idp3-empty_bucket_v2_seed0`).

## Next Steps: Spatial Generalization (R2RGen)

If local jitter/dropout + RGB color isn't enough to fix the "approaches the bucket then
stalls" behavior, the next step suggested by the user is **R2RGen** (arXiv:2510.08547,
"real-to-real" 3D data generation): a simulator/rendering-free method that repositions
objects/robot within real point clouds (via a group-wise backtracking strategy +
camera-aware post-processing) to generate spatially-diverse demonstrations.

**Caveat for this setup**: R2RGen retargets the *action trajectory* to match the
repositioned scene. This G1 policy's actions are 26 raw **joint angles**, not
end-effector poses — so repositioning the bucket and correspondingly retargeting the
joint-angle trajectory would require an **IK solver** for the G1 arms/hands. That's a
substantially bigger project than the augmentation added here, and should be scoped
separately if jitter/dropout/RGB don't resolve the stalling behavior.

Other things worth checking first if stalling persists:
- Whether demonstrations have enough coverage of the final grasp/contact phase (vs. mostly
  "approach" trajectories) — diffusion policies trained mostly on approach motions can
  regress toward "no movement" near contact.
- Whether the gripper/hand occludes the bucket in the point cloud right before grasp in a
  way training data doesn't represent.

## File Reference Index

| Component | File |
|---|---|
| Task config | `diffusion_policy_3d/config/task/g1_dex-3d.yaml` |
| Base policy/training config | `diffusion_policy_3d/config/idp3.yaml` |
| Dataset | `diffusion_policy_3d/dataset/gr1_dex_dataset_3d.py` |
| Point cloud sampling utils | `diffusion_policy_3d/model/vision_3d/point_process.py` |
| Point cloud encoder | `diffusion_policy_3d/model/vision_3d/pointnet_extractor.py` |
| Multi-Stage PointNet | `diffusion_policy_3d/model/vision_3d/multi_stage_pointnet.py` |
| Diffusion policy | `diffusion_policy_3d/policy/diffusion_pointcloud_policy.py` |
| UNet1D | `diffusion_policy_3d/model/diffusion/conditional_unet1d.py` |
| EMA | `diffusion_policy_3d/model/diffusion/ema_model.py` |
| Training loop | `diffusion_policy_3d/workspace/idp3_workspace.py` |
| Dataset conversion | `scripts/convert_unitree_to_zarr.py` |
| Training entrypoint | `scripts/train_policy.sh` |
