"""Replace raw G1 joint-angle arrays with EE-pose + hand-joint arrays via forward kinematics.

Computes left/right hand palm pose (xyz + quaternion xyzw) from the 7-DOF arm
joint angles stored in the original `state`/`action` ([0:7]=left arm,
[7:14]=right arm, [14:20]=left hand, [20:26]=right hand, see
diffusion_policy_3d/common/g1_action_util.py), using pinocchio FK against the
G1 URDF, and folds the hand joint angles in alongside the arm pose. The
joint-angle `state`/`action` arrays are overwritten in place with this new
representation (same key names, same 26-dim shape, new semantics) so they
stay plug-compatible with dataset loaders that expect `state`/`action`.

`state` becomes the absolute observation (arm EE pose + absolute hand angles);
`action` becomes a frame-to-frame delta for both the arm pose (position +
orientation) and the hand joint angles, zeroed/identity at the start of each
episode — mirrors the abs-observation / delta-action convention used by GR1
(diffusion_policy_3d/common/gr1_action_util.py). Note this makes `action` a
relative EE-space command, not a joint-space one — deployment code must
integrate/IK it back into joint targets.

`state`/`action` layout after conversion (still 26-dim each):
  state  (N, 26) - [left_xyz(3), left_quat_xyzw(4), right_xyz(3), right_quat_xyzw(4),
                     left_hand(6), right_hand(6)], hand angles absolute
  action (N, 26) - same layout, frame-to-frame delta of the original action
                    (arm: position + orientation delta; hand: joint-angle delta)
"""
import argparse

import numpy as np
import pinocchio as pin
import zarr
from numcodecs import Blosc
from scipy.spatial.transform import Rotation

LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [n.replace("left_", "right_") for n in LEFT_ARM_JOINTS]
LEFT_EE_FRAME = "left_hand_palm_link"
RIGHT_EE_FRAME = "right_hand_palm_link"

# Keys from earlier iterations of this script that should be cleaned up if present.
LEGACY_KEYS = ["ee_pose", "ee_pose_delta_action", "ee_pose_action", "ee_pos_delta", "ee_pos_delta_action"]


class G1ArmFK:
    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.q_neutral = pin.neutral(self.model)
        self.left_idx = [self.model.joints[self.model.getJointId(n)].idx_q for n in LEFT_ARM_JOINTS]
        self.right_idx = [self.model.joints[self.model.getJointId(n)].idx_q for n in RIGHT_ARM_JOINTS]
        self.left_frame_id = self.model.getFrameId(LEFT_EE_FRAME)
        self.right_frame_id = self.model.getFrameId(RIGHT_EE_FRAME)

    def compute(self, joints26: np.ndarray) -> np.ndarray:
        """joints26: (T, 26) -> (T, 14) [left_xyz, left_quat_xyzw, right_xyz, right_quat_xyzw]."""
        T = joints26.shape[0]
        out = np.zeros((T, 14), dtype=np.float32)
        q = self.q_neutral.copy()
        for t in range(T):
            q[self.left_idx] = joints26[t, 0:7]
            q[self.right_idx] = joints26[t, 7:14]
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            left_se3 = self.data.oMf[self.left_frame_id]
            right_se3 = self.data.oMf[self.right_frame_id]
            out[t, 0:3] = left_se3.translation
            out[t, 3:7] = pin.Quaternion(left_se3.rotation).coeffs()
            out[t, 7:10] = right_se3.translation
            out[t, 10:14] = pin.Quaternion(right_se3.rotation).coeffs()
        return out


def _position_delta(xyz: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(xyz)
    delta[1:] = xyz[1:] - xyz[:-1]
    return delta


def _orientation_delta(quat_xyzw: np.ndarray) -> np.ndarray:
    """Frame-to-frame relative rotation R_t * R_{t-1}^-1, as xyzw quaternion. Identity at t=0."""
    T = quat_xyzw.shape[0]
    delta = np.zeros((T, 4), dtype=np.float32)
    delta[:, 3] = 1.0
    rot_t = Rotation.from_quat(quat_xyzw[1:])
    rot_prev = Rotation.from_quat(quat_xyzw[:-1])
    delta[1:] = (rot_t * rot_prev.inv()).as_quat().astype(np.float32)
    return delta


def compute_arm_pose_deltas(abs_arm_pose: np.ndarray, episode_ends: np.ndarray) -> np.ndarray:
    """abs_arm_pose: (T,14) -> (T,14) frame-to-frame pose delta, zeroed/identity at episode starts."""
    left_dxyz = _position_delta(abs_arm_pose[:, 0:3])
    left_dquat = _orientation_delta(abs_arm_pose[:, 3:7])
    right_dxyz = _position_delta(abs_arm_pose[:, 7:10])
    right_dquat = _orientation_delta(abs_arm_pose[:, 10:14])

    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    left_dxyz[episode_starts] = 0.0
    right_dxyz[episode_starts] = 0.0
    left_dquat[episode_starts] = [0.0, 0.0, 0.0, 1.0]
    right_dquat[episode_starts] = [0.0, 0.0, 0.0, 1.0]

    return np.concatenate([left_dxyz, left_dquat, right_dxyz, right_dquat], axis=1).astype(np.float32)


def compute_hand_deltas(hand_abs: np.ndarray, episode_ends: np.ndarray) -> np.ndarray:
    """hand_abs: (T,12) -> (T,12) frame-to-frame joint-angle delta, zeroed at episode starts."""
    delta = _position_delta(hand_abs)
    episode_starts = np.concatenate([[0], episode_ends[:-1]])
    delta[episode_starts] = 0.0
    return delta.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing state/action arrays if present")
    args = parser.parse_args()

    root = zarr.open(args.zarr_path, mode="a")
    raw_state = root["data"]["state"][:]
    raw_action = root["data"]["action"][:]
    episode_ends = root["meta"]["episode_ends"][:]
    chunk_t = root["data"]["state"].chunks[0]

    fk = G1ArmFK(args.urdf)
    print(f"Computing FK for {raw_state.shape[0]} timesteps (state)...")
    arm_pose_state = fk.compute(raw_state)
    print(f"Computing FK for {raw_action.shape[0]} timesteps (action, for delta)...")
    arm_pose_action_abs = fk.compute(raw_action)

    new_state = np.concatenate([arm_pose_state, raw_state[:, 14:26]], axis=1).astype(np.float32)

    arm_pose_delta_action = compute_arm_pose_deltas(arm_pose_action_abs, episode_ends)
    hand_delta_action = compute_hand_deltas(raw_action[:, 14:26], episode_ends)
    new_action = np.concatenate([arm_pose_delta_action, hand_delta_action], axis=1).astype(np.float32)

    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.NOSHUFFLE)
    data_group = root["data"]

    # Drop the raw joint-angle state/action (already extracted above) and any legacy keys
    # from earlier script versions before writing the new state/action under the same names.
    for name in ["state", "action"] + LEGACY_KEYS:
        if name in data_group:
            del data_group[name]
            print(f"Removed key: {name}")

    arrays = {
        "state": new_state,
        "action": new_action,
    }
    for name, arr in arrays.items():
        if name in data_group:
            if not args.overwrite:
                raise ValueError(f"'{name}' already exists in {args.zarr_path}; pass --overwrite to replace it")
            del data_group[name]
        data_group.create_dataset(
            name, data=arr, chunks=(chunk_t, arr.shape[1]), compressor=compressor, dtype=arr.dtype,
        )
        print(f"Wrote {name}: shape={arr.shape}")

    print(root.tree())


if __name__ == "__main__":
    main()
