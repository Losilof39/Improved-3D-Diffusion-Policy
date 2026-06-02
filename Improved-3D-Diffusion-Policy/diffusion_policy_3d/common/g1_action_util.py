"""
Unitree G1 joint layout helpers for iDP3 deployment.

G1 with Inspire / BrainCo hands — 26 DOF, no unused padding:
  [0:7]   left_arm   (shoulder_pitch, shoulder_roll, shoulder_yaw,
                       elbow, wrist_roll, wrist_pitch, wrist_yaw)
  [7:14]  right_arm  (same order)
  [14:20] left_hand  (6 DOF)
  [20:26] right_hand (6 DOF)

State == Action: direct 1-to-1 mapping (unlike GR1 which has unused slots).
"""

import numpy as np

G1_STATE_DIM  = 26
G1_ACTION_DIM = 26

LEFT_ARM_SLICE   = slice(0,  7)
RIGHT_ARM_SLICE  = slice(7,  14)
LEFT_HAND_SLICE  = slice(14, 20)
RIGHT_HAND_SLICE = slice(20, 26)

# Home pose in radians — update with your robot's measured rest position.
init_q26 = np.zeros(26, dtype=np.float32)


def split_action(action: np.ndarray):
    """Split a (26,) action into (left_arm, right_arm, left_hand, right_hand)."""
    return (
        action[LEFT_ARM_SLICE],
        action[RIGHT_ARM_SLICE],
        action[LEFT_HAND_SLICE],
        action[RIGHT_HAND_SLICE],
    )


def join_action(left_arm, right_arm, left_hand, right_hand) -> np.ndarray:
    """Reconstruct a (26,) action from its parts."""
    return np.concatenate([left_arm, right_arm, left_hand, right_hand]).astype(np.float32)
