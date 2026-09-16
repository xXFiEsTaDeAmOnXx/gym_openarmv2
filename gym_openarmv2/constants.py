from __future__ import annotations

from typing import Final

# Control rate of the dataset, one action per recorded frame.
FPS: Final = 30
DT: Final = 1.0 / FPS

# Arm order in observation.state and action vectors.
ARMS: Final = ("right", "left")

JOINTS_PER_ARM: Final = 7

# Driver values per arm: 7 joints plus one gripper value.
ARM_DIM: Final = JOINTS_PER_ARM + 1
STATE_DIM: Final = ARM_DIM * len(ARMS)

# Element names of observation.state / action, in dataset order.
STATE_NAMES: Final = tuple(
    f"{arm}_{name}.pos"
    for arm in ARMS
    for name in (*(f"joint{i}" for i in range(1, JOINTS_PER_ARM + 1)), "gripper")
)

# Cell cameras, named as in the dataset image keys.
CAMERAS: Final = ("ceiling", "head_left", "head_right", "wrist_left", "wrist_right")
DEFAULT_CAMERAS: Final = ("ceiling", "head_left", "head_right", "wrist_left", "wrist_right")

# Native camera resolutions (height, width) as defined by the MJCF.
CAMERA_RESOLUTIONS: Final = {
    "ceiling": (600, 960),
    "head_left": (720, 1280),
    "head_right": (720, 1280),
    "wrist_left": (600, 960),
    "wrist_right": (600, 960),
}

MUJOCO_CAMERA_PREFIX: Final = "camera_"
TASK_DESCRIPTION: Final = "Pick up the cube."
HOME_KEYFRAME: Final = "home"

# Arm reset pose at episode start, in dataset order. Mean of the first frame of
# every dataset episode; the MJCF home keyframe is never seen during training.
RESET_ARM_POSE: Final = (
    -0.64233931,
    0.16334824,
    0.50120666,
    1.93533809,
    0.77147906,
    -0.59173910,
    0.92559504,
    -0.78541878,
    0.65011197,
    -0.19272734,
    -0.36186164,
    1.94909451,
    -0.59373608,
    0.52681484,
    -0.66823393,
    0.78539419,
)
