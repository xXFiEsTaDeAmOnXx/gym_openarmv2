from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from gym_openarm.tasks.base import Task
from gym_openarm.utils import (
    bodies_in_contact,
    body_geom_ids,
    body_id,
    free_joint_qpos_address,
    yaw_quaternion,
)

CUBE_BODY = "orange_cube"
BOX_BODY = "black_frame"
FINGER_BODY_SUFFIXES = ("ee_inner_finger", "ee_outer_finger")

GRASP_REWARD = 0.5

# Tray walls sit at +-0.065 (inner faces at +-0.06), cube half-size is 0.02.
BOX_INNER_HALF: float = 0.04
BOX_RIM_HEIGHT: float = 0.04

# Tray floor top sits 0.01 above the origin, cube half-size is 0.02.
BOX_REST_HEIGHT: float = 0.03
BOX_REST_TOL: float = 0.01
BOX_REST_SPEED: float = 0.1


class PickCubeTask(Task):
    description = "Pick up the cube and place it in the black box."

    def __init__(
        self,
        position_noise: float = 0.0,
        randomize_yaw: bool = False,
    ) -> None:
        self.position_noise = position_noise
        self.randomize_yaw = randomize_yaw

        self._cube_body = -1
        self._box_body = -1
        self._cube_qpos = -1
        self._cube_dof = -1
        self._finger_bodies: set[int] = set()
        self._resting_height = 0.0

    @property
    def xml_path(self) -> str:
        return str(Path(__file__).resolve().parent.parent / "assets" / "cell" / "demo.xml")

    def setup(self, model: mujoco.MjModel) -> None:
        self._cube_body = body_id(model, CUBE_BODY)
        self._box_body = body_id(model, BOX_BODY)
        self._cube_qpos = free_joint_qpos_address(model, self._cube_body)
        joint = int(model.body_jntadr[self._cube_body])
        self._cube_dof = int(model.jnt_dofadr[joint])
        self._finger_bodies = {
            body
            for body in range(model.nbody)
            if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body))
            and name.endswith(FINGER_BODY_SUFFIXES)
        }
        if not self._finger_bodies:
            raise ValueError("no gripper finger bodies found in the MuJoCo model")
        if len(body_geom_ids(model, self._cube_body)) == 0:
            raise ValueError(f"body '{CUBE_BODY}' has no geoms")

    def initialize_episode(
        self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator
    ) -> None:
        address = self._cube_qpos
        home = model.qpos0[address : address + 7]

        position = home[:3] + np.array(
            [*rng.uniform(-self.position_noise, self.position_noise, size=2), 0.0]
        )
        orientation = (
            yaw_quaternion(rng.uniform(-np.pi, np.pi)) if self.randomize_yaw else home[3:7]
        )

        data.qpos[address : address + 3] = position
        data.qpos[address + 3 : address + 7] = orientation
        self._resting_height = float(position[2])

    def settle(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._resting_height = float(data.xpos[self._cube_body][2])

    def get_reward(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        if self.is_success(model, data):
            return 1.0
        return GRASP_REWARD if self._is_grasped(model, data) else 0.0

    def is_success(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        cube = data.xpos[self._cube_body]
        box = data.xpos[self._box_body]
        inside_xy = (
            abs(float(cube[0] - box[0])) <= BOX_INNER_HALF
            and abs(float(cube[1] - box[1])) <= BOX_INNER_HALF
        )
        below_rim = float(cube[2]) <= float(box[2]) + BOX_RIM_HEIGHT
        resting = (
            abs(float(cube[2] - box[2]) - BOX_REST_HEIGHT) <= BOX_REST_TOL
            and float(np.linalg.norm(data.qvel[self._cube_dof : self._cube_dof + 6]))
            <= BOX_REST_SPEED
        )
        released = not self._is_grasped(model, data)
        return bool(inside_xy and below_rim and resting and released)

    def _is_grasped(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        touching = bodies_in_contact(model, data, self._cube_body)
        return len(touching & self._finger_bodies) >= 2
