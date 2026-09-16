from __future__ import annotations

import mujoco
import numpy as np


def name_to_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"{obj_type.name} '{name}' not found in the MuJoCo model")
    return obj_id


def body_id(model: mujoco.MjModel, name: str) -> int:
    return name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def camera_id(model: mujoco.MjModel, name: str) -> int:
    return name_to_id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)


def keyframe_qpos(model: mujoco.MjModel, name: str) -> np.ndarray:
    key = name_to_id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    return np.array(model.key_qpos[key], dtype=np.float64)


def camera_resolution(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    width, height = model.cam_resolution[camera_id(model, name)]
    return int(height), int(width)


def free_joint_qpos_address(model: mujoco.MjModel, body: int) -> int:
    joint = int(model.body_jntadr[body])
    if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"body {body} does not own a free joint")
    return int(model.jnt_qposadr[joint])


def body_geom_ids(model: mujoco.MjModel, body: int) -> np.ndarray:
    start = int(model.body_geomadr[body])
    return np.arange(start, start + int(model.body_geomnum[body]), dtype=np.intp)


def bodies_in_contact(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> set[int]:
    touching: set[int] = set()
    for contact in data.contact[: data.ncon]:
        first, second = (int(model.geom_bodyid[geom]) for geom in contact.geom)
        if first == body:
            touching.add(second)
        elif second == body:
            touching.add(first)
    return touching


def yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def _joint_qpos(model: mujoco.MjModel, name: str) -> int:
    joint = name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[joint])


class JointResolver:
    def __init__(self, model: mujoco.MjModel) -> None:
        self._joints = {}
        self._ctrl = {}
        for arm in ("right", "left"):
            self._joints[arm] = (
                np.array(
                    [_joint_qpos(model, f"openarm_{arm}_joint{i}") for i in range(1, 8)],
                    dtype=np.intp,
                ),
                _joint_qpos(model, f"openarm_{arm}_finger_joint1"),
                _joint_qpos(model, f"openarm_{arm}_finger_joint2"),
            )
            self._ctrl[arm] = np.array(
                [
                    name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_joint{i}_ctrl")
                    for i in range(1, 8)
                ]
                + [name_to_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}_finger1_ctrl")],
                dtype=np.intp,
            )

    def set_qpos(self, qpos: np.ndarray, driver: np.ndarray, arm: str) -> np.ndarray:
        joints, finger, mirror = self._joints[arm]
        qpos[joints] = driver[:7]
        qpos[finger] = driver[7]
        qpos[mirror] = driver[7]
        return qpos

    def set_ctrl(self, ctrl: np.ndarray, driver: np.ndarray, arm: str) -> np.ndarray:
        ctrl[self._ctrl[arm]] = driver[:8]
        return ctrl

    def get_driver(self, qpos: np.ndarray, arm: str) -> tuple[np.ndarray, float | np.ndarray]:
        joints, finger, _ = self._joints[arm]
        return qpos[joints], qpos[finger]
