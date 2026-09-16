from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from gym_openarm.constants import (
    ARM_DIM,
    ARMS,
    CAMERAS,
    DEFAULT_CAMERAS,
    DT,
    FPS,
    HOME_KEYFRAME,
    JOINTS_PER_ARM,
    MUJOCO_CAMERA_PREFIX,
    RESET_ARM_POSE,
    STATE_DIM,
    STATE_NAMES,
)
from gym_openarm.tasks import Task, make_task
from gym_openarm.utils import (
    JointResolver,
    camera_id,
    camera_resolution,
    keyframe_qpos,
    name_to_id,
)

OBS_TYPES = ("pixels", "pixels_agent_pos")

SETTLE_STEPS = 200

# MuJoCo soft limits let a joint overshoot its range slightly.
JOINT_LIMIT_TOLERANCE = 0.1


class OpenArmEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        task: str = "pick_cube",
        obs_type: str = "pixels_agent_pos",
        cameras: Sequence[str] = DEFAULT_CAMERAS,
        observation_height: int | None = None,
        observation_width: int | None = None,
        render_mode: str = "rgb_array",
        render_camera: str = "ceiling",
        visualization_height: int = 600,
        visualization_width: int = 960,
        task_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        if obs_type not in OBS_TYPES:
            raise ValueError(f"unknown obs_type {obs_type!r}, expected one of {OBS_TYPES}")
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        unknown = set(cameras) - set(CAMERAS)
        if unknown:
            raise ValueError(f"unknown cameras {sorted(unknown)}, expected from {CAMERAS}")

        self.task = task
        self.obs_type = obs_type
        self.cameras = tuple(cameras)
        self.render_mode = render_mode
        self.render_camera = render_camera
        self.visualization_height = visualization_height
        self.visualization_width = visualization_width

        self._task: Task = make_task(task, **(task_kwargs or {}))
        self._model = mujoco.MjModel.from_xml_path(self._task.xml_path)
        self._data = mujoco.MjData(self._model)
        self._task.setup(self._model)

        self._joints = JointResolver(self._model)
        self._actuators = self._resolve_actuators()
        self._home_qpos = keyframe_qpos(self._model, HOME_KEYFRAME)
        self._home_ctrl = self._ctrl_from_qpos(self._home_qpos)
        self._n_substeps = round(DT / self._model.opt.timestep)

        self._resolutions = {
            camera: (
                (observation_height, observation_width)
                if observation_height and observation_width
                else camera_resolution(self._model, self._mujoco_camera(camera))
            )
            for camera in self.cameras
        }
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._grow_offscreen_buffer(
            [
                *self._resolutions.values(),
                (self.visualization_height, self.visualization_width),
            ]
        )

        self.action_space = self._make_action_space()
        self.observation_space = self._make_observation_space()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self._data.qpos[:] = self._home_qpos
        reset_pose = np.asarray(RESET_ARM_POSE, dtype=np.float64)
        self._joints.set_qpos(self._data.qpos, reset_pose[:ARM_DIM], "right")
        self._joints.set_qpos(self._data.qpos, reset_pose[ARM_DIM:], "left")
        self._data.qvel[:] = 0.0
        self._data.ctrl[:] = self._home_ctrl
        self._joints.set_ctrl(self._data.ctrl, reset_pose[:ARM_DIM], "right")
        self._joints.set_ctrl(self._data.ctrl, reset_pose[ARM_DIM:], "left")
        mujoco.mj_forward(self._model, self._data)

        self._task.initialize_episode(self._model, self._data, self.np_random)
        mujoco.mj_step(self._model, self._data, nstep=SETTLE_STEPS)
        self._task.settle(self._model, self._data)

        return self._get_obs(), {"is_success": False}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(STATE_DIM)
        targets = np.clip(action, self.action_space.low, self.action_space.high)
        self._data.ctrl[self._actuators] = targets
        mujoco.mj_step(self._model, self._data, nstep=self._n_substeps)

        reward = self._task.get_reward(self._model, self._data)
        is_success = self._task.is_success(self._model, self._data)
        # Truncation is handled by the TimeLimit wrapper added at registration.
        return self._get_obs(), reward, is_success, False, {"is_success": is_success}

    def render(self) -> np.ndarray:
        return self._render(
            self.render_camera, (self.visualization_height, self.visualization_width)
        )

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    @property
    def task_description(self) -> str:
        return self._task.description

    @property
    def state_names(self) -> tuple[str, ...]:
        return STATE_NAMES

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    def _resolve_actuators(self) -> np.ndarray:
        names = [
            name
            for arm in ARMS
            for name in (
                *(f"{arm}_joint{i}_ctrl" for i in range(1, JOINTS_PER_ARM + 1)),
                f"{arm}_finger1_ctrl",
            )
        ]
        return np.array(
            [name_to_id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names],
            dtype=np.intp,
        )

    def _ctrl_from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        joints = self._model.actuator_trnid[:, 0]
        ctrl = qpos[self._model.jnt_qposadr[joints]]
        return np.clip(
            ctrl, self._model.actuator_ctrlrange[:, 0], self._model.actuator_ctrlrange[:, 1]
        )

    def _make_action_space(self) -> spaces.Box:
        low, high = self._model.actuator_ctrlrange[self._actuators].T
        return spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32))

    def _make_observation_space(self) -> spaces.Space:
        images = spaces.Dict(
            {
                camera: spaces.Box(
                    low=0, high=255, shape=(*self._resolutions[camera], 3), dtype=np.uint8
                )
                for camera in self.cameras
            }
        )
        if self.obs_type == "pixels":
            return images
        return spaces.Dict(
            {
                "pixels": images,
                "agent_pos": spaces.Box(
                    low=self.action_space.low - JOINT_LIMIT_TOLERANCE,
                    high=self.action_space.high + JOINT_LIMIT_TOLERANCE,
                    dtype=np.float32,
                ),
            }
        )

    def _get_obs(self) -> dict:
        images = {
            camera: self._render(camera, self._resolutions[camera]) for camera in self.cameras
        }
        if self.obs_type == "pixels":
            return images
        return {"pixels": images, "agent_pos": self._get_agent_pos()}

    def _get_agent_pos(self) -> np.ndarray:
        state = np.empty(STATE_DIM, dtype=np.float32)
        for index, arm in enumerate(ARMS):
            joints, gripper = self._joints.get_driver(self._data.qpos, arm)
            state[index * ARM_DIM : (index + 1) * ARM_DIM] = [*joints, gripper]
        return state

    def _mujoco_camera(self, camera: str) -> str:
        return f"{MUJOCO_CAMERA_PREFIX}{camera}"

    def _grow_offscreen_buffer(self, resolutions: list[tuple[int, int]]) -> None:
        self._model.vis.global_.offheight = max(height for height, _ in resolutions)
        self._model.vis.global_.offwidth = max(width for _, width in resolutions)

    def _render(self, camera: str, resolution: tuple[int, int]) -> np.ndarray:
        height, width = resolution
        renderer = self._renderers.get(resolution)
        if renderer is None:
            renderer = mujoco.Renderer(self._model, height=height, width=width)
            self._renderers[resolution] = renderer
        renderer.update_scene(
            self._data, camera=camera_id(self._model, self._mujoco_camera(camera))
        )
        return renderer.render()
