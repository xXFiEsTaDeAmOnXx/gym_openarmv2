from __future__ import annotations

import abc

import mujoco
import numpy as np


class Task(abc.ABC):
    description: str

    @property
    @abc.abstractmethod
    def xml_path(self) -> str: ...

    def setup(self, model: mujoco.MjModel) -> None: ...  # noqa: B027

    def settle(self, model: mujoco.MjModel, data: mujoco.MjData) -> None: ...  # noqa: B027

    @abc.abstractmethod
    def initialize_episode(
        self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator
    ) -> None: ...

    @abc.abstractmethod
    def get_reward(self, model: mujoco.MjModel, data: mujoco.MjData) -> float: ...

    @abc.abstractmethod
    def is_success(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool: ...
