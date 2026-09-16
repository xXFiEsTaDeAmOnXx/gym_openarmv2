from __future__ import annotations

import contextlib

from gymnasium.envs.registration import register

from gym_openarmv2.constants import FPS, STATE_NAMES
from gym_openarmv2.env import OpenArmEnv

__version__ = "0.1.0"

register(
    id="gym_openarmv2/OpenArmPickCube-v0",
    entry_point="gym_openarmv2.env:OpenArmEnv",
    max_episode_steps=300,
    # Rendered observations differ slightly between runs even when seeded.
    nondeterministic=True,
    kwargs={"task": "pick_cube", "obs_type": "pixels_agent_pos"},
)

with contextlib.suppress(ImportError):
    from gym_openarmv2 import lerobot_config  # noqa: F401

__all__ = ["FPS", "STATE_NAMES", "OpenArmEnv"]
